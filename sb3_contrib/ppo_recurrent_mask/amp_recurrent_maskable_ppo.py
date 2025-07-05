from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from torch.cuda.amp import GradScaler
from torch import autocast

from stable_baselines3.common.utils import explained_variance
from sb3_contrib.ppo_recurrent_mask.ppo_recurrent_mask import RecurrentMaskablePPO

SelfAMPRecurrentMaskablePPO = TypeVar("SelfAMPRecurrentMaskablePPO", bound="AMPRecurrentMaskablePPO")


class AMPRecurrentMaskablePPO(RecurrentMaskablePPO):
    """RecurrentMaskablePPO trained using automatic mixed precision (AMP)."""

    def __init__(
        self,
        *args: Any,
        scaler: GradScaler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if scaler is None:
            scaler = GradScaler(enabled=self.device.type == "cuda")
        self.scaler = scaler

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer using AMP."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                        action_masks=rollout_data.action_masks,
                    )
                    values = values.flatten()

                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[mask])
                    else:
                        entropy_loss = -th.mean(entropy[mask])

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                entropy_losses.append(entropy_loss.item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.policy.optimizer)
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.policy.optimizer)
                self.scaler.update()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
