from functools import partial
from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.maskable.distributions import (
    MaskableDistribution,
    make_masked_proba_distribution,
)
from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class MaskableRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """Recurrent policy with support for invalid action masking."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Override the action distribution with maskable version
        self.action_dist = make_masked_proba_distribution(self.action_space)
        self._build(self._dummy_schedule)  # rebuild networks
        self.optimizer = self.optimizer_class(self.parameters(), lr=self._dummy_schedule(1), **self.optimizer_kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Create the networks and the optimizer."""
        self._build_mlp_extractor()

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        actions, values, log_prob, lstm_states = super().forward(obs, lstm_states, episode_starts, deterministic)
        if action_masks is not None:
            self.action_dist.apply_masking(action_masks)
            actions = self.action_dist.get_actions(deterministic=deterministic)
            log_prob = self.action_dist.log_prob(actions)
        return actions, values, log_prob, lstm_states

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        values, log_prob, entropy = super().evaluate_actions(obs, actions, lstm_states, episode_starts)
        if action_masks is not None:
            self.action_dist.apply_masking(action_masks)
            log_prob = self.action_dist.log_prob(actions)
            entropy = self.action_dist.entropy()
        return values, log_prob, entropy

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts, action_masks)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.distributions.Distribution, tuple[th.Tensor, ...]]:
        distribution, lstm_states = super().get_distribution(obs, lstm_states, episode_starts)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        if state is None:
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)


class MaskableRecurrentActorCriticCnnPolicy(MaskableRecurrentActorCriticPolicy, RecurrentActorCriticCnnPolicy):
    pass


class MaskableRecurrentMultiInputActorCriticPolicy(MaskableRecurrentActorCriticPolicy, RecurrentMultiInputActorCriticPolicy):
    pass
