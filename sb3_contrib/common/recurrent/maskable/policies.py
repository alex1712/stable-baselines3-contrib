from typing import Optional

import numpy as np
import torch as th

from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
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


class MaskableRecurrentActorCriticCnnPolicy(MaskableRecurrentActorCriticPolicy, RecurrentActorCriticCnnPolicy):
    pass


class MaskableRecurrentMultiInputActorCriticPolicy(MaskableRecurrentActorCriticPolicy, RecurrentMultiInputActorCriticPolicy):
    pass
