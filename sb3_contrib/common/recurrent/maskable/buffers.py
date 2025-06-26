from collections.abc import Generator
from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import (
    MaskableRecurrentDictRolloutBufferSamples,
    MaskableRecurrentRolloutBufferSamples,
    RNNStates,
)


class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """Rollout buffer that stores LSTM states and action masks."""

    def reset(self) -> None:
        super().reset()
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int)
            mask_dims = 2 * self.action_space.n
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")
        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

    def add(self, *args, action_masks: Optional[np.ndarray] = None, lstm_states: RNNStates, **kwargs) -> None:  # type: ignore[override]
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableRecurrentRolloutBufferSamples, None, None]:
        assert self.full
        if not self.generator_ready:
            self.action_masks = self.swap_and_flatten(self.action_masks)
        yield from super().get(batch_size)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskableRecurrentRolloutBufferSamples:
        data = super()._get_samples(batch_inds, env_change)
        action_masks = self.pad(self.action_masks[batch_inds]).reshape(-1, self.mask_dims)
        action_masks = th.as_tensor(action_masks, device=self.device)
        return MaskableRecurrentRolloutBufferSamples(
            observations=data.observations,
            actions=data.actions,
            old_values=data.old_values,
            old_log_prob=data.old_log_prob,
            advantages=data.advantages,
            returns=data.returns,
            lstm_states=data.lstm_states,
            episode_starts=data.episode_starts,
            mask=data.mask,
            action_masks=action_masks,
        )


class MaskableRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """Dict version of :class:`MaskableRecurrentRolloutBuffer`."""

    def reset(self) -> None:
        super().reset()
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int)
            mask_dims = 2 * self.action_space.n
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")
        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

    def add(self, *args, action_masks: Optional[np.ndarray] = None, lstm_states: RNNStates, **kwargs) -> None:  # type: ignore[override]
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskableRecurrentDictRolloutBufferSamples:
        data = super()._get_samples(batch_inds, env_change)
        action_masks = self.pad(self.action_masks[batch_inds]).reshape(-1, self.mask_dims)
        action_masks = th.as_tensor(action_masks, device=self.device)
        return MaskableRecurrentDictRolloutBufferSamples(
            observations=data.observations,
            actions=data.actions,
            old_values=data.old_values,
            old_log_prob=data.old_log_prob,
            advantages=data.advantages,
            returns=data.returns,
            lstm_states=data.lstm_states,
            episode_starts=data.episode_starts,
            mask=data.mask,
            action_masks=action_masks,
        )
