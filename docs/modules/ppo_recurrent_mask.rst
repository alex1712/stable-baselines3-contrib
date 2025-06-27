.. _ppo_recurrent_mask:

.. automodule:: sb3_contrib.ppo_recurrent_mask

Recurrent Maskable PPO
======================

Implementation of Proximal Policy Optimization (PPO) with support for recurrent policies (LSTM)
and invalid action masking. Other than those additions, the behaviour is the same as in SB3's core PPO algorithm.

.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpLstmMaskPolicy
    CnnLstmMaskPolicy
    MultiInputLstmMaskPolicy

Example
-------

.. code-block:: python

  import numpy as np
  from sb3_contrib import RecurrentMaskablePPO
  from sb3_contrib.common.envs import InvalidActionEnvDiscrete
  from sb3_contrib.common.maskable.utils import get_action_masks

  env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
  model = RecurrentMaskablePPO("MlpLstmMaskPolicy", env, verbose=1)
  model.learn(5_000)

  obs, _ = env.reset()
  lstm_states = None
  episode_starts = np.ones((1,), dtype=bool)
  while True:
      action_masks = get_action_masks(env)
      action, lstm_states = model.predict(
          obs,
          state=lstm_states,
          episode_start=episode_starts,
          action_masks=action_masks,
          deterministic=True,
      )
      obs, reward, terminated, truncated, info = env.step(action)
      episode_starts = np.array([terminated or truncated])

Parameters
----------

.. autoclass:: RecurrentMaskablePPO
  :members:
  :inherited-members:

.. _ppo_recurrent_mask_policies:

RecurrentMaskablePPO Policies
-----------------------------

.. autoclass:: MlpLstmMaskPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.common.recurrent_maskable.policies.RecurrentMaskableActorCriticPolicy
  :members:
  :noindex:

.. autoclass:: CnnLstmMaskPolicy
  :members:

.. autoclass:: sb3_contrib.common.recurrent_maskable.policies.RecurrentMaskableActorCriticCnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputLstmMaskPolicy
  :members:

.. autoclass:: sb3_contrib.common.recurrent_maskable.policies.RecurrentMaskableMultiInputActorCriticPolicy
  :members:
  :noindex:
