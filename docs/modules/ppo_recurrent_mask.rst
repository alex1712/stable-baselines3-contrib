.. _ppo_recurrent_mask:

.. automodule:: sb3_contrib.ppo_recurrent_mask

Recurrent Maskable PPO
======================

Implementation of invalid action masking and recurrent policies (LSTM)
for the Proximal Policy Optimization (PPO) algorithm. Other than adding
support for LSTM policies and action masking, the behaviour is the same
as in SB3's core PPO algorithm.

.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpLstmMaskPolicy
    CnnLstmMaskPolicy
    MultiInputLstmMaskPolicy


Notes
-----

- Paper (invalid action masking): https://arxiv.org/abs/2006.14171
- Blog post: https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html
- Additional Blog post: https://boring-guy.sh/posts/masking-rl/


Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ❌      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
Dict          ❌      ✔️
============= ====== ===========

.. warning::
  You must use ``MaskableEvalCallback`` from ``sb3_contrib.common.maskable.callbacks`` instead of the base ``EvalCallback`` to properly evaluate a model with action masks.
  Similarly, you must use ``evaluate_policy`` from ``sb3_contrib.common.maskable.evaluation`` instead of the SB3 one.

.. warning::
  In order to use ``SubprocVecEnv`` with ``RecurrentMaskablePPO``, you must implement the ``action_masks`` inside the environment (``ActionMasker`` cannot be used).
  You can have a look at the `built-in environments with invalid action masks <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py>`_ to have a working example.


Example
-------

Train a Recurrent Maskable PPO agent on ``InvalidActionEnvDiscrete``.  ``InvalidActionEnvDiscrete`` has an ``action_masks`` method that returns the invalid action mask (``True`` if the action is valid, ``False`` otherwise).

.. code-block:: python

  import numpy as np
  from sb3_contrib import RecurrentMaskablePPO
  from sb3_contrib.common.envs import InvalidActionEnvDiscrete
  from sb3_contrib.common.maskable.evaluation import evaluate_policy
  from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

  env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
  model = RecurrentMaskablePPO("MlpLstmMaskPolicy", env, gamma=0.4, seed=32, verbose=1)
  model.learn(5_000)

  evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

  obs, _ = env.reset()
  lstm_states = None
  episode_starts = np.ones((env.num_envs,), dtype=bool)
  while True:
      action_masks = env.action_masks()
      action, lstm_states = model.predict(
          obs,
          state=lstm_states,
          episode_start=episode_starts,
          action_masks=action_masks,
      )
      obs, reward, terminated, truncated, info = env.step(action)
      episode_starts = terminated or truncated


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

.. autoclass:: CnnLstmMaskPolicy
  :members:

.. autoclass:: MultiInputLstmMaskPolicy
  :members:

