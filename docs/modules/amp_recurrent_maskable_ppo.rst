.. _amp_recurrent_maskable_ppo:

.. automodule:: sb3_contrib.ppo_recurrent_mask.amp_recurrent_maskable_ppo

AMPRecurrentMaskablePPO
=======================

Automatic mixed precision version of :class:`~sb3_contrib.ppo_recurrent_mask.RecurrentMaskablePPO`.
Other than using AMP during training, the behaviour is the same as in the base algorithm.

Example
-------

.. code-block:: python

  from sb3_contrib import AMPRecurrentMaskablePPO
  from sb3_contrib.common.envs import InvalidActionEnvDiscrete

  env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
  model = AMPRecurrentMaskablePPO("MlpLstmMaskPolicy", env, verbose=1)
  model.learn(5_000)

Parameters
----------

.. autoclass:: AMPRecurrentMaskablePPO
  :members:
  :inherited-members:

