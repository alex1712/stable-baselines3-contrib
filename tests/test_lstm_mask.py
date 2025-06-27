import numpy as np
import pytest
from stable_baselines3.common.envs import IdentityEnv, IdentityEnvBox
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.envs import (
    InvalidActionEnvDiscrete,
    InvalidActionEnvMultiBinary,
    InvalidActionEnvMultiDiscrete,
)
from sb3_contrib import RecurrentMaskablePPO


def action_mask_fn(env: IdentityEnv) -> list[int]:
    assert hasattr(env, "state")
    return [int(i == env.state) for i in range(env.action_space.n)]


def test_learn_with_action_masker():
    env = IdentityEnv(dim=4)
    env = ActionMasker(env, action_mask_fn)
    model = RecurrentMaskablePPO("MlpLstmMaskPolicy", env, n_steps=8, seed=0)
    model.learn(64)
    evaluate_policy(model, env, n_eval_episodes=2, warn=False)


def test_save_load_predict(tmp_path):
    env = IdentityEnv(dim=4)
    env = ActionMasker(env, action_mask_fn)
    model = RecurrentMaskablePPO("MlpLstmMaskPolicy", env, n_steps=8, seed=1)
    model.learn(32)

    obs, _ = env.reset()
    mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)

    save_path = tmp_path / "model.zip"
    model.save(save_path)
    del model

    model = RecurrentMaskablePPO.load(save_path, env=env)
    loaded_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    assert np.allclose(action, loaded_action)


def test_env_compatibility():
    RecurrentMaskablePPO("MlpLstmMaskPolicy", InvalidActionEnvDiscrete(), n_steps=8)
    RecurrentMaskablePPO(
        "MlpLstmMaskPolicy",
        InvalidActionEnvMultiDiscrete(dims=[2, 3], n_invalid_actions=1),
        n_steps=8,
    )
    RecurrentMaskablePPO(
        "MlpLstmMaskPolicy",
        InvalidActionEnvMultiBinary(dims=3, n_invalid_actions=1),
        n_steps=8,
    )
    with pytest.raises(AssertionError):
        RecurrentMaskablePPO("MlpLstmMaskPolicy", IdentityEnvBox(), n_steps=8)
