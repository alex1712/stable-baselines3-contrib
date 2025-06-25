import random

import gymnasium as gym
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import FakeImageEnv, IdentityEnv, IdentityEnvBox
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib import MaskableRecurrentPPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import is_masking_supported
from sb3_contrib.common.wrappers import ActionMasker


def make_env():
    return InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)


class ToDictWrapper(gym.Wrapper):
    """Simple wrapper to test MultiInputPolicy on Dict observations."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"obs": self.env.observation_space})

    def reset(self, **kwargs):
        return {"obs": self.env.reset(**kwargs)[0]}, {}

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        return {"obs": obs}, reward, terminated, truncated, infos


@pytest.mark.parametrize("vec_env_cls", [SubprocVecEnv, DummyVecEnv])
def test_supports_multi_envs(vec_env_cls):
    env = make_vec_env(make_env, n_envs=2, vec_env_cls=vec_env_cls)
    assert is_masking_supported(env)
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, n_steps=16, seed=32)
    model.learn(32)
    evaluate_policy(model, env, warn=False)

    env = make_vec_env(IdentityEnv, n_envs=2, env_kwargs={"dim": 2}, vec_env_cls=vec_env_cls)
    assert not is_masking_supported(env)
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, n_steps=16, seed=32)
    with pytest.raises(ValueError):
        model.learn(32)
    with pytest.raises(ValueError):
        evaluate_policy(model, env, warn=False)
    model.learn(32, use_masking=False)
    evaluate_policy(model, env, warn=False, use_masking=False)


def test_maskable_policy_required():
    env = make_env()
    with pytest.raises(ValueError):
        MaskableRecurrentPPO(ActorCriticPolicy, env)


def test_discrete_action_space_required():
    env = IdentityEnvBox()
    with pytest.raises(AssertionError, match="The algorithm only supports"):
        MaskableRecurrentPPO("MlpLstmPolicy", env)


@pytest.mark.parametrize("share_features_extractor", [True, False])
def test_cnn(share_features_extractor):
    def action_mask_fn(env):
        random_invalid_action = random.randrange(env.action_space.n)
        return [i != random_invalid_action for i in range(env.action_space.n)]

    env = FakeImageEnv()
    env = ActionMasker(env, action_mask_fn)

    model = MaskableRecurrentPPO(
        "CnnLstmPolicy",
        env,
        n_steps=16,
        seed=32,
        policy_kwargs=dict(
            features_extractor_kwargs=dict(features_dim=32),
            share_features_extractor=share_features_extractor,
        ),
    )
    model.learn(32)
    evaluate_policy(model, env, warn=False)


def test_dict_obs():
    env = make_env()
    env = ToDictWrapper(env)
    model = MaskableRecurrentPPO("MultiInputLstmPolicy", env, n_steps=16, seed=8)
    model.learn(16)
    evaluate_policy(model, env, warn=False)

    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    env = ToDictWrapper(env)
    model = MaskableRecurrentPPO("MultiInputLstmPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)
