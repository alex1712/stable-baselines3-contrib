from stable_baselines3.common.envs import IdentityEnv

from sb3_contrib import AMPRecurrentMaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker


def action_mask_fn(env: IdentityEnv) -> list[int]:
    assert hasattr(env, "state")
    return [int(i == env.state) for i in range(env.action_space.n)]


def test_amp_learn():
    env = IdentityEnv(dim=4)
    env = ActionMasker(env, action_mask_fn)
    model = AMPRecurrentMaskablePPO("MlpLstmMaskPolicy", env, n_steps=8, seed=0)
    model.learn(64)
    evaluate_policy(model, env, n_eval_episodes=2, warn=False)


def test_amp_env_compatibility():
    AMPRecurrentMaskablePPO("MlpLstmMaskPolicy", InvalidActionEnvDiscrete(), n_steps=8)
