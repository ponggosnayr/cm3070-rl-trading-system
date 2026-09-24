import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from controlled_ppo_trial import make_env
from ppo_learning_diagnostic import synthetic, market_stacks


def test_synthetic_has_continuous_execution_prices_and_both_trends():
    df = synthetic(123,256)
    np.testing.assert_array_equal(df.open.to_numpy()[1:],df.close.to_numpy()[:-1])
    assert (df.close>df.open).any() and (df.close<df.open).any()
    assert df.market_regime.eq(0).all()  # No privileged trend label in observations.


def test_probe_features_match_policy_observations():
    df=synthetic(7,120)
    x=market_stacks(df)
    env=VecFrameStack(DummyVecEnv([lambda:make_env(df)]),n_stack=8)
    obs=env.reset()
    for i in range(12):
        np.testing.assert_array_equal(x[i],obs.reshape(8,17)[:,:13].reshape(-1))
        obs,_,_,_=env.step([0])
    env.close()
