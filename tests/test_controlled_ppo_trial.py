import numpy as np
import pandas as pd

from controlled_ppo_trial import evaluate, make_env


def prices():
    # Deliberate gaps distinguish next-open execution from current-close execution.
    close = np.array([100., 105., 98., 108., 110.])
    return pd.DataFrame(dict(open=[100., 102., 103., 99., 109.],
                             high=close + 5, low=close - 5, close=close,
                             volume=np.ones(5)*100, market_regime=np.zeros(5)))


def test_buy_hold_charges_entry_and_terminal_exit_at_correct_prices():
    result = evaluate(prices(), constant=1)
    expected = 100 * (110 * (1-.0005) / (102 * (1+.0005)) * (1-.001)**2 - 1)
    assert np.isclose(result['roi'], expected, atol=1e-5)


def test_plain_reward_sums_to_realized_log_wealth():
    env = make_env(prices(), plain=True)
    env.reset()
    reward_sum = 0
    for action in [1, 0, 2, 0]:
        _, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
    assert terminated or truncated
    assert np.isclose(reward_sum, 10*np.log(env.unwrapped.net_worth/10000))


def test_cash_baseline_is_zero():
    result = evaluate(prices(), constant=0)
    assert result['roi'] == result['drawdown'] == result['score'] == 0


def test_backtest_initial_stack_matches_training():
    from trading_utils import run_single_fold_backtest
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    class CaptureCash:
        first = None

        def predict(self, obs, deterministic=True):
            if self.first is None:
                self.first = obs.copy()
            return np.array([0]), None

    model = CaptureCash()
    run_single_fold_backtest(model, prices(), max_trade_duration=0)
    env = VecFrameStack(DummyVecEnv([lambda: make_env(prices())]), n_stack=8)
    np.testing.assert_array_equal(model.first, env.reset())
    env.close()
