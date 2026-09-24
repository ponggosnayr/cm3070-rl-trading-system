import sys
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/experiments'))
import run_tiny_vs_deep as experiment


def test_terminal_loss_and_trades_survive_vecenv_reset(monkeypatch, tmp_path):
    class TerminalEnv(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed=None, options=None):
            self.net_worth, self.trade_count = 10000., 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.net_worth, self.trade_count = 9000., 2
            return np.zeros(4, dtype=np.float32), 0., True, False, {
                'net_worth': 9000., 'episode_metrics': {'trades': 2, 'final_capital': 9000.}}

    class Policy:
        def predict(self, obs, deterministic=True):
            return np.array([1]), None

    monkeypatch.setattr(experiment, 'ActiveCryptoEnv', TerminalEnv)
    path = tmp_path / 'trace.csv'
    result = experiment.evaluate_on_window(Policy(), None, pd.DataFrame({'close': [100., 90.]}), trace_path=path)
    assert result['roi_pct'] == -10.
    assert result['max_drawdown_pct'] == 10.
    assert result['trades'] == 2
    assert pd.read_csv(path).net_worth.iloc[-1] == 9000.
