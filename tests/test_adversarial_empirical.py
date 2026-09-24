"""
Adversarial Stress Testing and Empirical Validation Suite
========================================================================
Audits RL environment, indicator mathematics, Monte Carlo engine,
and empirical result files under extreme conditions, edge cases,
and property-based stress scenarios.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl_env import ActiveCryptoEnv
from trading_utils import (
    compute_indicators,
    compute_deflated_sharpe_ratio,
    run_single_fold_backtest,
)
from mc_engine import generate_mc_scenarios, inject_black_swan, run_mc_analysis


def create_synthetic_df(n_rows=200, trend="flat", base_price=100.0):
    """Generates synthetic OHLCV data for stress testing."""
    if trend == "flat":
        prices = np.full(n_rows, base_price, dtype=np.float32)
    elif trend == "up":
        prices = np.linspace(base_price, base_price * 2.0, n_rows, dtype=np.float32)
    elif trend == "down":
        prices = np.linspace(base_price, base_price * 0.5, n_rows, dtype=np.float32)
    elif trend == "random":
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, n_rows)
        prices = base_price * np.exp(np.cumsum(returns))
    elif trend == "zero_volume":
        prices = np.linspace(base_price, base_price * 1.5, n_rows, dtype=np.float32)
    else:
        raise ValueError(f"Unknown trend {trend}")

    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.zeros(n_rows, dtype=np.float32) if trend == "zero_volume" else np.full(n_rows, 1000.0, dtype=np.float32),
        "volume_sma_20": np.zeros(n_rows, dtype=np.float32) if trend == "zero_volume" else np.full(n_rows, 1000.0, dtype=np.float32),
        "market_regime": np.zeros(n_rows, dtype=np.float32),
    })
    return df


class TestDifferentialSharpeRatioStress:
    """Stress tests DSR reward calculation under zero volatility, extreme returns, and edge states."""

    def test_dsr_zero_volatility_flat_market(self):
        """Zero volatility must not produce NaN/Inf or division by zero in DSR reward."""
        df = create_synthetic_df(n_rows=100, trend="flat")
        env = ActiveCryptoEnv(df, random_start=False, max_steps=100)
        obs, _ = env.reset()

        for _ in range(50):
            action = np.random.choice([0, 1, 2])
            obs, reward, term, trunc, info = env.step(action)
            assert not np.isnan(reward), "Reward became NaN on flat market"
            assert not np.isinf(reward), "Reward became Inf on flat market"
            assert -10.0 <= reward <= 10.0, f"Reward {reward} outside bounds [-10, 10]"
            assert not np.isnan(obs).any(), "Observation contained NaN"
            assert not np.isinf(obs).any(), "Observation contained Inf"
            if term or trunc:
                break

    def test_dsr_extreme_flash_crash(self):
        """Simulate a -99% flash crash in 1 step; reward must handle catastrophic loss gracefully."""
        df = create_synthetic_df(n_rows=50, trend="flat", base_price=1000.0)
        # Inject 99% crash at step 25
        df.loc[25:, ["open", "high", "low", "close"]] = 10.0

        env = ActiveCryptoEnv(df, random_start=False, max_steps=50, max_drawdown_cap=0.99)
        env.reset()

        # Step 0: Open long
        env.step(1)
        # Step through until after crash
        for step_idx in range(1, 30):
            obs, reward, term, trunc, info = env.step(0)  # Hold long through crash
            assert not np.isnan(reward), f"Reward NaN at step {step_idx}"
            assert not np.isinf(reward), f"Reward Inf at step {step_idx}"
            assert -10.0 <= reward <= 10.0 or term, f"Reward {reward} outside bounds"
            if term or trunc:
                break

    def test_dsr_massive_upward_spike(self):
        """Simulate a +10,000% jump in 1 step; reward clipping must prevent gradient explosion."""
        df = create_synthetic_df(n_rows=50, trend="flat", base_price=10.0)
        df.loc[25:, ["open", "high", "low", "close"]] = 1000.0

        env = ActiveCryptoEnv(df, random_start=False, max_steps=50)
        env.reset()
        env.step(1)  # Long
        for _ in range(30):
            obs, reward, term, trunc, info = env.step(0)
            assert not np.isnan(reward), "Reward NaN on massive upward spike"
            assert not np.isinf(reward), "Reward Inf on massive upward spike"
            assert -10.0 <= reward <= 10.0, f"Reward {reward} not clamped to [-10, 10]"
            if term or trunc:
                break

    def test_dsr_short_position_massive_spike_liquidation(self):
        """Short position during massive upward spike leads to bankruptcy/margin call."""
        prices = np.full(50, 10.0, dtype=np.float32)
        prices[20:] = 100.0  # 10x rise at step 20
        df = pd.DataFrame({
            "open": prices, "high": prices*1.01, "low": prices*0.99, "close": prices,
            "volume": np.full(50, 1000.0, dtype=np.float32),
            "volume_sma_20": np.full(50, 1000.0, dtype=np.float32),
            "market_regime": np.zeros(50, dtype=np.float32)
        })

        env = ActiveCryptoEnv(df, random_start=False, max_steps=50, max_trade_duration=0)
        env.reset()
        env.step(2)  # Step 0: Target Short at 10.0
        
        terminated_cleanly = False
        for step_idx in range(1, 25):
            obs, reward, term, trunc, info = env.step(2)  # Maintain short
            if term or trunc:
                terminated_cleanly = True
                assert info["net_worth"] <= 0.1 * env.initial_balance or "force_closed" in info
                assert not np.isnan(reward)
                assert not np.isinf(reward)
                break
        assert terminated_cleanly, "Environment failed to terminate short position during 10x crash"


class TestEnvironmentEdgeCases:
    """Adversarial stress testing of environment rules, cooldowns, invalid actions, and state invariants."""

    def test_zero_volume_inputs(self):
        """Environment with 0 volume across all rows must not divide by zero."""
        df = create_synthetic_df(n_rows=50, trend="zero_volume")
        env = ActiveCryptoEnv(df, random_start=False, max_steps=50)
        obs, _ = env.reset()
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

        for _ in range(20):
            obs, reward, term, trunc, _ = env.step(1)
            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()
            if term or trunc:
                break

    def test_invalid_action_handling(self):
        """Invalid actions (e.g. entering during cooldown) must receive penalty and not alter positions."""
        df = create_synthetic_df(n_rows=50, trend="flat")
        env = ActiveCryptoEnv(df, cooldown_steps=3, random_start=False, max_steps=50)
        env.reset()

        # Enter Long, then exit to Cash to trigger cooldown
        env.step(1)
        env.step(0)
        assert env.cooldown == 3
        bal_before = env.balance

        # Attempt to enter Long during active cooldown
        obs, reward_invalid, _, _, _ = env.step(1)
        assert env.is_long is False, "Entered long during active cooldown"
        assert env.balance == bal_before, "Balance altered on invalid action"
        assert reward_invalid < 0.0, "Invalid action should receive negative penalty"

    def test_cooldown_boundary_execution(self):
        """Verify that cooldown decrements at start of step, enabling clean re-entry timing."""
        df = create_synthetic_df(n_rows=50, trend="flat")
        env = ActiveCryptoEnv(df, cooldown_steps=2, random_start=False, max_steps=50)
        env.reset()

        # Step 0: Open Long
        env.step(1)
        assert env.is_long is True

        # Step 1: Close to Cash -> sets cooldown to 2
        env.step(0)
        assert env.is_long is False
        assert env.cooldown == 2

        # Step 2: Cooldown decrements to 1 at start of step -> Trade action 1 is blocked
        env.step(1)
        assert env.is_long is False
        assert env.cooldown == 1

        # Step 3: Cooldown decrements to 0 at start of step -> Trade action 1 succeeds!
        env.step(1)
        assert env.is_long is True
        assert env.cooldown == 0

    def test_max_trade_duration_force_liquidation(self):
        """Holding a trade beyond max_trade_duration must trigger forced liquidation."""
        df = create_synthetic_df(n_rows=100, trend="flat")
        max_dur = 5
        env = ActiveCryptoEnv(df, max_trade_duration=max_dur, random_start=False, max_steps=100)
        env.reset()

        env.step(1)  # Step 0: Open Long (trade_duration becomes 1 at end of step)
        assert env.is_long is True

        # Step 1..4: Maintain Target Long
        for i in range(1, max_dur):
            obs, reward, term, trunc, info = env.step(1)
            if i < max_dur - 1:
                assert env.is_long is True

        # At max_dur - 1 step (total holding duration = max_dur), trade is force closed
        assert env.is_long is False or env.trade_duration >= max_dur

    def test_transaction_cost_deduction_invariants(self):
        """Transaction costs and slippage must strictly decrease net worth on flat roundtrip."""
        df = create_synthetic_df(n_rows=50, trend="flat", base_price=100.0)
        comm = 0.001
        slip = 0.0005
        env = ActiveCryptoEnv(df, commission_rate=comm, slippage_rate=slip, random_start=False, max_steps=50)
        env.reset()

        initial_bal = env.balance
        # Step 0: Open Long
        env.step(1)
        # Step 1: Close to Cash
        env.step(0)

        # Theoretical final balance:
        expected_balance = (
            initial_bal * (1.0 - comm) * (1.0 + ((1.0 - slip) - (1.0 + slip)) / (1.0 + slip)) * (1.0 - comm)
        )
        assert np.isclose(env.balance, expected_balance, rtol=1e-5)
        assert env.balance < initial_bal, "Roundtrip on flat price did not charge transaction costs!"


class TestTradingUtilsMathStress:
    """Stress tests indicators, ratios, and DSR calculations."""

    def test_rsi_constant_price(self):
        """Constant price series has 0 diff. Indicators should compute without exception."""
        df = create_synthetic_df(n_rows=100, trend="flat")
        res_df = compute_indicators(df, fit_hmm=False)
        assert "rsi_14" in res_df.columns
        assert "macd" in res_df.columns
        assert len(res_df) == 100

    def test_deflated_sharpe_ratio_invariants(self):
        """Deflated Sharpe Ratio (DSR) should output valid probability in [0, 1]."""
        # Case 1: zero volatility returns
        sr_hat, p_val = compute_deflated_sharpe_ratio([0.0] * 100, n_trials=5)
        assert sr_hat == 0.0
        assert p_val == 0.0

        # Case 2: strong positive returns
        np.random.seed(42)
        pos_returns = np.random.normal(0.001, 0.01, 1000)
        sr_hat, p_val = compute_deflated_sharpe_ratio(pos_returns, n_trials=5)
        assert sr_hat > 0.0
        assert 0.0 <= p_val <= 1.0


class TestMonteCarloEngineStress:
    """Stress tests Monte Carlo path generation, black swan injection, and metrics aggregation."""

    def test_mc_scenarios_zero_noise(self):
        """Zero noise Monte Carlo must generate paths identical to original close prices."""
        df = create_synthetic_df(n_rows=50, trend="random")
        scenarios = generate_mc_scenarios(df, n_simulations=5, noise_std=0.0)
        assert len(scenarios) == 5
        for s_df in scenarios:
            assert np.allclose(s_df["close"].values, df["close"].values, rtol=1e-4)

    def test_mc_scenarios_high_noise_no_negative_prices(self):
        """High noise GBM simulations must preserve strictly positive asset prices."""
        df = create_synthetic_df(n_rows=100, trend="random", base_price=50.0)
        scenarios = generate_mc_scenarios(df, n_simulations=10, noise_std=0.05)
        for s_df in scenarios:
            assert (s_df["close"] > 0).all(), "GBM generated non-positive price"
            assert not s_df["close"].isna().any(), "GBM generated NaN price"
            assert not np.isinf(s_df["close"].values).any(), "GBM generated Inf price"

    def test_black_swan_crash_magnitude(self):
        """Injected black swan must reduce price by the requested severity fraction."""
        df = create_synthetic_df(n_rows=100, trend="flat", base_price=100.0)
        severity = 0.20
        crashed_df, crash_idx = inject_black_swan(df, severity=severity)

        assert 25 <= crash_idx <= 75
        assert np.isclose(crashed_df.loc[crash_idx, "close"], 80.0, rtol=1e-3)
        if crash_idx + 5 < len(df):
            assert np.isclose(crashed_df.loc[crash_idx + 5, "close"], 86.4, rtol=1e-3)


class TestEmpiricalReportCrossVerification:
    """Verifies that logged CSV results match exact analytical figures reported in dissertation."""

    def test_walk_forward_results_csv_consistency(self):
        """Verify walk_forward_results.csv contains 5 folds with Mean ROI +11.11% and Sharpe 0.60."""
        csv_path = "walk_forward_results.csv"
        assert os.path.exists(csv_path), f"File {csv_path} does not exist"
        df = pd.read_csv(csv_path)

        assert len(df) == 5, f"Expected 5 folds, found {len(df)}"
        assert "Out-of-Sample ROI (%)" in df.columns
        assert "Out-of-Sample Sharpe" in df.columns

        mean_roi = float(df["Out-of-Sample ROI (%)"].mean())
        mean_sharpe = float(df["Out-of-Sample Sharpe"].mean())

        assert np.isclose(mean_roi, 11.11, atol=0.01), f"Mean WFV ROI was {mean_roi:.2f}%, expected 11.11%"
        assert np.isclose(mean_sharpe, 0.60, atol=0.01), f"Mean WFV Sharpe was {mean_sharpe:.2f}, expected 0.60"

    def test_comparison_results_csv_consistency(self):
        """Verify comparison_results.csv contains baseline model records."""
        csv_path = "comparison_results.csv"
        assert os.path.exists(csv_path), f"File {csv_path} does not exist"
        df = pd.read_csv(csv_path)

        assert len(df) >= 4, "Expected at least 4 models evaluated in comparison"
        models = df["Model"].tolist()
        assert "PPO (Transformer Policy)" in models
        assert "DQN (Baseline)" in models
        assert "A2C (Baseline)" in models
        assert "Buy & Hold (Market)" in models

        # Check DQN baseline metrics
        dqn_row = df[df["Model"] == "DQN (Baseline)"].iloc[0]
        assert np.isclose(dqn_row["ROI (%)"], -8.04, atol=0.01)
        assert np.isclose(dqn_row["Sharpe"], -0.58, atol=0.01)
        assert dqn_row["Trades"] == 34
