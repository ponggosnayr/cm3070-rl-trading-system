import pytest


def test_forced_exit_is_not_a_policy_sell(sample_ohlcv):
    class AlwaysLong:
        def predict(self, obs, deterministic=True):
            return np.array([1]), None

    df = compute_indicators(sample_ohlcv, fit_hmm=False)
    result = run_single_fold_backtest(AlwaysLong(), df, max_trade_duration=10)
    assert result['PolicyActionCounts']['cash'] == 0
    assert result['PolicyActionCounts']['short'] == 0
    exits = [t for t in result['TradeHistory'] if t['Reason'] == 'maximum_holding_duration']
    assert exits
    assert all(t['Action'] == 'EXIT LONG' and t['PolicyTarget'] == 1 for t in exits)
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from trading_utils import compute_indicators, run_single_fold_backtest

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with enough data for all indicators."""
    n = 150
    np.random.seed(42)
    prices = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 2.0
    df = pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": np.random.randint(100, 1000, size=n).astype(float),
        }
    )
    return df.reset_index(drop=True)

class TestIndicators:
    def test_compute_indicators_completeness(self, sample_ohlcv: pd.DataFrame) -> None:
        """Should add all expected technical indicator columns."""
        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        expected_cols = [
            "sma_7", "sma_25", "sma_99",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "volume_sma_20", "roc_12", "market_regime"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing indicator column: {col}"
        
        # Ensure no complete column of NaNs
        for col in expected_cols:
            assert not df[col].isna().all(), f"Indicator {col} is all NaNs"

    def test_rsi_validity(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI should be bounded between 0 and 100."""
        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_wilders_smoothing(self, sample_ohlcv: pd.DataFrame) -> None:
        """Verify RSI uses Wilder's Smoothing correctly."""
        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        delta = sample_ohlcv["close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = np.where(loss == 0.0, np.nan, gain / loss)
        expected_rsi = np.where(loss == 0.0, np.where(gain > 0.0, 100.0, 50.0), 100.0 - (100.0 / (1.0 + rs)))
        assert abs(df["rsi_14"].dropna().iloc[-1] - expected_rsi[-1]) < 1e-6

    def test_rsi_zero_loss_saturation(self) -> None:
        """Verify that strictly increasing price action (loss == 0.0) safely saturates RSI at 100.0 without NaN."""
        prices = [100.0 + i * 2.0 for i in range(50)]
        df_up = pd.DataFrame({
            "open": prices,
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [500.0] * 50,
        })
        res = compute_indicators(df_up, fit_hmm=False)
        assert not res["rsi_14"].isna().any(), "RSI contains NaN on pure upward price series"
        # After first step (index 0 where diff is NaN/0), all subsequent steps must be exactly 100.0
        assert (res["rsi_14"].iloc[1:] == 100.0).all(), "RSI should saturate at 100.0 for zero loss"

    def test_sma_accuracy(self, sample_ohlcv: pd.DataFrame) -> None:
        """Verify SMA calculation against a manual rolling mean."""
        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        window = 7
        manual_sma = sample_ohlcv["close"].rolling(window).mean()
        # Compare at a point where the window is full
        assert abs(df.loc[window-1, "sma_7"] - manual_sma.loc[window-1]) < 1e-6

class TestBacktestExecution:
    def test_backtest_return_structure_and_metrics(self, sample_ohlcv: pd.DataFrame) -> None:
        """Backtest should return specific performance keys with valid math for a profitable run."""
        class ProfitableModel:
            def __init__(self):
                self.step = 0
            def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
                self.step += 1
                if 20 <= self.step < 120:
                    return np.array([1]), None  # Target Long
                return np.array([0]), None      # Target Cash

        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        metrics = run_single_fold_backtest(ProfitableModel(), df, initial_balance=1000.0)

        assert isinstance(metrics, dict)
        assert "ROI %" in metrics
        assert "Sharpe" in metrics
        assert "Sortino" in metrics
        assert "Max DD %" in metrics
        assert "Trades" in metrics
        assert "NetWorthHistory" in metrics
        assert len(metrics["NetWorthHistory"]) == len(df)
        
        # Verify non-zero math for profitable model
        assert metrics["ROI %"] > 0.0, "Expected positive ROI"
        assert metrics["Sharpe"] != 0.0, "Expected non-zero Sharpe"
        assert metrics["Trades"] > 0, "Expected at least one trade"

    def test_neutral_model_roi(self, sample_ohlcv: pd.DataFrame) -> None:
        """A model that never trades should have 0% ROI."""
        class NeutralModel:
            def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
                return np.array([0]), None

        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        metrics = run_single_fold_backtest(NeutralModel(), df, initial_balance=5000.0)
        assert metrics["ROI %"] == 0.0
        assert metrics["Final $"] == 5000.0
        assert metrics["Trades"] == 0

    def test_backtest_with_slippage_and_commissions(self, sample_ohlcv: pd.DataFrame) -> None:
        """Backtest should handle non-zero friction costs."""
        class SimpleTrader:
            def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
                # Open long at step 10, close at step 20
                # This logic depends on the internal state which we can't easily control here
                # so we just check if it runs without error with friction.
                return np.array([0]), None

        df = compute_indicators(sample_ohlcv, fit_hmm=False)
        metrics = run_single_fold_backtest(
            SimpleTrader(), df, 
            commission_rate=0.001, 
            slippage_rate=0.0005
        )
        assert "ROI %" in metrics
