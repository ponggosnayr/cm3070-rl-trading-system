import pytest
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Any

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from mc_engine import generate_mc_scenarios, inject_black_swan, run_mc_analysis

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal synthetic DataFrame for testing."""
    np.random.seed(42)
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n) * 10,
            "high": prices + 50,
            "low": prices - 50,
            "close": prices,
            "volume": np.random.randint(100, 1000, size=n).astype(float),
        }
    )
    return df

class TestMCGeneration:
    def test_generate_mc_scenarios_length(self, sample_df: pd.DataFrame) -> None:
        """Should generate the requested number of simulations with correct length."""
        n_sims = 10
        scenarios = generate_mc_scenarios(sample_df, n_simulations=n_sims)
        assert len(scenarios) == n_sims
        for s in scenarios:
            assert len(s) == len(sample_df)
            assert isinstance(s, pd.DataFrame)

    def test_generate_mc_scenarios_variability(self, sample_df: pd.DataFrame) -> None:
        """Scenarios should exhibit increased variance due to injected noise."""
        noise_std = 0.005
        n_sims = 10
        scenarios = generate_mc_scenarios(sample_df, n_simulations=n_sims, noise_std=noise_std)
        
        orig_log_returns = np.log(sample_df["close"] / sample_df["close"].shift(1)).dropna()
        orig_var = orig_log_returns.var()
        
        # Check statistical properties of the simulated returns
        simulated_vars = []
        for s in scenarios:
            s_log_returns = np.log(s["close"] / s["close"].shift(1)).dropna()
            simulated_vars.append(s_log_returns.var())
            
            # The start price should be exactly the same
            assert s["close"].iloc[0] == sample_df["close"].iloc[0]
            
            # Should not be identically equal to original
            assert not np.array_equal(s["close"].values, sample_df["close"].values)
            assert "sma_25" in s.columns

        # Mean variance of simulations should roughly equal orig_var + noise_std^2
        # Allow some tolerance for sample size variation
        mean_sim_var = np.mean(simulated_vars)
        expected_var = orig_var + noise_std**2
        assert expected_var * 0.5 < mean_sim_var < expected_var * 1.5

class TestBlackSwan:
    def test_inject_black_swan_impact(self, sample_df: pd.DataFrame) -> None:
        """Black swan should significantly drop the price at the crash index."""
        severity = 0.25
        perturbed_df, crash_idx = inject_black_swan(sample_df, severity=severity)

        price_before = sample_df.loc[crash_idx, "close"]
        price_after = perturbed_df.loc[crash_idx, "close"]

        # Drop should be close to severity
        drop_pct = (price_before - price_after) / price_before
        assert abs(drop_pct - severity) < 0.05

    def test_inject_black_swan_recovery_phase(self, sample_df: pd.DataFrame) -> None:
        """Price should show a recovery trend after the initial crash."""
        perturbed_df, crash_idx = inject_black_swan(sample_df, severity=0.2)
        # Recovery starts after 1 step and lasts for ~5 steps
        price_at_crash = perturbed_df.loc[crash_idx, "close"]
        # Check price 5 steps later
        recovery_idx = min(crash_idx + 5, len(perturbed_df) - 1)
        price_later = perturbed_df.loc[recovery_idx, "close"]
        assert price_later > price_at_crash

class TestMCAggregation:
    def test_run_mc_analysis_completeness(self, sample_df: pd.DataFrame) -> None:
        """Should return a comprehensive summary and all equity curves."""
        
        class MockModel:
            def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
                # Always return HOLD (action 0)
                return np.array([0]), None

        model = MockModel()
        n_sims = 5
        summary, curves = run_mc_analysis(model, sample_df, n_simulations=n_sims)

        # Verify summary keys
        expected_keys = ["mean_roi", "var_5th", "mean_drawdown", "success_rate", "max_roi", "min_roi"]
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (float, int))

        # Verify curves
        assert len(curves) == n_sims
        for curve in curves:
            assert len(curve) == len(sample_df)
            # Equity should start at 10000 by default in run_mc_analysis
            assert curve[0] == 10000.0

    def test_run_mc_analysis_with_swan(self, sample_df: pd.DataFrame) -> None:
        """Should run correctly when black swan injection is enabled."""
        class MockModel:
            def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
                return np.array([0]), None

        summary, curves = run_mc_analysis(
            MockModel(), sample_df, n_simulations=3, inject_swan=True
        )
        assert len(curves) == 3
        assert "mean_roi" in summary
