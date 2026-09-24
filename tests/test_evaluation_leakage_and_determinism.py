import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.trading_utils import (
    compute_indicators,
    fit_market_regimes_hmm,
    predict_market_regimes_hmm,
    compute_market_regimes
)

class TestEvaluationIntegrity:
    def test_future_price_changes_do_not_leak_into_train_features(self):
        """Modifying future/test prices must not alter earlier causal indicators or fitted training parameters."""
        # Create base series
        np.random.seed(42)
        n = 200
        prices = 100.0 + np.cumsum(np.random.randn(n))
        df_a = pd.DataFrame({
            "close": prices,
            "volume": np.random.uniform(100, 500, n)
        })
        
        # Train partition is first 120 rows; test partition is next 80 rows
        train_len = 120
        train_df_a = df_a.iloc[:train_len].copy()
        
        # Scenario B: Future prices (rows 120..200) are drastically different (e.g. 5x spike)
        df_b = df_a.copy()
        df_b.loc[train_len:, "close"] = df_b.loc[train_len:, "close"] * 5.0
        train_df_b = df_b.iloc[:train_len].copy()
        
        # Fit HMM on train partition
        hmm_a, map_a = fit_market_regimes_hmm(train_df_a)
        hmm_b, map_b = fit_market_regimes_hmm(train_df_b)
        
        # 1. Fitted HMM parameters must be strictly identical
        np.testing.assert_array_almost_equal(hmm_a.means_, hmm_b.means_, decimal=6)
        np.testing.assert_array_almost_equal(hmm_a.transmat_, hmm_b.transmat_, decimal=6)
        
        # 2. Causal indicators on training partition must be strictly identical
        ind_a = compute_indicators(train_df_a, hmm_model=hmm_a, state_map=map_a)
        ind_b = compute_indicators(train_df_b, hmm_model=hmm_b, state_map=map_b)
        
        for col in ["sma_7", "sma_25", "sma_99", "rsi_14", "macd", "market_regime"]:
            np.testing.assert_array_almost_equal(ind_a[col].values, ind_b[col].values, decimal=5)

    def test_evaluation_never_implicitly_fits_hmm_when_artifact_missing(self, tmp_path):
        """Evaluation with fit_hmm=False must raise FileNotFoundError if artifact is absent, never fit implicitly."""
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 101.5] * 20,
            "volume": [10.0, 12.0, 11.0, 10.5] * 20
        })
        non_existent_path = str(tmp_path / "absent_hmm.pkl")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            compute_indicators(df, fit_hmm=False, hmm_artifact_path=non_existent_path)
            
        assert "Required HMM artifact not found" in str(exc_info.value)
        assert not os.path.exists(non_existent_path)

    def test_hmm_inference_is_strictly_deterministic(self):
        """Multiple inference runs on the same input using a fixed model must return identical results (no random noise)."""
        df = pd.DataFrame({
            "close": [100.0 + i * 0.5 for i in range(100)],
            "volume": [50.0 + (i % 5) * 10 for i in range(100)]
        })
        hmm_model, state_map = fit_market_regimes_hmm(df)
        
        regime_run1 = predict_market_regimes_hmm(df, hmm_model, state_map)
        regime_run2 = predict_market_regimes_hmm(df, hmm_model, state_map)
        
        np.testing.assert_array_equal(regime_run1, regime_run2)
