import pytest
import numpy as np
import pandas as pd
import os
import sys
from typing import List, Dict, Optional, Any

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from xai_engine import XAIEngine

@pytest.fixture
def xai_engine() -> XAIEngine:
    """Create a standard XAI engine instance for testing."""
    return XAIEngine(
        coin_ticker="BTC",
        current_price=50000.0,
        rsi=45.0,
        macd=0.001,
        volume_ratio=1.2,
        action_name="BUY (LONG)",
        confidence=85.0,
        probs=[0.05, 0.85, 0.05, 0.02, 0.03],
        shap_values=None,
    )

class TestIntentClassification:
    @pytest.mark.parametrize("query, expected_intent", [
        ("Why did you trade?", "rationale"),
        ("explain your logic", "rationale"),
        ("How are you doing?", "performance"),
        ("show pnl", "performance"),
        ("is it risky?", "risk"),
        ("what is the drawdown", "risk"),
        ("explain rsi", "explain_rsi"),
        ("what is macd?", "explain_macd"),
        ("hello", "general"),
        ("who are you", "general"),
    ])
    def test_classify_intent(self, xai_engine: XAIEngine, query: str, expected_intent: str) -> None:
        """Should correctly classify various user queries."""
        assert xai_engine.classify_intent(query) == expected_intent

class TestExplanationGeneration:
    def test_generate_rationale_no_shap(self, xai_engine: XAIEngine) -> None:
        """Should provide a grounded fallback explanation when SHAP is missing."""
        resp = xai_engine.generate_response("why?")
        assert "Advisor Recommendation" in resp
        assert "RSI" in resp
        assert "50,000.00" in resp
        # 'confidence' is in the default response, not the rationale response
        # Indicators must not be presented as verified causes without SHAP.
        assert "reasons cannot be verified" in resp.lower()

    def test_generate_risk_msg(self, xai_engine: XAIEngine) -> None:
        """Should incorporate backtest metrics into risk assessment."""
        metrics = {"Max DD %": -12.5, "Sharpe": 1.8}
        resp = xai_engine.generate_response("is it safe?", backtest_metrics=metrics)
        assert "Risk Assessment" in resp
        assert "12.50%" in resp
        # 'Sharpe' is in performance msg, not risk msg. 
        # Risk msg uses Max DD.
        assert "volatility" in resp.lower()

    def test_generate_confidence_msg(self, xai_engine: XAIEngine) -> None:
        """Should explain confidence scores and probability gaps."""
        resp = xai_engine.generate_response("how sure are you?")
        assert "Confidence Analysis" in resp
        assert "85.0%" in resp
        assert "gap" in resp.lower()

class TestSHAPGrounding:
    def test_get_top_shap_features(self, xai_engine: XAIEngine) -> None:
        """Should extract top features correctly from dummy SHAP values."""
        # Mock SHAP values (5 classes, 1 sample, 136 features)
        dummy_shap = [np.zeros((1, 136)) for _ in range(5)]
        dummy_shap[1][0, 0] = 0.5   # Feature 0 (Price vs 25-Avg)
        dummy_shap[1][0, 1] = -0.3  # Feature 1 (RSI)
        dummy_shap[1][0, 17] = 0.2  # Feature 0 (at step t-1, stride 17)

        # Force feature names to be 12 items so num_features=12
        xai_engine.feature_names = ['Price vs 25-Avg', 'RSI (Momentum)', 'MACD (Trend)', 'Position Status', 'Unrealized PnL', 'Trade Cooldown', '1H Trend', 'Market Regime', '2H Trend', '3H Trend', 'Trading Volume', 'Trade Duration']
        xai_engine.shap_values = dummy_shap
        top_features = xai_engine._get_top_shap_features(top_n=2)

        assert len(top_features) == 2
        assert top_features[0]["feature"] == "Price vs 25-Avg"
        assert top_features[0]["direction"] == "positive"
        assert abs(top_features[0]["impact"] - 0.7) < 1e-5
        assert top_features[1]["feature"] == "RSI (Momentum)"
        assert top_features[1]["direction"] == "negative"
        assert abs(top_features[1]["impact"] - (-0.3)) < 1e-5

    def test_rationale_with_shap(self, xai_engine: XAIEngine) -> None:
        """Rationale should explicitly mention SHAP-identified features."""
        dummy_shap = [np.zeros((1, 136)) for _ in range(5)]
        dummy_shap[1][0, 0] = 0.5  # Feature 0
        xai_engine.feature_names = ['Price vs 25-Avg', 'RSI (Momentum)', 'MACD (Trend)', 'Position Status', 'Unrealized PnL', 'Trade Cooldown', '1H Trend', 'Market Regime', '2H Trend', '3H Trend', 'Trading Volume', 'Trade Duration']
        xai_engine.shap_values = dummy_shap

        resp = xai_engine.generate_response("why did you buy?")
        assert "Price vs 25-Avg" in resp
        assert "pushed me" in resp.lower()
        # Changed from 'positive impact' to 'pushed me positively' to match implementation
        assert "pushed me positively" in resp.lower()

    def test_get_top_shap_features_aligned(self) -> None:
        """Should extract top features correctly when features are properly aligned (17 dimensions)."""
        feature_names = [
            "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17"
        ]
        # Mock SHAP values (5 classes, 1 sample, 136 features)
        dummy_shap = [np.zeros((1, 136)) for _ in range(5)]
        dummy_shap[1][0, 0] = 0.5   # f1 at step 0
        dummy_shap[1][0, 17] = 0.2  # f1 at step 1 (17-stride)
        dummy_shap[1][0, 1] = -0.4  # f2 at step 0
        
        xai = XAIEngine(
            coin_ticker="BTC",
            current_price=50000.0,
            rsi=45.0,
            macd=0.001,
            volume_ratio=1.2,
            action_name="BUY (LONG)",
            confidence=85.0,
            probs=[0.05, 0.85, 0.05, 0.02, 0.03],
            shap_values=dummy_shap,
            feature_names=feature_names,
        )
        
        top_features = xai._get_top_shap_features(top_n=2)
        assert len(top_features) == 2
        # f1 total should be 0.5 + 0.2 = 0.7 (highest magnitude)
        # f2 total should be -0.4
        assert top_features[0]["feature"] == "f1"
        assert abs(top_features[0]["impact"] - 0.7) < 1e-5
        assert top_features[0]["direction"] == "positive"
        assert top_features[1]["feature"] == "f2"
        assert abs(top_features[1]["impact"] - (-0.4)) < 1e-5
        assert top_features[1]["direction"] == "negative"

class TestXAIFidelityAudit:
    def test_xai_explanation_fidelity_rho(self) -> None:
        """Verify that natural language explanation rank correlates with SHAP ranks (rho >= 0.95)."""
        from xai_audit import audit_explanation_fidelity
        res = audit_explanation_fidelity(num_samples=50)
        assert res["pass_audit"] is True, f"Fidelity score failed target: mean rho = {res['mean_spearman_rho']:.4f} < 0.95"
        assert res["mean_spearman_rho"] >= 0.95

