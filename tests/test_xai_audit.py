import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.xai_audit import compute_spearman_rank_correlation, audit_explanation_fidelity, audit_real_model_fidelity, audit_all_actions_fidelity
from src.xai_engine import XAIEngine
from api import validate_chat_response


class TestXAIAuditMetrics:
    def test_all_features_missing_fails(self):
        """All features missing (9999 sentinel) must return exactly 0.0 correlation."""
        ground_truth = [0, 1, 2]
        all_missing = [9999, 9999, 9999]
        rho = compute_spearman_rank_correlation(ground_truth, all_missing)
        assert rho == 0.0

    def test_one_missing_feature_penalized(self):
        """One missing feature out of 3 should produce a valid, penalized score without NaN."""
        ground_truth = [0, 1, 2]
        partial_missing = [0, 1, 9999]
        rho = compute_spearman_rank_correlation(ground_truth, partial_missing)
        assert not np.isnan(rho)
        penalized_rho = rho * (1.0 - 1.0 / 3.0)
        assert penalized_rho < 1.0
        assert not np.isnan(penalized_rho)

    def test_ties_handled_without_nan(self):
        """Tied rankings must be resolved with fractional mid-ranks via scipy.stats.spearmanr without NaN."""
        ground_truth = [0, 1, 2]
        tied_ranks = [1, 1, 2]
        rho = compute_spearman_rank_correlation(ground_truth, tied_ranks)
        assert not np.isnan(rho)
        assert -1.0 <= rho <= 1.0

    def test_constant_arrays_return_zero(self):
        """Constant rank arrays have zero variance and must return 0.0 instead of NaN or 1.0."""
        assert compute_spearman_rank_correlation([1, 1, 1], [1, 1, 1]) == 0.0
        assert compute_spearman_rank_correlation([0, 1, 2], [5, 5, 5]) == 0.0
        assert compute_spearman_rank_correlation([0], [0]) == 0.0

    def test_incompatible_action_dimensions(self):
        """XAIEngine should handle out-of-bounds or unexpected action indices gracefully."""
        engine = XAIEngine(
            coin_ticker="BTC",
            current_price=50000.0,
            rsi=45.0,
            macd=0.001,
            volume_ratio=1.2,
            action_name="UNKNOWN_ACTION_999",
            confidence=85.0,
            probs=[0.34, 0.33, 0.33],
            shap_values=[np.zeros((1, 136)) for _ in range(3)]
        )
        resp = engine.generate_response("why did you trade?")
        assert "Advisor Recommendation" in resp

    def test_opposite_action_responses_flagged(self):
        """Validation must reject responses advocating actions opposite to the model prediction."""
        is_valid, reason = validate_chat_response(
            "Bitcoin looks great, buy now and enter a long position immediately.",
            action_name="SELL (SHORT)",
            price=50000.0,
            ticker="BTC"
        )
        assert not is_valid
        assert "Action conflict" in reason

        is_valid, reason = validate_chat_response(
            "Downside momentum is strong, open a short to profit from falling prices.",
            action_name="BUY (LONG)",
            price=50000.0,
            ticker="BTC"
        )
        assert not is_valid
        assert "Action conflict" in reason

    def test_inverted_factual_claims_flagged(self):
        """Validation must reject responses citing prices deviating wildly from current live price."""
        is_valid, reason = validate_chat_response(
            "Bitcoin is currently trading at $95,000.00 which is near all time highs.",
            action_name="BUY (LONG)",
            price=50000.0,
            ticker="BTC",
            tolerance_pct=30.0
        )
        assert not is_valid
        assert "Price conflict" in reason

    def test_unavailable_shap_explicitly_disclosed(self):
        """When SHAP values are None, explanation must explicitly disclose limited technical indicators."""
        engine = XAIEngine(
            coin_ticker="BTC",
            current_price=50000.0,
            rsi=55.0,
            macd=0.002,
            volume_ratio=1.1,
            action_name="BUY (LONG)",
            confidence=80.0,
            probs=[0.1, 0.8, 0.1],
            shap_values=None
        )
        resp = engine.generate_response("why did you make this trade?")
        assert "unavailable" in resp.lower() or "technical indicator" in resp.lower()

    def test_audit_real_model_fidelity(self):
        """Evaluate explanation fidelity on real project checkpoint inferences across Cash, Long, Short."""
        res = audit_real_model_fidelity(num_eval_states=5)
        assert res["status"] in ("completed", "skipped")
        if res["status"] == "completed":
            assert "action_results" in res
            for act, act_res in res["action_results"].items():
                assert "mean_spearman_rho" in act_res
                assert "missing_feature_rate" in act_res
                assert not np.isnan(act_res["mean_spearman_rho"])

    def test_audit_all_actions_fidelity(self):
        """Audit synthetic explanation fidelity across all three actions (NEUTRAL, BUY, SELL)."""
        res = audit_all_actions_fidelity(num_samples_per_action=10, random_seed=42)
        assert res["overall_pass"]
        assert res["overall_mean_spearman_rho"] >= 0.95
        assert set(res["actions_evaluated"]) == {"NEUTRAL", "BUY (LONG)", "SELL (SHORT)"}
        for act in res["actions_evaluated"]:
            act_res = res["per_action"][act]
            assert act_res["mean_spearman_rho"] >= 0.95
            assert act_res["missing_feature_rate"] == 0.0

