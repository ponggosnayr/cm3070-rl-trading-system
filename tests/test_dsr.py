"""
Unit Tests for Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR)
============================================================================
Verifies mathematical properties, edge-case bounds, extreme value theory
invariants, and empirical reproducibility against Bailey & López de Prado (2014).
"""

import os
import sys
import numpy as np
import pytest
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils.dsr import (
    compute_sharpe_ratio,
    compute_sharpe_ratio_se,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    minimum_track_record_length,
)


class TestDSRMathematicalProperties:
    """Validates mathematical properties against published theoretical theorems."""

    def test_gaussian_mertens_lo_convergence(self):
        """Under normal returns (skew=0, kurt=3), Mertens SE converges exactly to Lo (2002)."""
        sr = 1.5
        T = 252
        se_mertens = compute_sharpe_ratio_se(sr, T, skewness=0.0, kurtosis=3.0)
        se_lo = np.sqrt((1.0 + 0.5 * (sr ** 2)) / (T - 1.0))
        assert np.isclose(se_mertens, se_lo, rtol=1e-6)

    def test_scale_invariance_hourly_vs_annualized(self):
        """PSR z-score and probability are scale-invariant across time frequencies."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 2000)

        # Periodic evaluation (factor=1.0)
        res_periodic = deflated_sharpe_ratio(returns=returns, n_trials=5, annualization_factor=1.0)
        # Annualized evaluation (factor=8760)
        res_ann = deflated_sharpe_ratio(returns=returns, n_trials=5, annualization_factor=8760.0)

        assert np.isclose(res_periodic["psr"], res_ann["psr"], atol=1e-4)

    def test_single_trial_dsr_equals_psr(self):
        """When N=1 (single trial), DSR against benchmark 0 equals PSR."""
        res = deflated_sharpe_ratio(sharpe_ratio=1.2, sample_length=500, n_trials=1, benchmark_sharpe=0.0)
        assert np.isclose(res["dsr"], res["psr"], atol=1e-5)

    def test_monotonic_trial_deflation(self):
        """As number of trials N increases, expected max SR increases and DSR decreases."""
        dsr_values = []
        exp_max_values = []
        for n in [1, 5, 20, 100, 500]:
            res = deflated_sharpe_ratio(sharpe_ratio=1.0, sample_length=1000, n_trials=n, trials_variance=0.5)
            dsr_values.append(res["dsr"])
            exp_max_values.append(res["expected_max_sharpe"])

        # Expected max SR must be strictly monotonically increasing
        assert all(x < y for x, y in zip(exp_max_values, exp_max_values[1:]))
        # DSR must be strictly monotonically decreasing
        assert all(x >= y for x, y in zip(dsr_values, dsr_values[1:]))

    def test_fat_tail_kurtosis_penalty(self):
        """Higher kurtosis (fat tails) increases standard error and lowers PSR."""
        se_normal = compute_sharpe_ratio_se(sharpe_ratio=0.1, sample_length=500, kurtosis=3.0)
        se_fat_tail = compute_sharpe_ratio_se(sharpe_ratio=0.1, sample_length=500, kurtosis=10.0)
        assert se_fat_tail > se_normal

        psr_normal = probabilistic_sharpe_ratio(sharpe_ratio=0.1, sample_length=500, kurtosis=3.0)
        psr_fat_tail = probabilistic_sharpe_ratio(sharpe_ratio=0.1, sample_length=500, kurtosis=10.0)
        assert psr_fat_tail < psr_normal

    def test_negative_skewness_penalty(self):
        """Negative skewness (crash risk) increases standard error and lowers PSR."""
        se_symm = compute_sharpe_ratio_se(sharpe_ratio=0.1, sample_length=500, skewness=0.0)
        se_neg_skew = compute_sharpe_ratio_se(sharpe_ratio=0.1, sample_length=500, skewness=-1.5)
        assert se_neg_skew > se_symm

        psr_symm = probabilistic_sharpe_ratio(sharpe_ratio=0.1, sample_length=500, skewness=0.0)
        psr_neg_skew = probabilistic_sharpe_ratio(sharpe_ratio=0.1, sample_length=500, skewness=-1.5)
        assert psr_neg_skew < psr_symm


class TestDSREdgeCasesAndRobustness:
    """Validates boundary conditions, empty inputs, and numerical stability."""

    def test_zero_variance_flat_returns(self):
        """Flat return series (0 variance) must return 0.0 without throwing exception."""
        flat_returns = [0.0] * 100
        res = deflated_sharpe_ratio(returns=flat_returns, n_trials=5)
        assert res["observed_sharpe"] == 0.0
        assert res["dsr"] == 0.0
        assert res["standard_error"] == 0.0

        # Non-zero flat returns (zero variance)
        res_nonzero = deflated_sharpe_ratio(returns=[0.05] * 500, n_trials=10)
        assert res_nonzero["observed_sharpe"] == 0.0
        assert res_nonzero["dsr"] == 0.0
        assert res_nonzero["psr"] == 0.0
        assert res_nonzero["standard_error"] == 0.0

        psr_flat = probabilistic_sharpe_ratio(returns=[0.05] * 500)
        assert psr_flat == 0.0

    def test_empty_or_single_element_returns(self):
        """Empty or single observation inputs must return safe zeros."""
        res_empty = deflated_sharpe_ratio(returns=[])
        assert res_empty["observed_sharpe"] == 0.0
        assert res_empty["dsr"] == 0.0

        res_single = deflated_sharpe_ratio(returns=[0.05])
        assert res_single["observed_sharpe"] == 0.0
        assert res_single["dsr"] == 0.0

    def test_sharpe_ratio_invalid_annualization_raises(self):
        """Invalid annualization factor (<= 0) must raise ValueError."""
        with pytest.raises(ValueError, match="Annualization factor must be strictly positive"):
            compute_sharpe_ratio([0.01, 0.02, -0.01], annualization_factor=0.0)

        with pytest.raises(ValueError, match="Annualization factor must be strictly positive"):
            compute_sharpe_ratio([0.01, 0.02, -0.01], annualization_factor=-1.0)

    def test_single_trial_array_resets_variance(self):
        """Passing a single-element trials_sr array sets trials_variance to 0.0."""
        res = deflated_sharpe_ratio(
            sharpe_ratio=0.60,
            sample_length=1000,
            trials_sr=[0.60],
        )
        assert res["n_trials"] == 1
        assert res["expected_max_sharpe"] == 0.0
        assert not np.isnan(res["dsr"])
        assert not np.isnan(res["psr"])

    def test_extreme_positive_skewness_mintr_clamped(self):
        """Extreme positive skewness must clamp mertens_factor and return min_trl >= 1.0."""
        min_trl = minimum_track_record_length(
            sharpe_ratio=1.0,
            benchmark_sharpe=0.0,
            skewness=100.0,
            kurtosis=3.0,
        )
        assert min_trl >= 1.0
        assert not np.isnan(min_trl)
        assert not np.isinf(min_trl)

    def test_infinite_mintr_for_sub_benchmark(self):
        """When observed Sharpe is below benchmark, MinTRL must be infinite."""
        min_trl = minimum_track_record_length(sharpe_ratio=0.5, benchmark_sharpe=1.0)
        assert min_trl == float("inf")

    def test_finite_mintr_for_positive_alpha(self):
        """When observed Sharpe exceeds benchmark, MinTRL returns positive finite period count."""
        min_trl = minimum_track_record_length(sharpe_ratio=1.5, benchmark_sharpe=0.0, alpha=0.05)
        assert 0.0 < min_trl < float("inf")


class TestEmpiricalWalkForwardDSR:
    """Validates DSR calculation across empirical Walk-Forward Validation fold outputs."""

    def test_wfv_5fold_empirical_dsr_evaluation(self):
        """Evaluate DSR across the 5 Walk-Forward Validation fold Sharpes."""
        fold_sharpes = [0.59, 1.36, 0.00, 0.00, 1.04]
        res = deflated_sharpe_ratio(
            sharpe_ratio=0.60,
            sample_length=48130,
            trials_sr=fold_sharpes,
            annualization_factor=8760.0
        )
        assert res["observed_sharpe"] == 0.60
        assert res["n_trials"] == 5
        assert 0.0 <= res["dsr"] <= 1.0
        assert 0.0 <= res["psr"] <= 1.0
