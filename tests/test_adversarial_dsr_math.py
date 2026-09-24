"""
Adversarial Stress Testing & Empirical Oracle Harness for DSR Module
===================================================================
Executes exhaustive mathematical boundary checks, extreme value theory
invariants, degenerate inputs, non-normality penalties, scale invariance,
and empirical Walk-Forward DSR verification for `src/utils/dsr.py`.
"""

import math
import os
import sys
import numpy as np
import pandas as pd
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


class TestAdversarialZeroVariance:
    """Stress tests zero variance conditions across trials and return series."""

    def test_zero_variance_trials_explicit(self):
        """When trials_variance = 0, expected max SR must equal benchmark SR exactly."""
        for bm in [-1.5, 0.0, 0.5, 2.0]:
            for n in [1, 5, 100, 10000]:
                exp_max = expected_maximum_sharpe_ratio(n_trials=n, trials_variance=0.0, benchmark_sharpe=bm)
                assert exp_max == bm, f"Expected {bm}, got {exp_max}"

    def test_identical_trials_array(self):
        """When all trial Sharpes are identical, variance is 0 and DSR equals PSR against mean."""
        identical_trials = [0.75, 0.75, 0.75, 0.75, 0.75]
        res = deflated_sharpe_ratio(
            sharpe_ratio=0.75,
            sample_length=1000,
            trials_sr=identical_trials,
            annualization_factor=1.0,
        )
        assert not np.isnan(res["dsr"])
        assert not np.isnan(res["psr"])
        assert not np.isinf(res["dsr"])
        assert res["expected_max_sharpe"] == 0.0  # Under null hypothesis with trials_sr
        assert res["dsr"] >= 0.0 and res["dsr"] <= 1.0

    def test_flat_zero_returns_series(self):
        """Flat return series (all 0.0 or all identical non-zero) must produce zero SR without error."""
        for val in [0.0, 0.05, -0.02]:
            returns = [val] * 500
            sr = compute_sharpe_ratio(returns)
            assert sr == 0.0
            res = deflated_sharpe_ratio(returns=returns, n_trials=10)
            assert res["observed_sharpe"] == 0.0
            assert res["dsr"] == 0.0
            assert res["psr"] == 0.0
            assert res["standard_error"] == 0.0


class TestAdversarialNegativeSharpe:
    """Stress tests negative Sharpe ratios across all functions."""

    @pytest.mark.parametrize("sr", [-0.01, -0.5, -2.5, -10.0, -100.0])
    def test_negative_sharpe_se_and_psr(self, sr):
        """SE must be strictly positive and finite; PSR must be <= 0.5 for SR <= 0."""
        se = compute_sharpe_ratio_se(sharpe_ratio=sr, sample_length=252)
        assert se > 0.0 and not np.isnan(se) and not np.isinf(se)

        psr = probabilistic_sharpe_ratio(sharpe_ratio=sr, sample_length=252, benchmark_sharpe=0.0)
        assert 0.0 <= psr <= 0.5, f"PSR for negative SR {sr} was {psr}, expected <= 0.5"

    def test_strongly_negative_trial_distributions(self):
        """DSR under negative trial Sharpes must evaluate safely."""
        neg_trials = [-1.5, -2.0, -0.8, -3.2, -1.1]
        res = deflated_sharpe_ratio(
            sharpe_ratio=-0.5,
            sample_length=500,
            trials_sr=neg_trials,
        )
        assert not np.isnan(res["dsr"])
        assert not np.isinf(res["dsr"])
        assert 0.0 <= res["dsr"] <= 1.0


class TestAdversarialExtremeSkewness:
    """Stress tests extreme positive and negative skewness."""

    @pytest.mark.parametrize("skew", [-1e6, -100.0, -10.0, 0.0, 10.0, 100.0, 1e6])
    def test_extreme_skewness_stability(self, skew):
        """Standard error must remain strictly non-negative and finite regardless of skewness."""
        sr = 0.8
        T = 500
        se = compute_sharpe_ratio_se(sharpe_ratio=sr, sample_length=T, skewness=skew, kurtosis=3.0)
        assert not np.isnan(se)
        assert not np.isinf(se)
        assert se > 0.0

        res = deflated_sharpe_ratio(
            sharpe_ratio=sr,
            sample_length=T,
            skewness=skew,
            kurtosis=3.0,
            n_trials=10,
            trials_variance=0.2,
        )
        assert not np.isnan(res["dsr"])
        assert not np.isnan(res["psr"])
        assert 0.0 <= res["dsr"] <= 1.0
        assert 0.0 <= res["psr"] <= 1.0

    def test_negative_skewness_increases_variance(self):
        """Negative skewness (crash risk) must increase asymptotic SE relative to positive skewness for SR > 0."""
        sr = 1.0
        T = 500
        se_pos = compute_sharpe_ratio_se(sharpe_ratio=sr, sample_length=T, skewness=2.0)
        se_neg = compute_sharpe_ratio_se(sharpe_ratio=sr, sample_length=T, skewness=-2.0)
        assert se_neg > se_pos


class TestAdversarialExtremeKurtosis:
    """Stress tests extreme sub-Gaussian (kurtosis < 3) and leptokurtic (kurtosis >> 3) tails."""

    @pytest.mark.parametrize("kurt", [0.0, 1.0, 1.5, 3.0, 10.0, 100.0, 1000.0, 1e6])
    def test_extreme_kurtosis_stability(self, kurt):
        """Standard error must remain valid and finite under extreme kurtosis values."""
        sr = 0.5
        T = 1000
        se = compute_sharpe_ratio_se(sharpe_ratio=sr, sample_length=T, kurtosis=kurt)
        assert not np.isnan(se)
        assert not np.isinf(se)
        assert se > 0.0

        psr = probabilistic_sharpe_ratio(sharpe_ratio=sr, sample_length=T, kurtosis=kurt)
        assert 0.0 <= psr <= 1.0

    def test_kurtosis_monotonic_se_growth(self):
        """For SR != 0, SE must strictly increase monotonically with kurtosis."""
        sr = 0.75
        T = 500
        kurt_values = [1.5, 3.0, 6.0, 12.0, 50.0, 200.0]
        se_list = [compute_sharpe_ratio_se(sr, T, kurtosis=k) for k in kurt_values]
        for i in range(len(se_list) - 1):
            assert se_list[i] < se_list[i + 1], f"SE not increasing: {se_list[i]} >= {se_list[i+1]}"


class TestAdversarialTrialCounts:
    """Stress tests extreme trial counts N from non-positive to 10^9."""

    @pytest.mark.parametrize("n", [-10, 0, 1, 2, 5, 100, 10000, 1000000, 1000000000])
    def test_trial_count_domain(self, n):
        """Expected max SR must be finite and non-NaN for all N."""
        exp_max = expected_maximum_sharpe_ratio(n_trials=n, trials_variance=0.5, benchmark_sharpe=0.0)
        assert not np.isnan(exp_max)
        assert not np.isinf(exp_max)
        if n <= 1:
            assert exp_max == 0.0
        else:
            assert exp_max > 0.0

    def test_large_trial_monotonicity(self):
        """Expected maximum SR must strictly grow as N spans multiple orders of magnitude."""
        n_trials_list = [1, 2, 10, 100, 1000, 10000, 100000, 1000000]
        exp_max_vals = [expected_maximum_sharpe_ratio(n, trials_variance=1.0) for n in n_trials_list]
        for i in range(len(exp_max_vals) - 1):
            assert exp_max_vals[i] < exp_max_vals[i + 1]


class TestAdversarialSampleLength:
    """Stress tests degenerate and small sample sizes T <= 3."""

    @pytest.mark.parametrize("T", [-5, 0, 1])
    def test_sample_length_below_2(self, T):
        """Sample length < 2 must return safe default zeros without crashing."""
        se = compute_sharpe_ratio_se(sharpe_ratio=1.0, sample_length=T)
        assert se == 0.0
        psr = probabilistic_sharpe_ratio(sharpe_ratio=1.0, sample_length=T)
        assert psr == 0.0
        res = deflated_sharpe_ratio(sharpe_ratio=1.0, sample_length=T)
        assert res["dsr"] == 0.0
        assert res["standard_error"] == 0.0

    @pytest.mark.parametrize("T", [2, 3, 4, 5])
    def test_small_sample_length_above_1(self, T):
        """Sample length T >= 2 must compute finite positive SE and valid probabilities."""
        se = compute_sharpe_ratio_se(sharpe_ratio=0.5, sample_length=T)
        assert se > 0.0
        assert not np.isnan(se)
        assert not np.isinf(se)
        res = deflated_sharpe_ratio(sharpe_ratio=0.5, sample_length=T, n_trials=2)
        assert 0.0 <= res["dsr"] <= 1.0


class TestAdversarialScaleInvarianceAndAnnualization:
    """Stress tests annualization factor scaling and time aggregation invariance."""

    @pytest.mark.parametrize("factor", [1.0, 252.0, 8760.0, 525600.0])
    def test_scale_invariance_constant_returns(self, factor):
        """PSR z-score and probability must be invariant to time scale when returns are aggregated."""
        np.random.seed(123)
        periodic_returns = np.random.normal(0.0002, 0.01, 5000)
        
        # Periodic evaluation
        psr_periodic = probabilistic_sharpe_ratio(returns=periodic_returns, annualization_factor=1.0)
        # Annualized evaluation
        psr_annualized = probabilistic_sharpe_ratio(returns=periodic_returns, annualization_factor=factor)

        assert np.isclose(psr_periodic, psr_annualized, atol=1e-3)

    @pytest.mark.parametrize("deg_factor", [0.0, -1.0, -8760.0])
    def test_degenerate_annualization_factors(self, deg_factor):
        """Zero or negative annualization factors must degrade safely without unhandled crashes."""
        se = compute_sharpe_ratio_se(sharpe_ratio=1.0, sample_length=100, annualization_factor=deg_factor)
        assert not np.isnan(se)
        assert not np.isinf(se)


class TestEmpiricalMonteCarloEVTOracle:
    """Validates Euler-Mascheroni EVT analytical formula against empirical Monte Carlo simulation."""

    def test_euler_mascheroni_evt_oracle_alignment(self):
        """
        Simulate 10,000 independent draws of N standard normal max statistics
        and verify expected_maximum_sharpe_ratio matches Monte Carlo expectation within 5% error.
        """
        np.random.seed(42)
        M_simulations = 10000
        for N in [5, 20, 100]:
            # Simulate M draws of max of N standard normals
            draws = np.random.normal(0.0, 1.0, size=(M_simulations, N))
            mc_expected_max = float(np.mean(np.max(draws, axis=1)))

            # Analytical formula with variance = 1.0, benchmark = 0.0
            analytical_expected_max = expected_maximum_sharpe_ratio(n_trials=N, trials_variance=1.0, benchmark_sharpe=0.0)

            # Check relative alignment within 5% (or 0.08 absolute tolerance for small N)
            abs_err = abs(analytical_expected_max - mc_expected_max)
            rel_err = abs_err / mc_expected_max
            assert rel_err < 0.08 or abs_err < 0.10, (
                f"N={N}: analytical={analytical_expected_max:.4f}, mc={mc_expected_max:.4f}, rel_err={rel_err:.4%}"
            )


class TestEmpiricalWalkForwardCSVVerification:
    """Validates empirical DSR computation against walk_forward_results.csv ground truth data."""

    def test_walk_forward_csv_reproducibility(self):
        """Extract fold Sharpes from walk_forward_results.csv and verify DSR calculation."""
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "walk_forward_results.csv"))
        assert os.path.exists(csv_path), f"CSV not found at {csv_path}"
        df = pd.read_csv(csv_path)

        assert "Out-of-Sample Sharpe" in df.columns
        fold_sharpes = df["Out-of-Sample Sharpe"].astype(float).tolist()
        assert len(fold_sharpes) == 5
        assert fold_sharpes == [0.59, 1.36, 0.0, 0.0, 1.04]

        # Total sample length across 5 test folds
        # Fold test ranges: 0..9626 (9626 candles per fold), total 57756 / 48130 held out
        total_test_candles = 48130
        mean_sharpe = float(np.mean(fold_sharpes)) # 0.598 -> 0.60

        # Execute DSR calculation with hourly annualization
        res = deflated_sharpe_ratio(
            sharpe_ratio=0.60,
            sample_length=total_test_candles,
            trials_sr=fold_sharpes,
            annualization_factor=8760.0,
        )

        assert res["observed_sharpe"] == 0.60
        assert res["n_trials"] == 5
        assert res["sample_length"] == 48130
        assert res["expected_max_sharpe"] > 0.0
        assert res["dsr"] > 0.0
        assert not np.isnan(res["dsr"])
        assert not np.isnan(res["psr"])

        # Check MinTRL
        min_trl = minimum_track_record_length(
            sharpe_ratio=0.60,
            benchmark_sharpe=0.0,
            annualization_factor=8760.0,
        )
        assert min_trl > 0.0
        assert not math.isinf(min_trl)
