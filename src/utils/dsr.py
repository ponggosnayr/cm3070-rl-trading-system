"""
Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) Module
====================================================================
Implements Bailey & López de Prado (2014) statistical framework for correcting
Sharpe ratio estimates against non-normality (skewness, kurtosis) and selection
bias / multiple testing trial inflation (Euler-Mascheroni EVT approximation).

References:
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio
  Management, 40(5), 94-107.
- Mertens, E. (2002). Comments on 'Variance of the Sharpe Ratio'. Risk Magazine.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal,
  58(4), 36-52.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats


def compute_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """
    Computes the sample Sharpe ratio from periodic returns.

    Parameters:
    - returns: Array-like periodic returns series.
    - risk_free_rate: Periodic risk-free rate (default: 0.0).
    - annualization_factor: Multiplier for time aggregation (e.g. 8760 for hourly).

    Returns:
    - Sample Sharpe ratio (float).
    """
    if annualization_factor <= 0:
        raise ValueError("Annualization factor must be strictly positive")
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate
    std = np.std(excess, ddof=1)
    if std <= 1e-12 or np.isnan(std):
        return 0.0
    return float((np.mean(excess) / std) * np.sqrt(annualization_factor))


def compute_sharpe_ratio_se(
    sharpe_ratio: float,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 1.0,
) -> float:
    """
    Computes the standard error of the Sharpe ratio estimator under non-normality
    using Mertens (2002) / Lo (2002) asymptotic variance formula.

    Parameters:
    - sharpe_ratio: Estimated Sharpe ratio (annualized or periodic).
    - sample_length: Number of return observations (T).
    - skewness: Return skewness gamma_3.
    - kurtosis: Return Pearson kurtosis gamma_4 (normal = 3.0).
    - annualization_factor: Multiplier applied to annualized Sharpe ratio (e.g. 8760 for hourly).

    Returns:
    - Asymptotic standard error of the Sharpe ratio estimator.
    """
    T = int(sample_length)
    if T < 2:
        return 0.0

    # Convert to periodic Sharpe ratio for variance calculation
    sr_period = sharpe_ratio / np.sqrt(annualization_factor) if annualization_factor > 0 else sharpe_ratio

    # Mertens (2002) non-normal variance formula
    mertens_factor = 1.0 - skewness * sr_period + ((kurtosis - 1.0) / 4.0) * (sr_period ** 2)
    if np.isnan(mertens_factor) or mertens_factor <= 0.0:
        mertens_factor = 1e-12
    var_period = max(mertens_factor / (T - 1.0), 1e-12)
    se_period = np.sqrt(var_period)

    # Scale standard error to match Sharpe ratio annualization
    return float(se_period * np.sqrt(annualization_factor) if annualization_factor > 0 else se_period)


def expected_maximum_sharpe_ratio(
    n_trials: int = 1,
    trials_variance: float = 1.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Computes the expected maximum Sharpe ratio under the null hypothesis of N independent
    trials using the Euler-Mascheroni analytical approximation (Bailey & López de Prado, 2014).

    Parameters:
    - n_trials: Number of independent strategy trials (N >= 1).
    - trials_variance: Variance of Sharpe ratio estimates across trials.
    - benchmark_sharpe: Expected mean Sharpe ratio across trials (default: 0.0).

    Returns:
    - Expected maximum Sharpe ratio under the Null Hypothesis.
    """
    N = max(int(n_trials), 1)
    if N <= 1:
        return float(benchmark_sharpe)

    std_trials = np.sqrt(max(trials_variance, 0.0))
    if std_trials <= 1e-12 or np.isnan(std_trials):
        return float(benchmark_sharpe)

    em_const = 0.57721566490153286  # Euler-Mascheroni constant
    q1 = stats.norm.ppf(1.0 - 1.0 / N)
    q2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))
    exp_max_z = (1.0 - em_const) * q1 + em_const * q2

    return float(benchmark_sharpe + std_trials * exp_max_z)


def probabilistic_sharpe_ratio(
    returns: Optional[Union[List[float], np.ndarray]] = None,
    sharpe_ratio: Optional[float] = None,
    benchmark_sharpe: float = 0.0,
    sample_length: Optional[int] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 1.0,
) -> float:
    """
    Computes the Probabilistic Sharpe Ratio (PSR) against a fixed benchmark SR*.

    Parameters:
    - returns: Optional raw return series. If provided, moments are estimated from returns.
    - sharpe_ratio: Sample Sharpe ratio (if returns not provided).
    - benchmark_sharpe: Benchmark Sharpe ratio hurdle SR* (default: 0.0).
    - sample_length: Sample length T (if returns not provided).
    - skewness: Sample skewness (if returns not provided).
    - kurtosis: Sample Pearson kurtosis (if returns not provided, normal = 3.0).
    - annualization_factor: Annualization multiplier.

    Returns:
    - Probabilistic Sharpe Ratio (0.0 to 1.0).
    """
    if returns is not None:
        r = np.asarray(returns, dtype=np.float64)
        sample_length = len(r)
        if sample_length < 2:
            return 0.0
        r_std = np.std(r, ddof=1)
        if r_std <= 1e-12 or np.isnan(r_std):
            return 0.0
        sharpe_ratio = compute_sharpe_ratio(r, annualization_factor=annualization_factor)
        raw_skew = stats.skew(r)
        raw_kurt = stats.kurtosis(r, fisher=False)
        skewness = 0.0 if (np.isnan(raw_skew) or np.isinf(raw_skew)) else float(raw_skew)
        kurtosis = 3.0 if (np.isnan(raw_kurt) or np.isinf(raw_kurt)) else float(raw_kurt)

    if sharpe_ratio is None or sample_length is None or sample_length < 2:
        return 0.0

    se = compute_sharpe_ratio_se(
        sharpe_ratio=sharpe_ratio,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis,
        annualization_factor=annualization_factor,
    )
    if se <= 1e-12 or np.isnan(se) or np.isnan(sharpe_ratio):
        return 0.0

    z_stat = (sharpe_ratio - benchmark_sharpe) / se
    return float(stats.norm.cdf(z_stat))


def deflated_sharpe_ratio(
    returns: Optional[Union[List[float], np.ndarray]] = None,
    trials_sr: Optional[Union[List[float], np.ndarray]] = None,
    sharpe_ratio: Optional[float] = None,
    sample_length: Optional[int] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_trials: int = 1,
    trials_variance: float = 1.0,
    benchmark_sharpe: float = 0.0,
    annualization_factor: float = 1.0,
) -> Dict[str, float]:
    """
    Computes the Deflated Sharpe Ratio (DSR) correcting for multiple testing and non-normality.
    Supports either raw return series or summary parameter statistics.

    Parameters:
    - returns: Optional raw return series.
    - trials_sr: Optional array of Sharpe ratios across all tested trials/folds.
    - sharpe_ratio: Point estimate of strategy Sharpe ratio.
    - sample_length: Number of return observations T.
    - skewness: Return skewness gamma_3.
    - kurtosis: Return Pearson kurtosis gamma_4 (normal = 3.0).
    - n_trials: Number of trials tested N.
    - trials_variance: Variance of Sharpe ratios across trials.
    - benchmark_sharpe: Baseline benchmark Sharpe ratio.
    - annualization_factor: Annualization multiplier.

    Returns:
    - Dictionary with observed_sharpe, expected_max_sharpe, dsr, psr, standard_error, etc.
    """
    if returns is not None:
        r = np.asarray(returns, dtype=np.float64)
        sample_length = len(r)
        if sample_length < 2:
            return {
                "observed_sharpe": 0.0,
                "expected_max_sharpe": 0.0,
                "dsr": 0.0,
                "psr": 0.0,
                "standard_error": 0.0,
                "sample_length": 0,
                "n_trials": n_trials,
                "skewness": 0.0,
                "kurtosis": 3.0,
            }
        r_std = np.std(r, ddof=1)
        if r_std <= 1e-12 or np.isnan(r_std):
            return {
                "observed_sharpe": 0.0,
                "expected_max_sharpe": 0.0,
                "dsr": 0.0,
                "psr": 0.0,
                "standard_error": 0.0,
                "sample_length": int(sample_length),
                "n_trials": int(n_trials),
                "skewness": 0.0,
                "kurtosis": 3.0,
            }
        sharpe_ratio = compute_sharpe_ratio(r, annualization_factor=annualization_factor)
        raw_skew = stats.skew(r)
        raw_kurt = stats.kurtosis(r, fisher=False)
        skewness = 0.0 if (np.isnan(raw_skew) or np.isinf(raw_skew)) else float(raw_skew)
        kurtosis = 3.0 if (np.isnan(raw_kurt) or np.isinf(raw_kurt)) else float(raw_kurt)

    if sharpe_ratio is None or sample_length is None or sample_length < 2:
        return {
            "observed_sharpe": 0.0,
            "expected_max_sharpe": 0.0,
            "dsr": 0.0,
            "psr": 0.0,
            "standard_error": 0.0,
            "sample_length": 0,
            "n_trials": n_trials,
            "skewness": 0.0,
            "kurtosis": 3.0,
        }

    # Extract trial parameters if trial Sharpe ratios array is provided
    if trials_sr is not None and len(trials_sr) > 0:
        trials_arr = np.asarray(trials_sr, dtype=np.float64)
        n_trials = len(trials_arr)
        trials_variance = float(np.var(trials_arr, ddof=1)) if n_trials > 1 else 0.0
        benchmark_sharpe = float(np.mean(trials_arr))

    # Standard error under non-normality
    se = compute_sharpe_ratio_se(
        sharpe_ratio=sharpe_ratio,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis,
        annualization_factor=annualization_factor,
    )

    # Expected maximum Sharpe ratio under null hypothesis
    exp_max_sr = expected_maximum_sharpe_ratio(
        n_trials=n_trials,
        trials_variance=trials_variance,
        benchmark_sharpe=benchmark_sharpe if trials_sr is None else 0.0,
    )

    # PSR against base benchmark (e.g. 0.0)
    psr = float(stats.norm.cdf((sharpe_ratio - benchmark_sharpe) / se)) if (se > 1e-12 and not np.isnan(se)) else 0.0

    # DSR against expected maximum Sharpe ratio
    dsr = float(stats.norm.cdf((sharpe_ratio - exp_max_sr) / se)) if (se > 1e-12 and not np.isnan(se)) else 0.0

    return {
        "observed_sharpe": round(float(sharpe_ratio), 4),
        "expected_max_sharpe": round(float(exp_max_sr), 4),
        "dsr": round(float(dsr), 4),
        "psr": round(float(psr), 4),
        "standard_error": round(float(se), 4),
        "sample_length": int(sample_length),
        "n_trials": int(n_trials),
        "skewness": round(float(skewness), 4),
        "kurtosis": round(float(kurtosis), 4),
    }


def minimum_track_record_length(
    sharpe_ratio: float,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05,
    annualization_factor: float = 1.0,
) -> float:
    """
    Computes the Minimum Track Record Length (MinTRL) required for statistical significance.

    Parameters:
    - sharpe_ratio: Estimated Sharpe ratio.
    - benchmark_sharpe: Benchmark hurdle Sharpe ratio (e.g., 0.0 or expected max SR).
    - skewness: Return skewness gamma_3.
    - kurtosis: Return Pearson kurtosis gamma_4 (normal = 3.0).
    - alpha: Significance level (default: 0.05 for 95% confidence).
    - annualization_factor: Annualization multiplier.

    Returns:
    - Minimum number of observations T required, or inf if SR <= benchmark.
    """
    sr_period = sharpe_ratio / np.sqrt(annualization_factor) if annualization_factor > 0 else sharpe_ratio
    bm_period = benchmark_sharpe / np.sqrt(annualization_factor) if annualization_factor > 0 else benchmark_sharpe
    delta = sr_period - bm_period
    if delta <= 1e-6:
        return float("inf")

    z_alpha = stats.norm.ppf(1.0 - alpha)
    mertens_factor = max(0.0, 1.0 - skewness * sr_period + ((kurtosis - 1.0) / 4.0) * (sr_period ** 2))
    min_trl = 1.0 + mertens_factor * ((z_alpha / delta) ** 2)
    return float(max(1.0, min_trl))

