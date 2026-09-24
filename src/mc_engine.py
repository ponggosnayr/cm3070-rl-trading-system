import numpy as np
import pandas as pd
from trading_utils import run_single_fold_backtest, compute_indicators


def generate_mc_scenarios(
    df: pd.DataFrame, n_simulations: int = 50, noise_std: float = 0.002
):
    """
    Perturb OHLCV price data using Geometric Brownian Motion noise.
    Returns a list of perturbed DataFrames.
    """
    scenarios = []
    # Use returns to model the noise
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    for _ in range(n_simulations):
        # Add Gaussian noise to log returns
        noise = np.random.normal(0, noise_std, size=len(log_returns))
        perturbed_returns = log_returns.values + noise

        # Reconstruct price series from perturbed returns
        # P_t = P_0 * exp(sum(returns))
        perturbed_close = df["close"].iloc[0] * np.exp(
            np.cumsum(np.concatenate([[0], perturbed_returns]))
        )

        perturbed_df = df.copy()
        # Scale all OHLC columns relatively to the new close
        scale = perturbed_close / df["close"]
        for col in ["open", "high", "low", "close"]:
            perturbed_df[col] = df[col] * scale

        # Recompute indicators for the new price series
        perturbed_df = compute_indicators(perturbed_df)
        scenarios.append(perturbed_df)

    return scenarios


def inject_black_swan(df: pd.DataFrame, severity: float = 0.15):
    """
    Randomly inject a sudden price crash (e.g. -15%) at a random point.
    """
    perturbed_df = df.copy()
    # Crash in the middle half of the data
    crash_idx = np.random.randint(len(df) // 4, 3 * len(df) // 4)

    # Apply crash
    perturbed_df.loc[crash_idx:, ["open", "high", "low", "close"]] *= 1 - severity
    # Partial recovery after 5 steps
    perturbed_df.loc[crash_idx + 5 :, ["open", "high", "low", "close"]] *= (
        1 + severity * 0.4
    )

    # Recompute indicators
    perturbed_df = compute_indicators(perturbed_df)
    return perturbed_df, crash_idx


def run_mc_analysis(model, df, n_simulations=50, noise_std=0.002, inject_swan=False):
    """
    Run backtest over multiple Monte Carlo scenarios and aggregate results.
    """
    scenarios = generate_mc_scenarios(df, n_simulations, noise_std)
    results = []
    equity_curves = []

    for scenario_df in scenarios:
        if inject_swan:
            scenario_df, _ = inject_black_swan(scenario_df)
            
        metrics = run_single_fold_backtest(model, scenario_df)
        results.append(metrics)
        equity_curves.append(metrics["NetWorthHistory"])

    # Aggregate summary
    summary = {
        "mean_roi": np.mean([r["ROI %"] for r in results]),
        "std_roi": np.std([r["ROI %"] for r in results]),
        "min_roi": np.min([r["ROI %"] for r in results]),
        "max_roi": np.max([r["ROI %"] for r in results]),
        "var_5th": np.percentile([r["ROI %"] for r in results], 5),  # Value at Risk
        "mean_sharpe": np.mean([r["Sharpe"] for r in results]),
        "mean_drawdown": np.mean([r["Max DD %"] for r in results]),
        "success_rate": len([r for r in results if r["ROI %"] > 0])
        / n_simulations
        * 100,
    }

    return summary, equity_curves
