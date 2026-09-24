"""
Ablation Experiment: Gradual Drawdown Penalty vs Hard Cliff Penalty
===================================================================
Tests whether replacing the catastrophic -5.0 drawdown cliff with a
calibrated gradual drawdown penalty prevents PPO from collapsing into
pure cash/short avoidance, while keeping all other training & selection
conditions identical.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT)
]

import torch
import numpy as np
import pandas as pd
from trading_utils import compute_indicators, run_single_fold_backtest
from train_agent import train_agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from rl_env import ActiveCryptoEnv

DIR_COND_A = PROJECT_ROOT / "models" / "experiments" / "small_ppo"
DIR_COND_B = PROJECT_ROOT / "models" / "experiments" / "gradual_drawdown_ppo"
DIR_COND_B.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 101, 777]
TRAINING_STEPS = 40000
COMMISSION = 0.001
SLIPPAGE = 0.0005
COOLDOWN = 1
MAX_TRADE_DURATION = 0


def run_training_condition_b():
    print("=" * 80)
    print("STARTING CONDITION B TRAINING: GRADUAL DRAWDOWN PENALTY PPO")
    print("Settings: gradual_drawdown=True | drawdown_cap_penalty=0.25 | drawdown_penalty_coef=0.25")
    print("=" * 80)

    for seed in SEEDS:
        save_path = str(DIR_COND_B / f"gradual_ppo_seed_{seed}.pth")
        print(f"\n>>> Starting Condition B Training for Seed {seed} -> {save_path}")
        train_agent(
            symbol="BTC/USDT",
            interval="1H",
            steps=TRAINING_STEPS,
            entropy=0.03,
            override_entropy=True,
            reset_weights=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            extractor="mlp",
            commission=COMMISSION,
            slippage=SLIPPAGE,
            cooldown=COOLDOWN,
            inactivity_penalty=0.0,
            negative_pnl_penalty=0.0,
            drawdown_penalty_coef=0.25,
            max_drawdown_cap=0.15,
            gradual_drawdown=True,
            drawdown_cap_penalty=0.25,
            max_trade_duration=MAX_TRADE_DURATION,
            purge_window=48,
            save_path=save_path,
            seed=seed,
        )
        print(f">>> Completed Seed {seed}.\n")


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint

    dummy_df = pd.DataFrame({
        "open": [100.0] * 120, "high": [100.0] * 120, "low": [100.0] * 120,
        "close": [100.0] * 120, "volume": [100.0] * 120,
        "timestamp": range(120), "datetime": pd.to_datetime(range(120), unit="h"),
    })
    dummy_df = compute_indicators(dummy_df).ffill().bfill().fillna(50.0)
    dummy_env = DummyVecEnv([lambda: ActiveCryptoEnv(
        dummy_df, render_mode=None, include_regime=True, include_context=True,
        max_trade_duration=0, commission_rate=COMMISSION, slippage_rate=SLIPPAGE, random_start=False
    )])
    dummy_env = VecFrameStack(dummy_env, n_stack=8)

    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO("MlpPolicy", dummy_env, policy_kwargs=policy_kwargs, device=device)
    model.policy.load_state_dict(state_dict)

    norm_path = model_path.replace(".pth", "_normalizer.pkl")
    if os.path.exists(norm_path):
        try:
            norm_env = VecNormalize.load(norm_path, dummy_env)
            norm_env.training = False
            model.set_env(norm_env)
        except Exception as e:
            print(f"Warning: Could not load normalizer {norm_path}: {e}")

    model.policy.set_training_mode(False)
    return model


def run_comparative_evaluation():
    print("\n" + "=" * 80)
    print("RUNNING COMPARATIVE EVALUATION: CONDITION A vs CONDITION B vs BENCHMARKS")
    print("=" * 80)

    csv_path = PROJECT_ROOT / "data" / "btc_usdt_1h.csv"
    df = pd.read_csv(csv_path)
    df_indicators = compute_indicators(df).ffill().bfill()
    if "datetime" in df_indicators.columns:
        df_indicators["dt"] = pd.to_datetime(df_indicators["datetime"])
    elif "timestamp" in df_indicators.columns:
        unit = "s" if df_indicators["timestamp"].max() < 1e11 else "ms"
        df_indicators["dt"] = pd.to_datetime(df_indicators["timestamp"], unit=unit)

    eval_periods = {
        "Recent 2500 (~104 Days)": df_indicators.iloc[-2500:].reset_index(drop=True),
        "Bear Crash (2025-2026)": df_indicators[(df_indicators["dt"] >= "2025-05-08") & (df_indicators["dt"] <= "2026-09-07")].reset_index(drop=True),
        "ETF Bull (2023-2024)": df_indicators[(df_indicators["dt"] >= "2023-10-01") & (df_indicators["dt"] <= "2024-03-31")].reset_index(drop=True),
    }

    import api
    preserved_model = api.model
    if preserved_model is not None and hasattr(preserved_model, "policy"):
        preserved_model.policy.set_training_mode(False)

    results = []

    for period_name, period_df in eval_periods.items():
        if len(period_df) < 50:
            continue
        bh_ret = float((period_df["close"].iloc[-1] - period_df["close"].iloc[0]) / period_df["close"].iloc[0] * 100.0)

        # Baseline: Cash
        results.append({
            "Period": period_name,
            "Condition": "Baseline",
            "Model": "Cash Baseline",
            "Seed": "N/A",
            "ROI %": 0.0,
            "B&H %": bh_ret,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Max DD %": 0.0,
            "Trades": 0,
            "Policy Cash": len(period_df) - 1,
            "Policy Long": 0,
            "Policy Short": 0,
        })

        # Baseline: Buy & Hold
        bh_prices = period_df["close"].to_numpy(dtype=np.float64)
        bh_equity = 10000.0 * (bh_prices / bh_prices[0])
        bh_rets = np.diff(bh_equity) / bh_equity[:-1]
        bh_sharpe = float((np.mean(bh_rets) / np.std(bh_rets)) * np.sqrt(8760)) if len(bh_rets) > 1 and np.std(bh_rets) > 0 else 0.0
        bh_downside = np.minimum(bh_rets, 0.0) ** 2
        bh_downside_rms = float(np.sqrt(np.mean(bh_downside))) if len(bh_downside) > 0 else 0.0
        bh_sortino = float((np.mean(bh_rets) / bh_downside_rms) * np.sqrt(8760)) if bh_downside_rms > 0 else 0.0
        bh_peak = np.maximum.accumulate(bh_equity)
        bh_drawdowns = (bh_equity - bh_peak) / bh_peak * 100.0
        bh_max_dd = float(np.min(bh_drawdowns))

        results.append({
            "Period": period_name,
            "Condition": "Baseline",
            "Model": "Buy & Hold",
            "Seed": "N/A",
            "ROI %": bh_ret,
            "B&H %": bh_ret,
            "Sharpe": round(bh_sharpe, 2),
            "Sortino": round(bh_sortino, 2),
            "Max DD %": round(bh_max_dd, 2),
            "Trades": 1,
            "Policy Cash": 0,
            "Policy Long": len(period_df) - 1,
            "Policy Short": 0,
        })

        # Baseline: Preserved Transformer
        if preserved_model is not None:
            res_pres = run_single_fold_backtest(
                preserved_model, period_df, max_trade_duration=0,
                commission_rate=COMMISSION, slippage_rate=SLIPPAGE
            )
            results.append({
                "Period": period_name,
                "Condition": "Legacy Baseline",
                "Model": "Preserved Transformer",
                "Seed": "Original",
                "ROI %": res_pres["ROI %"],
                "B&H %": res_pres["B&H %"],
                "Sharpe": res_pres["Sharpe"],
                "Sortino": res_pres["Sortino"],
                "Max DD %": res_pres["Max DD %"],
                "Trades": res_pres["Trades"],
                "Policy Cash": res_pres["PolicyActionCounts"]["cash"],
                "Policy Long": res_pres["PolicyActionCounts"]["long"],
                "Policy Short": res_pres["PolicyActionCounts"]["short"],
            })

        # Condition A: Hard Cliff Small PPO
        for seed in SEEDS:
            m_path = str(DIR_COND_A / f"small_ppo_seed_{seed}.pth")
            if not os.path.exists(m_path):
                continue
            model = load_model(m_path)
            res = run_single_fold_backtest(
                model, period_df, max_trade_duration=0,
                commission_rate=COMMISSION, slippage_rate=SLIPPAGE
            )
            results.append({
                "Period": period_name,
                "Condition": "Condition A (Hard Cliff)",
                "Model": f"Condition A (Seed {seed})",
                "Seed": str(seed),
                "ROI %": res["ROI %"],
                "B&H %": res["B&H %"],
                "Sharpe": res["Sharpe"],
                "Sortino": res["Sortino"],
                "Max DD %": res["Max DD %"],
                "Trades": res["Trades"],
                "Policy Cash": res["PolicyActionCounts"]["cash"],
                "Policy Long": res["PolicyActionCounts"]["long"],
                "Policy Short": res["PolicyActionCounts"]["short"],
            })

        # Condition B: Gradual Drawdown Small PPO
        for seed in SEEDS:
            m_path = str(DIR_COND_B / f"gradual_ppo_seed_{seed}.pth")
            if not os.path.exists(m_path):
                continue
            model = load_model(m_path)
            res = run_single_fold_backtest(
                model, period_df, max_trade_duration=0,
                commission_rate=COMMISSION, slippage_rate=SLIPPAGE
            )
            results.append({
                "Period": period_name,
                "Condition": "Condition B (Gradual Penalty)",
                "Model": f"Condition B (Seed {seed})",
                "Seed": str(seed),
                "ROI %": res["ROI %"],
                "B&H %": res["B&H %"],
                "Sharpe": res["Sharpe"],
                "Sortino": res["Sortino"],
                "Max DD %": res["Max DD %"],
                "Trades": res["Trades"],
                "Policy Cash": res["PolicyActionCounts"]["cash"],
                "Policy Long": res["PolicyActionCounts"]["long"],
                "Policy Short": res["PolicyActionCounts"]["short"],
            })

    results_df = pd.DataFrame(results)
    out_csv = DIR_COND_B / "ablation_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved full comparative results to {out_csv}")

    # Display clean comparative table
    display_cols = ["Period", "Condition", "Seed", "ROI %", "B&H %", "Sharpe", "Max DD %", "Trades", "Policy Cash", "Policy Long", "Policy Short"]
    print("\n" + results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    run_training_condition_b()
    run_comparative_evaluation()
