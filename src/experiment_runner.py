"""
Multi-Seed Controlled Experiment Runner: Small MLP PPO vs Benchmarks
===================================================================
Tests whether a lightweight PPO network (2-layer MLP [64, 64]) learns
adaptive behavior under realistic transaction frictions, with:
- No forced holding-time exits (max_trade_duration = 0)
- No artificial minimum trade quotas in checkpoint selection
- Realistic commissions (0.1%) and slippage (0.05%) throughout
- Multiple random seeds (42, 101, 777)
- Policy actions tracked separately from simulator execution events
- Side-by-side comparison with Cash, Buy & Hold, and Preserved Checkpoint
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

EXPERIMENT_DIR = PROJECT_ROOT / "models" / "experiments" / "small_ppo"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 101, 777]
TRAINING_STEPS = 40000  # Focused budget for quick convergence evaluation
COMMISSION = 0.001       # 0.1%
SLIPPAGE = 0.0005        # 0.05%
COOLDOWN = 1             # Realistic 1-candle execution cooldown
MAX_TRADE_DURATION = 0   # No forced holding-time exits


def run_training():
    print("=" * 80)
    print("PHASE 1: TRAINING SMALL MLP PPO ACROSS MULTIPLE SEEDS")
    print("Configuration: Extractor=MLP [64, 64] | Realistic Fees (0.1%/0.05%) | Max Dur=0 (No forced exits)")
    print("=" * 80)

    trained_models = {}
    for seed in SEEDS:
        save_path = str(EXPERIMENT_DIR / f"small_ppo_seed_{seed}.pth")
        print(f"\n>>> Starting Training for Seed {seed} -> Saving to {save_path}")
        
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
            inactivity_penalty=0.0,    # Zero artificial inactivity penalty
            negative_pnl_penalty=0.0,
            max_trade_duration=MAX_TRADE_DURATION,
            purge_window=48,
            save_path=save_path,
            seed=seed,
        )
        trained_models[seed] = save_path
        print(f">>> Completed Seed {seed}.\n")

    return trained_models


def load_small_ppo_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint

    # Dummy env for architecture initialization
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


def run_evaluation():
    print("\n" + "=" * 80)
    print("PHASE 2: EVALUATION ON UNTOUCHED DATASETS")
    print("Comparing: Cash vs Buy & Hold vs Preserved Checkpoint vs Small PPO (Multi-Seed)")
    print("=" * 80)

    # 1. Load Data
    csv_path = PROJECT_ROOT / "data" / "btc_usdt_1h.csv"
    df = pd.read_csv(csv_path)
    df_indicators = compute_indicators(df).ffill().bfill()
    if "datetime" in df_indicators.columns:
        df_indicators["dt"] = pd.to_datetime(df_indicators["datetime"])
    elif "timestamp" in df_indicators.columns:
        unit = "s" if df_indicators["timestamp"].max() < 1e11 else "ms"
        df_indicators["dt"] = pd.to_datetime(df_indicators["timestamp"], unit=unit)

    # Evaluate on untouched windows
    eval_periods = {
        "Recent 2500 (~104 Days)": df_indicators.iloc[-2500:].reset_index(drop=True),
        "Bear Crash (2025-2026)": df_indicators[(df_indicators["dt"] >= "2025-05-08") & (df_indicators["dt"] <= "2026-09-07")].reset_index(drop=True),
        "ETF Bull (2023-2024)": df_indicators[(df_indicators["dt"] >= "2023-10-01") & (df_indicators["dt"] <= "2024-03-31")].reset_index(drop=True),
    }

    # 2. Load Preserved Baseline Checkpoint
    import api
    preserved_model = api.model
    if preserved_model is not None and hasattr(preserved_model, "policy"):
        preserved_model.policy.set_training_mode(False)

    results = []

    for period_name, period_df in eval_periods.items():
        if len(period_df) < 50:
            continue
        bh_ret = float((period_df["close"].iloc[-1] - period_df["close"].iloc[0]) / period_df["close"].iloc[0] * 100.0)

        # Compute true Buy & Hold baseline equity curve and metrics
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

        # Baseline 1: Cash
        results.append({
            "Period": period_name,
            "Model": "Cash Baseline",
            "Seed": "N/A",
            "ROI %": 0.0,
            "B&H %": bh_ret,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Max DD %": 0.0,
            "Trades": 0,
            "Forced": 0,
            "Policy Cash": len(period_df) - 1,
            "Policy Long": 0,
            "Policy Short": 0,
        })

        # Baseline 2: Buy & Hold
        results.append({
            "Period": period_name,
            "Model": "Buy & Hold",
            "Seed": "N/A",
            "ROI %": bh_ret,
            "B&H %": bh_ret,
            "Sharpe": round(bh_sharpe, 2),
            "Sortino": round(bh_sortino, 2),
            "Max DD %": round(bh_max_dd, 2),
            "Trades": 1,
            "Forced": 0,
            "Policy Cash": 0,
            "Policy Long": len(period_df) - 1,
            "Policy Short": 0,
        })

        # Baseline 3: Preserved Checkpoint (Rule Disabled)
        if preserved_model is not None:
            res_pres = run_single_fold_backtest(
                preserved_model, period_df, max_trade_duration=0,
                commission_rate=COMMISSION, slippage_rate=SLIPPAGE
            )
            forced_pres = sum(1 for t in res_pres["TradeHistory"] if t.get("Reason") == "maximum_holding_duration")
            results.append({
                "Period": period_name,
                "Model": "Preserved Transformer",
                "Seed": "Original",
                "ROI %": res_pres["ROI %"],
                "B&H %": res_pres["B&H %"],
                "Sharpe": res_pres["Sharpe"],
                "Sortino": res_pres["Sortino"],
                "Max DD %": res_pres["Max DD %"],
                "Trades": res_pres["Trades"],
                "Forced": forced_pres,
                "Policy Cash": res_pres["PolicyActionCounts"]["cash"],
                "Policy Long": res_pres["PolicyActionCounts"]["long"],
                "Policy Short": res_pres["PolicyActionCounts"]["short"],
            })

        # Small MLP PPO models across seeds
        for seed in SEEDS:
            m_path = str(EXPERIMENT_DIR / f"small_ppo_seed_{seed}.pth")
            if not os.path.exists(m_path):
                print(f"Warning: {m_path} does not exist!")
                continue
            model = load_small_ppo_model(m_path)
            res_seed = run_single_fold_backtest(
                model, period_df, max_trade_duration=0,
                commission_rate=COMMISSION, slippage_rate=SLIPPAGE
            )
            forced_seed = sum(1 for t in res_seed["TradeHistory"] if t.get("Reason") == "maximum_holding_duration")
            results.append({
                "Period": period_name,
                "Model": f"Small MLP PPO (Seed {seed})",
                "Seed": str(seed),
                "ROI %": res_seed["ROI %"],
                "B&H %": res_seed["B&H %"],
                "Sharpe": res_seed["Sharpe"],
                "Sortino": res_seed["Sortino"],
                "Max DD %": res_seed["Max DD %"],
                "Trades": res_seed["Trades"],
                "Forced": forced_seed,
                "Policy Cash": res_seed["PolicyActionCounts"]["cash"],
                "Policy Long": res_seed["PolicyActionCounts"]["long"],
                "Policy Short": res_seed["PolicyActionCounts"]["short"],
            })

    res_df = pd.DataFrame(results)
    out_csv = EXPERIMENT_DIR / "experiment_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved experimental results to {out_csv}")
    
    # Print formatted summary table
    print("\n" + "=" * 130)
    print(f"{'Period':<25} | {'Model':<25} | {'ROI %':<8} | {'B&H %':<8} | {'Sharpe':<6} | {'MaxDD %':<8} | {'Trades':<6} | {'Policy (C/L/S)':<15}")
    print("-" * 130)
    for _, r in res_df.iterrows():
        actions_str = f"{r['Policy Cash']}/{r['Policy Long']}/{r['Policy Short']}"
        print(f"{r['Period']:<25} | {r['Model']:<25} | {r['ROI %']:<8.2f} | {r['B&H %']:<8.2f} | {r['Sharpe']:<6.2f} | {r['Max DD %']:<8.2f} | {r['Trades']:<6} | {actions_str:<15}")
    print("=" * 130)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only")
    args = parser.parse_args()

    if not args.eval_only:
        run_training()
    run_evaluation()
