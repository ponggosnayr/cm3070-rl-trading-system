import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from rl_env import ActiveCryptoEnv, DeepTransformerExtractor
from trading_utils import compute_indicators, compute_deflated_sharpe_ratio, run_single_fold_backtest, fit_market_regimes_hmm

def run_walk_forward_validation(
    n_splits=5,
    train_steps=50000,
    dataset_path="data/btc_usdt_1h.csv",
    purge_window=24,
):
    """
    Executes a Walk-Forward Validation (WFV) scheme using the DeepTransformerExtractor policy:
    1. Splits dataset into N sequential expanding training windows and out-of-sample test windows.
    2. Purges boundary samples to prevent look-ahead / autocorrelation leakage between folds.
    3. Fits HMM and preprocessing strictly on each fold's training partition.
    4. Evaluates out-of-sample performance on unseen market data using frozen fold parameters.
    5. Computes Deflated Sharpe Ratio (DSR) to test against backtest overfitting.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}", flush=True)
        return None

    print(f"Loading dataset from {dataset_path}...", flush=True)
    df_raw = pd.read_csv(dataset_path)

    total_len = len(df_raw)
    fold_size = total_len // (n_splits + 1)
    
    oof_results = []
    all_oof_returns = []
    
    print(f"\n========================================================", flush=True)
    print(f"   Walk-Forward Validation ({n_splits} Sequential Folds)")
    print(f"   Architecture: PPO + DeepTransformerExtractor (persistent_brain match)")
    print(f"   In-Sample Training Steps: {train_steps} per fold | Purge Window: {purge_window} steps", flush=True)
    print(f"========================================================\n", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure Transformer policy kwargs matching persistent_brain.pth
    policy_kwargs = dict(
        features_extractor_class=DeepTransformerExtractor,
        features_extractor_kwargs=dict(features_dim=256, n_stack=8),
        share_features_extractor=False,
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
    )
    
    for fold in range(n_splits):
        train_start = 0
        train_end = (fold + 1) * fold_size
        purge_gap = min(purge_window, fold_size // 10)
        test_start = min(train_end + purge_gap, total_len - 1)
        test_end = min(train_end + fold_size, total_len)
        
        train_df_raw = df_raw.iloc[train_start:train_end].reset_index(drop=True)
        test_df_raw = df_raw.iloc[test_start:test_end].reset_index(drop=True)

        # Fit HMM strictly on this fold's training partition to eliminate look-ahead leakage
        fold_hmm, fold_map = fit_market_regimes_hmm(train_df_raw, window=30)
        train_df = compute_indicators(train_df_raw, hmm_model=fold_hmm, state_map=fold_map)

        # Preprocess test partition using causal warm-up from training window tail (no future data)
        warmup_len = min(100, len(train_df_raw))
        warmup_slice = train_df_raw.iloc[-warmup_len:]
        combined_test = pd.concat([warmup_slice, test_df_raw], ignore_index=True)
        combined_indicators = compute_indicators(combined_test, hmm_model=fold_hmm, state_map=fold_map)
        test_df = combined_indicators.iloc[warmup_len:].reset_index(drop=True)
        
        print(f"--> Fold {fold+1}/{n_splits} | Train rows: {len(train_df)} ({train_start}:{train_end}) | Purged: {purge_gap} | Test rows: {len(test_df)} ({test_start}:{test_end})", flush=True)
        
        # Build training env
        def make_train_env():
            return ActiveCryptoEnv(train_df, render_mode=None, include_regime=True, warm_start_dsr=True)
        train_env = VecFrameStack(DummyVecEnv([make_train_env]), n_stack=8)
        
        # Train PPO agent with DeepTransformerExtractor on fold's in-sample data
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=128,
            gamma=0.97,
            gae_lambda=0.92,
            target_kl=0.02,
            verbose=0,
            device=device
        )
        model.learn(total_timesteps=train_steps)
        
        # Run out-of-sample evaluation using trading_utils runner
        res = run_single_fold_backtest(model, test_df, include_regime=True)
        
        roi = res.get("ROI %", 0.0)
        bh_roi = res.get("B&H %", 0.0)
        sharpe = res.get("Sharpe", 0.0)
        max_dd = res.get("Max DD %", 0.0)
        trades = res.get("Trades", 0)
        
        nw_history = res.get("NetWorthHistory", [])
        if len(nw_history) > 1:
            rets = pd.Series(nw_history).pct_change().dropna().tolist()
            all_oof_returns.extend(rets)
            
        oof_results.append({
            "Fold": fold + 1,
            "Train Range": f"0..{train_end}",
            "Test Range": f"{test_start}..{test_end}",
            "Out-of-Sample ROI (%)": roi,
            "Market B&H (%)": bh_roi,
            "Out-of-Sample Sharpe": sharpe,
            "Max DD (%)": max_dd,
            "Trades": trades
        })
        
        print(f"    [Result] Fold {fold+1}: Out-of-Sample ROI = {roi}%, Sharpe = {sharpe}, Max DD = {max_dd}%, Trades = {trades}\n", flush=True)

    wf_df = pd.DataFrame(oof_results)
    
    # Compute Deflated Sharpe Ratio over entire combined out-of-sample return stream
    dsr_sharpe, p_val = compute_deflated_sharpe_ratio(all_oof_returns, n_trials=n_splits)
    
    print("\n========================================================", flush=True)
    print("   WALK-FORWARD EVALUATION SUMMARY (OUT-OF-SAMPLE)", flush=True)
    print("========================================================", flush=True)
    print(wf_df.to_string(index=False), flush=True)
    print(f"\nAggregate Out-of-Sample Sharpe: {dsr_sharpe}", flush=True)
    is_sig = p_val >= 0.95
    print(f"Deflated Sharpe Ratio (DSR): {p_val:.4f} (DSR >= 0.95 indicates 95% statistical significance over noise/selection bias: {'SIGNIFICANT' if is_sig else 'NOT SIGNIFICANT'})", flush=True)
    print("========================================================\n", flush=True)
    
    output_path = "walk_forward_results.csv"
    wf_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)

    return wf_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Walk-Forward Validation on PPO trading agent")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of sequential walk-forward folds")
    parser.add_argument("--timesteps", type=int, default=30000, help="Training timesteps per fold")
    parser.add_argument("--data", type=str, default="data/btc_usdt_1h.csv", help="Dataset CSV path")
    parser.add_argument("--purge_window", type=int, default=24, help="Purge window between in-sample and out-of-sample folds")
    args = parser.parse_args()
    
    run_walk_forward_validation(n_splits=args.n_splits, train_steps=args.timesteps, dataset_path=args.data, purge_window=args.purge_window)
