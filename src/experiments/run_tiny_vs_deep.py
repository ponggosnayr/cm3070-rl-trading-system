import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root and src are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from rl_env import ActiveCryptoEnv, TinyTransformerExtractor, DeepTransformerExtractor
from trading_utils import compute_indicators

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_datasets(hmm_artifact_path="models/hmm_regime.pkl"):
    csv_path = os.path.join(PROJECT_ROOT, "data", "btc_usdt_1h.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df_full = pd.read_csv(csv_path)
    dt_col = pd.to_datetime(df_full["datetime"] if "datetime" in df_full.columns else df_full["timestamp"])
    
    # 1. Train Partition: before April 2025 with 48-step purge gap
    train_mask = dt_col < "2025-04-01"
    df_train_raw = df_full[train_mask].reset_index(drop=True).iloc[:-48].reset_index(drop=True)
    df_train = compute_indicators(df_train_raw, fit_hmm=True, hmm_artifact_path=hmm_artifact_path)
    
    # 2. Bull Validation Window: April 2025 - October 2025
    df_bull_raw = df_full[(dt_col >= "2025-04-01") & (dt_col < "2025-11-01")].reset_index(drop=True)
    df_val_bull = compute_indicators(df_bull_raw, fit_hmm=False, hmm_artifact_path=hmm_artifact_path)
    
    # 3. Bear Validation Window: November 2025 - May 2026
    df_bear_raw = df_full[(dt_col >= "2025-11-01") & (dt_col < "2026-06-01")].reset_index(drop=True)
    df_val_bear = compute_indicators(df_bear_raw, fit_hmm=False, hmm_artifact_path=hmm_artifact_path)
    
    return df_train, df_val_bull, df_val_bear

def make_env_factory(df, commission=0.001, slippage=0.0005, cooldown=1):
    def _init():
        return ActiveCryptoEnv(
            df,
            initial_balance=10000.0,
            commission_rate=commission,
            slippage_rate=slippage,
            cooldown_steps=cooldown,
            inactivity_penalty=0.0005,
            negative_pnl_penalty=0.0,
            volatility_scaling=0.5,
            drawdown_penalty_coef=0.1,
            max_drawdown_cap=0.15,
            include_regime=True,
            include_context=True,
            max_trade_duration=72,
            random_start=True,
            max_steps=2048,
            warm_start_dsr=False,
            use_dsor=False
        )
    return _init

def count_parameters(model):
    total = sum(p.numel() for p in model.policy.parameters())
    extractor_pi = sum(p.numel() for p in model.policy.features_extractor.parameters())
    extractor_vf = sum(p.numel() for p in model.policy.vf_features_extractor.parameters()) if hasattr(model.policy, "vf_features_extractor") else 0
    action_net = sum(p.numel() for p in model.policy.action_net.parameters())
    return {
        "total_params": total,
        "feature_extractor_params": extractor_pi + extractor_vf,
        "action_net_params": action_net
    }

def evaluate_on_window(model, normalizer, df, window_name="Validation", trace_path=None):
    env = DummyVecEnv([lambda: ActiveCryptoEnv(
        df,
        initial_balance=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        cooldown_steps=1,
        inactivity_penalty=0.0,
        negative_pnl_penalty=0.0,
        include_regime=True,
        include_context=True,
        max_trade_duration=0,
        random_start=False,
        max_steps=len(df),
    )])
    env = VecFrameStack(env, n_stack=8)
    if normalizer is not None:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)
        env.ret_rms = normalizer.ret_rms
        env.training = False

    obs = env.reset()
    done = False
    actions = []
    portfolio_history = [10000.0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(np.asarray(action).item()))
        obs, reward, done, info = env.step(action)
        # On termination VecEnv has reset; info retains the completed episode.
        portfolio_history.append(float(info[0]["net_worth"]))
        if done[0]:
            break

    initial_val = portfolio_history[0]
    final_val = portfolio_history[-1]
    roi = ((final_val - initial_val) / initial_val) * 100.0

    peaks = np.maximum.accumulate(portfolio_history)
    drawdowns = (peaks - portfolio_history) / peaks
    max_dd = float(np.max(drawdowns)) * 100.0

    actions_arr = np.array(actions)
    total_steps = len(actions)
    pct_cash = float(np.mean(actions_arr == 0) * 100.0) if total_steps > 0 else 0.0
    pct_long = float(np.mean(actions_arr == 1) * 100.0) if total_steps > 0 else 0.0
    pct_short = float(np.mean(actions_arr == 2) * 100.0) if total_steps > 0 else 0.0

    terminal = info[0]["episode_metrics"]
    trades = int(terminal["trades"])
    assert np.isclose(final_val, terminal["final_capital"])
    if trace_path is not None:
        pd.DataFrame({"step": np.arange(total_steps + 1), "net_worth": portfolio_history,
                      "action": [None] + actions}).to_csv(trace_path, index=False)
    env.close()

    # Benchmark: Buy and Hold
    bnh_roi = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100.0

    return {
        "window": window_name,
        "steps": total_steps,
        "roi_pct": round(roi, 2),
        "bnh_roi_pct": round(bnh_roi, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "trades": trades,
        "pct_cash": round(pct_cash, 1),
        "pct_long": round(pct_long, 1),
        "pct_short": round(pct_short, 1),
    }

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq_steps=25000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq_steps = check_freq_steps
        self.last_reported_step = 0
        self.last_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_reported_step >= self.check_freq_steps:
            now = time.time()
            elapsed = now - self.last_time
            steps_done = self.num_timesteps - self.last_reported_step
            sps = steps_done / elapsed if elapsed > 0 else 0
            pct = (self.num_timesteps / self.total_timesteps) * 100.0
            print(f"  [Progress] Step {self.num_timesteps:,}/{self.total_timesteps:,} ({pct:.1f}%) | Speed: {sps:.1f} steps/s", flush=True)
            self.last_reported_step = self.num_timesteps
            self.last_time = now
        return True

def train_and_evaluate_variant(variant_name, extractor_class, features_dim, num_layers, df_train, df_bull, df_bear, steps=32768, seed=42):
    print(f"\n=======================================================")
    print(f" TRAINING VARIANT: {variant_name.upper()}")
    print(f" Architecture: {extractor_class.__name__} (features_dim={features_dim}, num_layers={num_layers})")
    print(f" Steps: {steps:,} | Seed: {seed} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"=======================================================\n")
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4 parallel training environments
    num_envs = 4
    train_env = DummyVecEnv([make_env_factory(df_train) for _ in range(num_envs)])
    train_env = VecFrameStack(train_env, n_stack=8)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.97)
    
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
        share_features_extractor=False,
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=4,
        gamma=0.97,
        gae_lambda=0.92,
        ent_coef=0.02,
        vf_coef=0.5,
        verbose=0,
        seed=seed,
        device=device,
    )
    
    param_counts = count_parameters(model)
    print(f"Model Parameters: Total={param_counts['total_params']:,}, Extractor={param_counts['feature_extractor_params']:,}, ActionNet={param_counts['action_net_params']:,}")
    
    start_time = time.time()
    callback = ProgressLoggerCallback(total_timesteps=steps, check_freq_steps=max(steps // 20, 10000))
    model.learn(total_timesteps=steps, callback=callback)
    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f}s ({steps / train_duration:.1f} steps/sec)")
    
    # Inspect action_net weights
    action_w = model.policy.action_net.weight.detach().cpu().numpy()
    action_b = model.policy.action_net.bias.detach().cpu().numpy()
    weight_stats = {
        "action_net_w_abs_max": float(np.max(np.abs(action_w))),
        "action_net_w_std": float(np.std(action_w)),
        "action_net_w_mean": float(np.mean(action_w)),
        "action_net_bias": [float(b) for b in action_b]
    }
    
    print(f"ActionNet Weights: max={weight_stats['action_net_w_abs_max']:.5f}, std={weight_stats['action_net_w_std']:.5f}, bias={[round(b, 4) for b in action_b]}")
    
    # Save checkpoint
    out_dir = os.path.join(PROJECT_ROOT, "models", "experiments", "tiny_vs_deep_comparison")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{variant_name}_model.pth")
    norm_path = os.path.join(out_dir, f"{variant_name}_normalizer.pkl")
    
    torch.save({
        "policy": model.policy.state_dict(),
        "total_steps": steps,
        "seed": seed,
        "param_counts": param_counts,
        "weight_stats": weight_stats
    }, ckpt_path)
    train_env.save(norm_path)
    
    # Evaluate Out-Of-Sample
    bull_eval = evaluate_on_window(model, train_env, df_bull, "Bull (Apr-Oct 2025)")
    bear_eval = evaluate_on_window(model, train_env, df_bear, "Bear (Nov 2025-May 2026)")
    
    return {
        "variant": variant_name,
        "param_counts": param_counts,
        "train_duration_sec": round(train_duration, 2),
        "steps_per_sec": round(steps / train_duration, 1),
        "weight_stats": weight_stats,
        "bull_eval": bull_eval,
        "bear_eval": bear_eval,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200000, help="Training steps per model (default: 200000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    steps = args.steps
    seed = args.seed

    print("Loading and preparing datasets...")
    df_train, df_bull, df_bear = prepare_datasets()
    print(f"Dataset splits: Train={len(df_train)} rows | Bull Val={len(df_bull)} rows | Bear Val={len(df_bear)} rows")
    print(f"Target Training Budget: {steps:,} steps per architecture")
    
    # 1. Train Tiny Transformer (1 layer, 64 dim)
    tiny_res = train_and_evaluate_variant(
        variant_name="tiny_transformer",
        extractor_class=TinyTransformerExtractor,
        features_dim=64,
        num_layers=1,
        df_train=df_train,
        df_bull=df_bull,
        df_bear=df_bear,
        steps=steps,
        seed=seed
    )
    
    # 2. Train Deep Transformer (4 layers, 256 dim)
    deep_res = train_and_evaluate_variant(
        variant_name="deep_transformer",
        extractor_class=DeepTransformerExtractor,
        features_dim=256,
        num_layers=4,
        df_train=df_train,
        df_bull=df_bull,
        df_bear=df_bear,
        steps=steps,
        seed=seed
    )
    
    # Combined Summary
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": steps,
        "seed": seed,
        "tiny": tiny_res,
        "deep": deep_res
    }
    
    out_json = os.path.join(PROJECT_ROOT, "models", "experiments", "tiny_vs_deep_comparison", "comparison_results.json")
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
        
    print("\n=======================================================")
    print(" EXPERIMENT COMPLETE — SUMMARY COMPARISON")
    print("=======================================================")
    print(f"{'Metric':<30} | {'Tiny Transformer (1-L)':<24} | {'Deep Transformer (4-L)':<24}")
    print("-" * 84)
    print(f"{'Total Parameters':<30} | {tiny_res['param_counts']['total_params']:<24,} | {deep_res['param_counts']['total_params']:<24,}")
    print(f"{'Training Speed (steps/s)':<30} | {tiny_res['steps_per_sec']:<24} | {deep_res['steps_per_sec']:<24}")
    print(f"{'ActionNet Weight Std':<30} | {tiny_res['weight_stats']['action_net_w_std']:<24.5f} | {deep_res['weight_stats']['action_net_w_std']:<24.5f}")
    print(f"{'ActionNet Weight Max':<30} | {tiny_res['weight_stats']['action_net_w_abs_max']:<24.5f} | {deep_res['weight_stats']['action_net_w_abs_max']:<24.5f}")
    print(f"{'Bull ROI / BnH':<30} | {tiny_res['bull_eval']['roi_pct']:+.2f}% / {tiny_res['bull_eval']['bnh_roi_pct']:+.2f}% | {deep_res['bull_eval']['roi_pct']:+.2f}% / {deep_res['bull_eval']['bnh_roi_pct']:+.2f}%")
    print(f"{'Bull Actions (Cash/Long/Short)':<30} | {tiny_res['bull_eval']['pct_cash']}% / {tiny_res['bull_eval']['pct_long']}% / {tiny_res['bull_eval']['pct_short']}% | {deep_res['bull_eval']['pct_cash']}% / {deep_res['bull_eval']['pct_long']}% / {deep_res['bull_eval']['pct_short']}%")
    print(f"{'Bear ROI / BnH':<30} | {tiny_res['bear_eval']['roi_pct']:+.2f}% / {tiny_res['bear_eval']['bnh_roi_pct']:+.2f}% | {deep_res['bear_eval']['roi_pct']:+.2f}% / {deep_res['bear_eval']['bnh_roi_pct']:+.2f}%")
    print(f"{'Bear Actions (Cash/Long/Short)':<30} | {tiny_res['bear_eval']['pct_cash']}% / {tiny_res['bear_eval']['pct_long']}% / {tiny_res['bear_eval']['pct_short']}% | {deep_res['bear_eval']['pct_cash']}% / {deep_res['bear_eval']['pct_long']}% / {deep_res['bear_eval']['pct_short']}%")
    print(f"{'Bear Max Drawdown':<30} | {tiny_res['bear_eval']['max_drawdown_pct']:.2f}% | {deep_res['bear_eval']['max_drawdown_pct']:.2f}%")
    print("=" * 84)
    print(f"Results saved to: {out_json}")

if __name__ == "__main__":
    main()
