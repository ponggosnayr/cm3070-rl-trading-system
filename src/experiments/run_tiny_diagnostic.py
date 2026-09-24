"""Bounded Tiny Transformer safeguard diagnostic.

This is deliberately isolated from the historical tiny-vs-deep experiment.
It compares safeguard packages while evaluating both checkpoints under one
shared gradual/full-cost protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "src" / "experiments")]

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from rl_env import ActiveCryptoEnv, TinyTransformerExtractor
from run_tiny_vs_deep import prepare_datasets

SEEDS = (42, 101, 777)
FULL_COSTS = dict(commission_rate=0.001, slippage_rate=0.0005)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(df, *, gradual_drawdown: bool, drawdown_cap_penalty: float,
             random_start: bool, max_steps: int):
    return ActiveCryptoEnv(
        df, initial_balance=10000.0, **FULL_COSTS, cooldown_steps=1,
        inactivity_penalty=0.0, negative_pnl_penalty=0.0,
        volatility_scaling=0.5, drawdown_penalty_coef=0.1,
        max_drawdown_cap=0.15, gradual_drawdown=gradual_drawdown,
        drawdown_cap_penalty=drawdown_cap_penalty, include_regime=True,
        include_context=True, max_trade_duration=0, random_start=random_start,
        max_steps=max_steps, warm_start_dsr=False, use_dsor=False,
    )


def _vec(df, gradual, penalty, *, random_start=False, max_steps=None, n_envs=1):
    limit = max_steps or len(df)
    env = DummyVecEnv([lambda: make_env(df, gradual_drawdown=gradual,
                                         drawdown_cap_penalty=penalty,
                                         random_start=random_start,
                                         max_steps=limit) for _ in range(n_envs)])
    return VecFrameStack(env, n_stack=8)


class FixedPolicy:
    def __init__(self, action: int): self.action = int(action)
    def predict(self, obs, deterministic=True): return np.array([self.action]), None


def evaluate_policy(policy, df: pd.DataFrame, *, label: str, gradual=True,
                    penalty=0.25, trace_path: Path | None = None) -> dict:
    env = _vec(df, gradual, penalty, max_steps=len(df))
    obs = env.reset()
    actions, equity = [], [10000.0]
    terminal = None
    safeguards = []
    done = np.array([False])
    while not bool(done[0]):
        action, _ = policy.predict(obs, deterministic=True)
        actions.append(int(np.asarray(action).item()))
        obs, _, done, infos = env.step(action)
        info = infos[0]
        equity.append(float(info["net_worth"]))
        if str(info.get("force_closed", "")).startswith("DRAWDOWN"):
            safeguards.append(str(info["force_closed"]))
        if done[0]: terminal = info.get("episode_metrics")
    env.close()
    if terminal is None:
        raise AssertionError("terminal episode_metrics missing before VecEnv reset")
    values = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(values)
    prices = df["close"].to_numpy(dtype=float)
    # The environment enters on the next candle's open. Keep raw Buy & Hold
    # separate from a cost-adjusted comparator using the same entry convention.
    opens = df["open"].to_numpy(dtype=float)
    entry = float(opens[1] if len(opens) > 1 else opens[0])
    hold_equity = 10000.0 * (1 - FULL_COSTS["commission_rate"]) * prices[1:] / (entry * (1 + FULL_COSTS["slippage_rate"]))
    hold_equity[-1] *= (1 - FULL_COSTS["slippage_rate"]) * (1 - FULL_COSTS["commission_rate"])
    hold_equity = np.r_[10000., hold_equity]
    net_hold = hold_equity[-1]
    result = {
        "window": label, "steps": len(actions),
        "horizon": "full" if len(actions) >= len(df) - 1 else "cut_short",
        "roi_pct": round(float(terminal["roi"]), 4),
        "max_drawdown_pct": round(float(np.max(1 - values / peaks) * 100), 4),
        "trades": int(terminal["trades"]),
        "policy_counts": {str(i): int(np.sum(np.asarray(actions) == i)) for i in range(3)},
        "safeguard_events": safeguards,
        "cash_roi_pct": 0.0,
        "buy_hold_roi_pct": round(float((prices[-1] / prices[0] - 1) * 100), 4),
        "buy_hold_net_costs_roi_pct": round(float((net_hold / 10000.0 - 1) * 100), 4),
        "buy_hold_net_costs_max_drawdown_pct": float(np.max(1 - hold_equity / np.maximum.accumulate(hold_equity)) * 100),
    }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"step": range(len(equity)), "net_worth": equity,
                      "action": [None] + actions}).to_csv(trace_path, index=False)
    return result


def build_model(env, seed: int):
    return PPO("MlpPolicy", env, policy_kwargs=dict(
        features_extractor_class=TinyTransformerExtractor,
        features_extractor_kwargs=dict(features_dim=64, n_stack=8),
        share_features_extractor=False, net_arch=dict(pi=[128, 128], vf=[256, 256])),
        learning_rate=1e-4, n_steps=1024, batch_size=512, n_epochs=4,
        gamma=0.97, gae_lambda=0.92, ent_coef=0.02, vf_coef=0.5,
        verbose=0, seed=seed, device="cpu")


class SparseProgress(BaseCallback):
    """Emit bounded progress output without changing training behavior."""
    def __init__(self, total_steps, interval=100_000):
        super().__init__()
        self.total_steps = total_steps
        self.interval = interval
        self.next_report = interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_report:
            print(f"Progress: {self.num_timesteps:,}/{self.total_steps:,}", flush=True)
            self.next_report += self.interval
        return True


def train_variant(name, train_df, out: Path, steps: int, seed: int, gradual: bool, penalty: float):
    set_seed(seed)
    env = _vec(train_df, gradual, penalty, random_start=True, max_steps=2048, n_envs=4)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.97)
    model = build_model(env, seed)
    print(f'Training {name}, seed={seed}, budget={steps}', flush=True)
    model.learn(total_timesteps=steps, callback=SparseProgress(steps))
    actual = int(model.num_timesteps)
    checkpoint = out / f"{name}_seed{seed}.pth"
    normalizer = out / f"{name}_seed{seed}_normalizer.pkl"
    torch.save({"policy": model.policy.state_dict(), "total_steps": actual,
                "requested_steps": steps, "seed": seed}, checkpoint)
    env.save(str(normalizer))
    env.close()
    return model, checkpoint, normalizer, actual


def synthetic_frame(prices):
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.001, "low": p * .999,
                         "close": p, "volume": np.ones(len(p)),
                         "market_regime": np.zeros(len(p)), "volume_sma_20": np.ones(len(p))})


def run_synthetic_learn(out: Path, steps=16384, seed=42):
    frames = [synthetic_frame(np.linspace(100, 130, 96)),
              synthetic_frame(np.linspace(130, 100, 96)),
              synthetic_frame(np.full(96, 100.0))]
    train = pd.concat(frames, ignore_index=True)
    model, _, _, actual = train_variant("synthetic_gradual", train, out, steps, seed, True, .25)
    return {"mode": "synthetic_learn", "requested_steps": steps, "actual_steps": actual,
            "note": "Small-budget diagnostic on a repeated rising/falling/flat trajectory; not market evidence.",
            "behavior": {name: evaluate_policy(model, frame, label=name,
                         trace_path=out / f'{name}.csv')
                         for name, frame in zip(('rising', 'falling', 'flat'), frames)}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models/experiments/tiny_diagnostic")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-learn", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    out = args.out / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        df = synthetic_frame(np.linspace(100, 110, 12))
        result = {"mode": "smoke", "cash": evaluate_policy(FixedPolicy(0), df, label="smoke_cash", trace_path=out / "smoke.csv")}
        (out / "manifest.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2)); return
    if args.synthetic_learn:
        result = run_synthetic_learn(out, min(args.steps, 16384), args.seed)
        (out / "manifest.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2)); return
    train, bull, bear = prepare_datasets(str(out / "hmm_regime.pkl"))
    settings = {"architecture": "TinyTransformerExtractor features_dim=64 num_layers=1",
                "commission_per_side": .001, "slippage_per_side": .0005,
                "cooldown_steps": 1, "max_trade_duration": 0, "inactivity_penalty": 0.0,
                "drawdown_penalty_coef": .1, "max_drawdown_cap": .15,
                "screening_budget": args.steps, "seed": args.seed,
                "training_envs": 4, "frame_stack": 8, "norm_obs": False,
                "norm_reward_train": True, "norm_reward_eval": False,
                "ppo": {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 512,
                        "n_epochs": 4, "gamma": .97, "gae_lambda": .92,
                        "ent_coef": .02, "vf_coef": .5},
                "windows": {"train": "before 2025-04-01, purge last 48 rows",
                            "bull_validation": "2025-04-01 to 2025-11-01 exclusive",
                            "bear_validation": "2025-11-01 to 2026-06-01 exclusive"},
                "evaluation_protocol": "shared gradual_drawdown=True, penalty=.25",
                "device": "cpu",
                "limitation": "Environment cap-event liquidation sets balance to net_worth without explicit exit fees."}
    variants = {"harsh": (False, 5.0), "gradual": (True, .25)}
    manifest = {"settings": settings, "dataset_sha256": sha256(PROJECT_ROOT / "data/btc_usdt_1h.csv"),
                "hmm_sha256": sha256(out / "hmm_regime.pkl"), "models": {}}
    for name, (gradual, penalty) in variants.items():
        model, checkpoint, normalizer, actual = train_variant(name, train, out, args.steps, args.seed, gradual, penalty)
        rec = {"training": {"actual_steps": actual, "checkpoint_sha256": sha256(checkpoint), "normalizer_sha256": sha256(normalizer),
                             "rule": {"gradual_drawdown": gradual, "drawdown_cap_penalty": penalty}}, "evaluation": {}}
        for period, df in (("bull", bull), ("bear", bear)):
            rec["evaluation"][period] = evaluate_policy(model, df, label=period, gradual=True, penalty=.25,
                                                         trace_path=out / f"{name}_{period}.csv")
        manifest["models"][name] = rec
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
