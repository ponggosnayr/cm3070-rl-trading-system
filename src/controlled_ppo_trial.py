"""Bounded, retrospective reward ablation; never replaces application models."""
import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from rl_env import ActiveCryptoEnv
from trading_utils import compute_indicators, fit_market_regimes_hmm

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(commission_rate=.001, slippage_rate=.0005,
                cooldown_steps=1, max_trade_duration=0, inactivity_penalty=0,
                negative_pnl_penalty=0, drawdown_penalty_coef=0,
                max_drawdown_cap=0)


class NetReturn(gym.Wrapper):
    def step(self, action):
        before = self.unwrapped.net_worth
        obs, _, terminated, truncated, info = self.env.step(action)
        after = self.unwrapped.net_worth
        reward = 10 * np.log(max(after, 1e-8) / max(before, 1e-8))
        return obs, float(reward), terminated, truncated, info


def make_env(df, training=False, plain=False):
    env = ActiveCryptoEnv(df, random_start=training,
                          max_steps=720 if training else len(df), **SETTINGS)
    return NetReturn(env) if plain else env


def evaluate(df, model=None, constant=None):
    raw = make_env(df)
    env = VecFrameStack(DummyVecEnv([lambda: raw]), n_stack=8)
    obs = env.reset()
    wealth, counts = [raw.net_worth], [0, 0, 0]
    while True:
        action = int(constant) if constant is not None else int(model.predict(obs, deterministic=True)[0][0])
        counts[action] += 1
        obs, _, done, infos = env.step([action])
        # DummyVecEnv resets automatically; terminal capital is captured in info.
        wealth.append(infos[0]['net_worth'])
        if done[0]:
            break
    env.close()
    w = np.asarray(wealth)
    roi = 100 * (w[-1] / w[0] - 1)
    dd = 100 * np.min(w / np.maximum.accumulate(w) - 1)
    return dict(roi=float(roi), drawdown=float(dd), actions=counts,
                score=float(roi + .5 * dd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=32768)
    parser.add_argument('--output', default='models/experiments/controlled_reward_20260916')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data = ROOT / 'data/btc_usdt_1h.csv'
    df = pd.read_csv(data)
    dates = pd.to_datetime(df.datetime)
    train_raw = df[dates < '2025-04-01'].iloc[:-48].reset_index(drop=True)
    hmm, mapping = fit_market_regimes_hmm(train_raw)
    joblib.dump(dict(model=hmm, map=mapping), out / 'hmm.pkl')
    features = compute_indicators(df, hmm_model=hmm, state_map=mapping)
    train = features.iloc[:len(train_raw)].reset_index(drop=True)
    windows = {name: features[(dates >= start) & (dates < end)].reset_index(drop=True)
               for name, start, end in [('bull', '2025-04-01', '2025-11-01'),
                                         ('bear', '2025-11-01', '2026-06-01')]}
    manifest = dict(steps=args.steps, seeds=[42,101,777], settings=SETTINGS,
                    score='mean(window ROI percent + 0.5 * negative max drawdown percent)',
                    evidence='Retrospective development only; no sealed test used.',
                    data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                    source_sha256={p: hashlib.sha256((ROOT/'src'/p).read_bytes()).hexdigest()
                                   for p in ['rl_env.py','trading_utils.py','controlled_ppo_trial.py']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    for name, action in [('cash',0),('buy_hold',1)]:
        metrics = {key:evaluate(frame, constant=action) for key,frame in windows.items()}
        results.append(dict(model=name, windows=metrics))
    for plain in [False, True]:
        for seed in manifest['seeds']:
            name = f'{"plain" if plain else "shaped"}_{seed}'
            env = VecNormalize(VecFrameStack(DummyVecEnv([
                lambda: make_env(train, training=True, plain=plain)]), n_stack=8),
                norm_obs=False, norm_reward=True, gamma=.99)
            model = PPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                        n_steps=1024, batch_size=256, n_epochs=5,
                        learning_rate=.0003, ent_coef=.01, gamma=.99,
                        policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64])))
            model.learn(total_timesteps=args.steps)
            model.save(out/name)
            env.save(out/f'{name}_normalizer.pkl')
            metrics = {key:evaluate(frame, model=model) for key,frame in windows.items()}
            results.append(dict(model=name, windows=metrics))
            (out/'results.json').write_text(json.dumps(results, indent=2))
            print(name, json.dumps(metrics), flush=True)
            env.close()
    (out/'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
