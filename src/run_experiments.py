import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from rl_env import ActiveCryptoEnv, DeepTransformerExtractor, TinyTransformerExtractor
from trading_utils import compute_indicators

def run_eval(model_name, model, env):
    obs = env.reset()
    done = False
    net_worth_history = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, info_array = env.step(action)
        info_dict = info_array[0] if isinstance(info_array, (list, tuple, np.ndarray)) else info_array
        if "net_worth" in info_dict:
            net_worth_history.append(float(info_dict["net_worth"]))
        if np.any(done_array):
            if 'terminal_observation' in info_dict:
                info_dict = info_dict.get('terminal_info', info_dict)
            ep_metrics = info_dict.get('episode_metrics', {})
            roi = ep_metrics.get('roi', 0)
            trades = ep_metrics.get('trades', 0)
            max_dd = ep_metrics.get('max_drawdown', 0)
            
            # Compute genuine annualized Sharpe ratio from step returns
            if len(net_worth_history) > 1:
                pct_rets = pd.Series(net_worth_history).pct_change().dropna()
                ret_std = float(pct_rets.std())
                if ret_std > 1e-8:
                    sharpe = float((pct_rets.mean() / ret_std) * np.sqrt(8760))
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
            return {"Model": model_name, "ROI (%)": round(roi, 2), "Max DD (%)": round(max_dd, 2), "Trades": trades, "Sharpe": round(sharpe, 2)}
    return None

def run_experiments():
    print("Loading data for experiments...")
    df = pd.read_csv("data/btc_usdt_1h.csv")
    df = compute_indicators(df)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    def make_train_env(): return ActiveCryptoEnv(train_df, render_mode=None, include_regime=True)
    def make_test_env(): return ActiveCryptoEnv(test_df, render_mode=None, include_regime=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # --- 1. DQN ---
    print("\n--- Training & Evaluating DQN ---")
    dqn_env = DummyVecEnv([make_train_env for _ in range(4)])
    dqn_env = VecFrameStack(dqn_env, n_stack=8)
    dqn_model = DQN("MlpPolicy", dqn_env, verbose=0, learning_starts=100, device=device)
    dqn_model.learn(total_timesteps=10000)
    
    dqn_test_env = VecFrameStack(DummyVecEnv([make_test_env]), n_stack=8)
    res = run_eval("DQN (Baseline)", dqn_model, dqn_test_env)
    if res: results.append(res)

    # --- 2. A2C ---
    print("\n--- Training & Evaluating A2C ---")
    a2c_env = DummyVecEnv([make_train_env for _ in range(4)])
    a2c_env = VecFrameStack(a2c_env, n_stack=8)
    a2c_model = A2C("MlpPolicy", a2c_env, verbose=0, device=device)
    a2c_model.learn(total_timesteps=10000)
    
    a2c_test_env = VecFrameStack(DummyVecEnv([make_test_env]), n_stack=8)
    res = run_eval("A2C (Baseline)", a2c_model, a2c_test_env)
    if res: results.append(res)

    # --- 3. PPO (Pre-trained) ---
    print("\n--- Evaluating Pre-trained PPO ---")
    if os.path.exists("models/persistent_brain.pth"):
        try:
            # Patch torch load
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            
            checkpoint = torch.load("models/persistent_brain.pth", map_location="cpu")
            state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint
            
            dynamic_kwargs = dict(
                features_extractor_class=DeepTransformerExtractor,
                features_extractor_kwargs=dict(features_dim=256, n_stack=8),
                share_features_extractor=False,
                net_arch=dict(pi=[128, 128], vf=[256, 256]),
            )
            ppo_env = VecFrameStack(DummyVecEnv([make_test_env]), n_stack=8)
            ppo_model = PPO("MlpPolicy", ppo_env, policy_kwargs=dynamic_kwargs)
            ppo_model.policy.load_state_dict(state_dict)
            
            res = run_eval("PPO (Transformer)", ppo_model, ppo_env)
            if res: results.append(res)
            torch.load = original_load # restore
        except Exception as e:
            print(f"Failed to load PPO: {e}")

    res_df = pd.DataFrame(results)
    print("\n--- FINAL BENCHMARK RESULTS ---")
    print(res_df.to_string(index=False))
    res_df.to_csv("comparison_results.csv", index=False)
    print("Saved to comparison_results.csv")

if __name__ == "__main__":
    run_experiments()
