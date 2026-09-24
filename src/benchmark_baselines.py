import os
import sys
import argparse
import pandas as pd
import torch
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl_env import ActiveCryptoEnv
from trading_utils import compute_indicators

def train_baseline(model_type, timesteps=100000):
    print(f"[{model_type}] Initializing data and environment...")
    
    from stable_baselines3.common.vec_env import VecNormalize
    
    # Load dataset and apply strictly training split to prevent test set leakage
    df_full = pd.read_csv("data/btc_usdt_1h.csv")
    test_size = max(50, int(len(df_full) * 0.2))
    df_train = compute_indicators(df_full.iloc[:-test_size].reset_index(drop=True), fit_hmm=True)
    
    # Setup environment
    def make_env():
        return ActiveCryptoEnv(df_train, render_mode=None, include_regime=True)
    
    env = DummyVecEnv([make_env for _ in range(4)])
    env = VecFrameStack(env, n_stack=8)
    # Add VecNormalize to match PPO's reward scaling
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    print(f"[{model_type}] Building model architecture (MlpPolicy)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == "DQN":
        # DQN requires discrete action space (which we have) and standard hyperparameters
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0005,
            buffer_size=100000,
            learning_starts=100,
            batch_size=128,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            device=device
        )
    elif model_type == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0007,
            n_steps=20,
            ent_coef=0.01,
            device=device
        )
    else:
        raise ValueError("Invalid model type. Choose DQN or A2C.")

    print(f"[{model_type}] Starting training for {timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps)
    
    elapsed = time.time() - start_time
    print(f"[{model_type}] Training completed in {elapsed:.2f} seconds.")
    
    # Save the model (SB3 adds .zip automatically)
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_type.lower()}_baseline"
    model.save(model_path)
    print(f"[{model_type}] Model saved to {model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["DQN", "A2C", "ALL"], default="ALL", help="Model to train")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    args = parser.parse_args()
    
    if args.model in ["DQN", "ALL"]:
        train_baseline("DQN", timesteps=args.timesteps)
    
    if args.model in ["A2C", "ALL"]:
        train_baseline("A2C", timesteps=args.timesteps)
