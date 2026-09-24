import io
import os
import sys
import pandas as pd
import numpy as np
import torch

# Patch torch.load for PyTorch 2.6 compatibility with SB3 zip streams
_orig_torch_load = torch.load
def _safe_torch_load(f, *args, **kwargs):
    if hasattr(f, "read") and not isinstance(f, io.BytesIO):
        f = io.BytesIO(f.read())
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "weights_only"}
    return _orig_torch_load(f, *args, **kwargs_clean, weights_only=False)

torch.load = _safe_torch_load

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from rl_env import ActiveCryptoEnv, DeepTransformerExtractor, TinyTransformerExtractor
from trading_utils import compute_indicators

def evaluate_models():
    print("Loading test data...")
    df = pd.read_csv("data/btc_usdt_1h.csv")
    df = compute_indicators(df, fit_hmm=False) # Ensure we don't fit HMM on test evaluation!
    
    # Split: use last 20% for testing
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    def make_env():
        return ActiveCryptoEnv(
            test_df,
            render_mode=None,
            include_regime=True,
            random_start=False,
            max_steps=len(test_df)
        )
        
    def run_eval(model_name, model, env):
        print(f"Running evaluation for {model_name}...")
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

    results = []
    
    # 1. PPO
    if os.path.exists("models/persistent_brain.pth"):
        try:
            try:
                checkpoint = torch.load("models/persistent_brain.pth", map_location="cpu", weights_only=False)
            except Exception:
                checkpoint = torch.load("models/persistent_brain.pth", map_location="cpu")
            state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint
            
            # Autodetect features dimension from weights
            features_dim = 256
            extractor_class = DeepTransformerExtractor
            for key in state_dict.keys():
                if "features_extractor.embedding.weight" in key:
                    out_features = state_dict[key].shape[0]
                    if out_features == 64:
                        extractor_class = TinyTransformerExtractor
                        features_dim = 64
                    elif out_features == 256:
                        extractor_class = DeepTransformerExtractor
                        features_dim = 256
                    break

            dynamic_kwargs = dict(
                features_extractor_class=extractor_class,
                features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
                share_features_extractor=False,
                net_arch=dict(pi=[128, 128], vf=[256, 256]),
            )
            
            ppo_env = DummyVecEnv([make_env])
            ppo_env = VecFrameStack(ppo_env, n_stack=8)
            
            ppo_model = PPO("MlpPolicy", ppo_env, policy_kwargs=dynamic_kwargs)
            ppo_model.policy.load_state_dict(state_dict)
            res = run_eval("PPO (Transformer Policy)", ppo_model, ppo_env)
            if res: results.append(res)
        except Exception as e:
            print(f"Error loading PPO: {e}")

    # 2. DQN
    if os.path.exists("models/dqn_baseline.zip"):
        try:
            dqn_env = DummyVecEnv([make_env])
            dqn_env = VecFrameStack(dqn_env, n_stack=8)
            dqn_model = DQN.load("models/dqn_baseline.zip", env=dqn_env, device="cpu")
            res = run_eval("DQN (Baseline)", dqn_model, dqn_env)
            if res: results.append(res)
        except Exception as e:
            print(f"Error loading DQN: {e}")
        
    # 3. A2C
    if os.path.exists("models/a2c_baseline.zip"):
        try:
            a2c_env = DummyVecEnv([make_env])
            a2c_env = VecFrameStack(a2c_env, n_stack=8)
            a2c_model = A2C.load("models/a2c_baseline.zip", env=a2c_env, device="cpu")
            res = run_eval("A2C (Baseline)", a2c_model, a2c_env)
            if res: results.append(res)
        except Exception as e:
            print(f"Error loading A2C: {e}")

    # 4. Buy & Hold Benchmark
    try:
        bh_start = float(test_df["close"].iloc[0])
        bh_end = float(test_df["close"].iloc[-1])
        bh_roi = round(((bh_end - bh_start) / bh_start) * 100, 2)
        dd_series = (test_df["close"].cummax() - test_df["close"]) / test_df["close"].cummax()
        bh_dd = round(float(dd_series.max()) * 100, 2)
        
        pct_rets = test_df["close"].pct_change().dropna()
        bh_sharpe = round(float((pct_rets.mean() / (pct_rets.std() + 1e-8)) * np.sqrt(8760)), 2)
        
        results.append({
            "Model": "Buy & Hold (Market)",
            "ROI (%)": bh_roi,
            "Max DD (%)": bh_dd,
            "Trades": 1,
            "Sharpe": bh_sharpe
        })
    except Exception as e:
        print(f"Error computing Buy & Hold: {e}")

    # 5. Cash Baseline (Flat)
    results.append({
        "Model": "Cash (Flat)",
        "ROI (%)": 0.0,
        "Max DD (%)": 0.0,
        "Trades": 0,
        "Sharpe": 0.0
    })
        
    if results:
        res_df = pd.DataFrame(results)
        print("\n--- Model Evaluation Results ---")
        print(res_df.to_string(index=False))
        output_file = sys.argv[1] if len(sys.argv) > 1 else "comparison_results_latest.csv"
        res_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("No models were successfully evaluated.")
        
if __name__ == "__main__":
    evaluate_models()
