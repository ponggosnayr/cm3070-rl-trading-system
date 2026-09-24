import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from trading_utils import compute_indicators
from rl_env import ActiveCryptoEnv

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_usdt_1h.csv'))
df = compute_indicators(df)
env = ActiveCryptoEnv(df, random_start=True)
obs, _ = env.reset()
for _ in range(20):
    obs, rew, done, _, info = env.step(env.action_space.sample())
nw = info.get('net_worth', 0)
print(f'OK | obs shape: {obs.shape} | net_worth: {round(nw, 2)}')
