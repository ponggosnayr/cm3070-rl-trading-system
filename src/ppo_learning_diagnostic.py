"""Synthetic learning check and chronological feature probe. Development only."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from controlled_ppo_trial import make_env, evaluate, ROOT
from trading_utils import compute_indicators


def synthetic(seed, n):
    rng = np.random.default_rng(seed)
    signs = []
    sign = 1
    while len(signs) < n:
        signs.extend([sign] * int(rng.integers(48, 97)))
        sign *= -1
    returns = np.asarray(signs[:n]) * .003 + rng.normal(0, .0001, n)
    close = 100 * np.exp(np.cumsum(returns))
    opening = np.r_[100., close[:-1]]
    return pd.DataFrame(dict(open=opening, high=np.maximum(opening,close)*1.0001,
                             low=np.minimum(opening,close)*.9999, close=close,
                             volume=np.full(n,100.), volume_sma_20=np.full(n,100.),
                             market_regime=np.zeros(n)))


class Capture:
    def __init__(self, model):
        self.model, self.actions = model, []

    def predict(self, obs, deterministic=True):
        result = self.model.predict(obs, deterministic=deterministic)
        self.actions.append(int(result[0][0]))
        return result


def toy(out):
    train, test = synthetic(123, 8192), synthetic(987, 1024)
    train.to_csv(out/'synthetic_train.csv', index=False)
    test.to_csv(out/'synthetic_eval.csv', index=False)
    result = dict(baselines={name:evaluate(test,constant=a) for name,a in
                             [('cash',0),('long',1),('short',2)]}, runs=[])
    for seed in [42,101,777]:
        env = VecNormalize(VecFrameStack(DummyVecEnv([
            lambda: make_env(train, training=True, plain=True)]),n_stack=8),
            norm_obs=False,norm_reward=True,gamma=.99)
        model = PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,
                    n_steps=1024,batch_size=256,n_epochs=5,learning_rate=.0003,
                    ent_coef=.01,gamma=.99,
                    policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])))
        model.learn(total_timesteps=32768)
        model.save(out/f'toy_{seed}')
        env.save(out/f'toy_{seed}_normalizer.pkl')
        capture = Capture(model)
        metrics = evaluate(test,model=capture)
        signs = np.sign(test.close.to_numpy()-test.open.to_numpy())[:len(capture.actions)]
        actions = np.asarray(capture.actions)
        metrics['long_on_up'] = float(np.mean(actions[signs>0]==1))
        metrics['short_on_down'] = float(np.mean(actions[signs<0]==2))
        metrics['pass'] = bool(metrics['long_on_up']>=.7 and metrics['short_on_down']>=.7
                               and metrics['roi']>max(v['roi'] for v in result['baselines'].values()))
        result['runs'].append(dict(seed=seed,**metrics))
        (out/'toy_results.json').write_text(json.dumps(result,indent=2))
        print('TOY',seed,json.dumps(metrics),flush=True)
        env.close()


def market_stacks(df):
    """Same observable market features and eight frames as PPO; omit portfolio state."""
    env = make_env(df)
    env.reset()
    rows = []
    for i in range(len(df)):
        env.current_step = i
        rows.append(env._get_observation()[:13])
    rows = np.pad(np.asarray(rows),((7,0),(0,0)))
    return np.asarray([rows[i:i+8].reshape(-1) for i in range(len(df))])


class ProbePolicy:
    def __init__(self, model):
        self.model, self.step, self.target = model, 0, 0

    def predict(self, obs, deterministic=True):
        if self.step % 24 == 0:
            x = obs.reshape(8,17)[:,:13].reshape(1,-1)
            probability = self.model.predict_proba(x)[0,1]
            self.target = 1 if probability >= .55 else (2 if probability <= .45 else 0)
        self.step += 1
        return np.asarray([self.target]), None


def probe(out):
    raw = pd.read_csv(ROOT/'data/btc_usdt_1h.csv')
    raw = raw[pd.to_datetime(raw.datetime)<'2026-06-01'].reset_index(drop=True)
    # Reuse the prior experiment's training-only HMM, not the application artifact.
    path = ROOT/'models/experiments/controlled_reward_20260916/hmm.pkl'
    artifact = joblib.load(path)
    joblib.dump(artifact,out/'probe_hmm.pkl')
    df = compute_indicators(raw,hmm_model=artifact['model'],state_map=artifact['map'])
    dates = pd.to_datetime(df.datetime)
    x = market_stacks(df)
    future_return = df.close.shift(-24)/df.open.shift(-1)-1
    label = (future_return>0).astype(int)
    train_end = int(np.sum(dates<'2025-04-01'))-48
    # Every training target ends before the purged training boundary.
    model = make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=500))
    model.fit(x[99:train_end-24], label.iloc[99:train_end-24])
    joblib.dump(model,out/'probe_classifier.pkl')
    results = {}
    for name,start,end in [('bull','2025-04-01','2025-11-01'),('bear','2025-11-01','2026-06-01')]:
        mask = (dates>=start)&(dates<end)
        indices = np.flatnonzero(mask)
        # Non-overlapping 24-hour outcomes reduce overlap in diagnostic statistics.
        samples = indices[:-24:24]
        windows = df.loc[mask].reset_index(drop=True)
        probabilities = model.predict_proba(market_stacks(windows)[samples-indices[0]])[:,1]
        results[name] = dict(n=len(samples),
            balanced_accuracy=float(balanced_accuracy_score(label.iloc[samples],probabilities>=.5)),
            auc=float(roc_auc_score(label.iloc[samples],probabilities)),
            strategy=evaluate(windows,model=ProbePolicy(model)),
            cash=evaluate(windows,constant=0),buy_hold=evaluate(windows,constant=1))
    (out/'probe_results.json').write_text(json.dumps(results,indent=2))
    print('PROBE',json.dumps(results),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['toy','probe'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=ROOT/'models/experiments'/f'learning_diagnostic_{args.mode}_20260916'
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(mode=args.mode,description='Retrospective diagnostic, not performance certification',
                  toy_steps=32768,toy_seeds=[42,101,777],toy_pass='both direction rates >=0.7 and ROI above all constant baselines',
                  probe='logistic C=1 balanced classes; 24h target; 0.55/0.45 thresholds; no tuning',
                  sources={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                           for name in ['rl_env.py','controlled_ppo_trial.py','ppo_learning_diagnostic.py']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (toy if args.mode=='toy' else probe)(out)


if __name__=='__main__':
    main()
