"""Replay a frozen policy on predeclared windows; never train or modify inputs."""
from pathlib import Path
import argparse, csv, hashlib, importlib.metadata, json, os, random, sys

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    root=args.repo.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text())
    for name, digest in protocol['input_sha256'].items():
        assert sha(root/name)==digest, f'Input changed: {name}'
    sys.path.insert(0,str(root/'src'))
    import numpy as np, pandas as pd, torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
    from rl_env import ActiveCryptoEnv, DeepTransformerExtractor, TinyTransformerExtractor
    from trading_utils import compute_indicators
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    raw=pd.read_csv(root/'data/btc_usdt_1h.csv')
    # Use only OHLCV/time inputs, recomputing indicators with the frozen HMM.
    raw=raw[['datetime','open','high','low','close','volume']].copy()
    assert raw.notna().all().all()
    prepared=compute_indicators(raw,hmm_artifact_path=str(root/'models/hmm_regime.pkl'))
    weights=torch.load(root/'models/persistent_brain.pth',map_location='cpu',weights_only=False)
    state=weights['policy'] if 'policy' in weights else weights
    embedding=next(v for k,v in state.items() if 'features_extractor.embedding.weight' in k)
    dim, obs_dim=embedding.shape
    cfg=dict(protocol['environment']); cfg.update(include_context=obs_dim in (16,17),include_regime=obs_dim in (12,17))
    sample=prepared.iloc[protocol['windows'][0]['start']:protocol['windows'][0]['end']].reset_index(drop=True)
    vec=VecFrameStack(DummyVecEnv([lambda:ActiveCryptoEnv(sample,**cfg)]),n_stack=8)
    norm=VecNormalize.load(root/'models/persistent_brain_normalizer.pkl',vec)
    norm.training=False; norm.norm_reward=False
    model=PPO('MlpPolicy',norm,device='cpu',seed=42,policy_kwargs=dict(
        features_extractor_class=DeepTransformerExtractor if dim==256 else TinyTransformerExtractor,
        features_extractor_kwargs=dict(features_dim=dim,n_stack=8),
        share_features_extractor=False,net_arch=dict(pi=[128,128],vf=[256,256])))
    model.policy.load_state_dict(state); model.policy.set_training_mode(False)
    rows=[]; all_trace=[]
    for win in protocol['windows']:
        df=prepared.iloc[win['start']:win['end']].reset_index(drop=True)
        env=ActiveCryptoEnv(df,**cfg)
        obs,_=env.reset(seed=42)
        assert env.n_rows==len(df) and env.current_step==0
        buffer=np.zeros((8,obs_dim),dtype=np.float32); buffer[-1]=obs
        equity=[10000.0]; actions=[]; exposed=[]
        with torch.inference_mode():
            for i in range(len(df)-1):
                x=torch.as_tensor(norm.normalize_obs(buffer.reshape(1,-1)),dtype=torch.float32)
                logits=model.policy.get_distribution(x).distribution.logits.clone()
                mask=torch.as_tensor(env.action_masks(),dtype=torch.bool)
                logits[:,~mask]=-1e9
                action=int(logits.argmax(-1).item())
                next_obs,reward,terminated,truncated,info=env.step(action)
                equity.append(float(info['net_worth'])); actions.append(action)
                exposed.append(int(env.is_long or env.is_short))
                buffer[:-1]=buffer[1:]; buffer[-1]=next_obs
                all_trace.append(dict(window=win['name'],row=win['start']+i+1,
                    datetime=str(df.iloc[i+1]['datetime']),action=action,equity=equity[-1],
                    long=bool(env.is_long),short=bool(env.is_short),entries=env.trade_count))
                if terminated or truncated: break
        assert len(actions)==len(df)-1,'Unexpected early termination'
        assert not env.is_long and not env.is_short,'Terminal position not liquidated'
        # Cost-adjusted buy-and-hold: same next-open entry, final-close exit,
        # float32 prices and per-side costs; no policy holding-duration rule.
        opens=df['open'].to_numpy(dtype=np.float32); closes=df['close'].to_numpy(dtype=np.float32)
        fee=cfg['commission_rate']; slip=cfg['slippage_rate']
        bh=np.r_[10000.,10000.*(1-fee)*closes[1:].astype(float)/(float(opens[1])*(1+slip))]
        bh[-1]*=(1-slip)*(1-fee)
        # Independent check of the baseline against an always-long environment.
        check_cfg=dict(cfg,max_trade_duration=0)
        check=ActiveCryptoEnv(df,**check_cfg); check.reset(seed=42)
        for _ in range(len(df)-1): _,_,done,trunc,check_info=check.step(1)
        assert abs(float(check_info['net_worth'])-bh[-1])<0.01
        for name,curve,entries in [('PPO',np.array(equity),env.trade_count),('Cash',np.full(len(df),10000.),0),('Buy-and-Hold',bh,1)]:
            roi=(curve[-1]/10000.-1)*100
            dd=np.max(1-curve/np.maximum.accumulate(curve))*100
            rows.append(dict(window=win['name'],start=str(df.iloc[0]['datetime']),end=str(df.iloc[-1]['datetime']),
                candles=len(df),strategy=name,roi_pct=round(float(roi),6),max_drawdown_pct=round(float(dd),6),entries=int(entries),
                cash_actions=actions.count(0) if name=='PPO' else '',long_actions=actions.count(1) if name=='PPO' else '',
                short_actions=actions.count(2) if name=='PPO' else ''))
        print(win['name'],rows[-3:],flush=True)
    for filename,records in [('results.csv',rows),('policy_trace.csv',all_trace)]:
        with (out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    for name,digest in protocol['input_sha256'].items(): assert sha(root/name)==digest
    meta=dict(protocol_sha256=sha(args.protocol),runner_sha256=sha(__file__),python=sys.version,
        device='cpu',threads=1,seed=42,normalizer_frozen=True,normalise_reward=False,
        input_unchanged=True,baseline_analytic_check='passed within one cent in all windows',
        packages={x:importlib.metadata.version(x) for x in ['torch','stable-baselines3','gymnasium','numpy','pandas','hmmlearn','scikit-learn']},
        output_sha256={name:sha(out/name) for name in ['results.csv','policy_trace.csv']})
    (out/'runtime.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Replay completed; inputs unchanged.',flush=True)

if __name__=='__main__': main()
