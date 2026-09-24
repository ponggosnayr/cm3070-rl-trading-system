"""Re-evaluate preserved weights; never trains PPO or replaces original results."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from run_tiny_vs_deep import (
    PROJECT_ROOT, PPO, DummyVecEnv, VecFrameStack, prepare_datasets,
    make_env_factory, evaluate_on_window, TinyTransformerExtractor, DeepTransformerExtractor,
)


class FixedPolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return np.array([self.action]), None


def main():
    torch.set_num_threads(2)
    root = Path(PROJECT_ROOT)
    saved = root / 'models/experiments/tiny_vs_deep_comparison'
    out = saved / ('reevaluation_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    # Original training did not archive its HMM. Reconstruct deterministically,
    # in an isolated path, from the same training split (never validation).
    train, bull, bear = prepare_datasets(str(out / 'reconstructed_hmm.pkl'))
    results = {'note': 'Single seed validation, not untouched test evidence. HMM reconstructed from original training split; original HMM was not checkpointed. Observation normalization was disabled.', 'models': {}, 'baselines': {}}
    def evaluate(name, policy):
        values = {}
        for period, df in [('bull', bull), ('bear', bear)]:
            print(f'Evaluating {name} / {period}', flush=True)
            values[period] = evaluate_on_window(policy, None, df, period, out / f'{name}_{period}.csv')
            print(values[period], flush=True)
        return values
    for name, cls, dim in [('tiny_transformer', TinyTransformerExtractor, 64), ('deep_transformer', DeepTransformerExtractor, 256)]:
        path = saved / f'{name}_model.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        env = VecFrameStack(DummyVecEnv([make_env_factory(train)]), n_stack=8)
        model = PPO('MlpPolicy', env, device='cpu', seed=checkpoint['seed'],
                    policy_kwargs=dict(features_extractor_class=cls,
                        features_extractor_kwargs=dict(features_dim=dim, n_stack=8),
                        share_features_extractor=False, net_arch=dict(pi=[128, 128], vf=[256, 256])))
        model.policy.load_state_dict(checkpoint['policy'], strict=True)
        results['models'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'seed': checkpoint['seed'], 'training_steps': checkpoint['total_steps'], 'evaluation': evaluate(name, model)}
        env.close()
        (out / 'results.json').write_text(json.dumps(results, indent=2))
    results['baselines']['cash'] = evaluate('cash', FixedPolicy(0))
    # Always-Long uses identical execution costs and environment risk rules;
    # raw price Buy & Hold remains separately reported as bnh_roi_pct.
    results['baselines']['always_long_same_rules'] = evaluate('always_long', FixedPolicy(1))
    for period, df in [('bull', bull), ('bear', bear)]:
        prices = df.close.to_numpy(dtype=float)
        values = 10000 * (1 - .001) * prices / (prices[0] * (1 + .0005))
        values[-1] *= (1 - .0005) * (1 - .001)
        equity = np.r_[10000, values]
        peaks = np.maximum.accumulate(equity)
        results['baselines'].setdefault('buy_hold_net_costs', {})[period] = {
            'roi_pct': float((equity[-1] / 10000 - 1) * 100),
            'max_drawdown_pct': float(np.max(1 - equity / peaks) * 100),
            'note': 'Buy first close, hold to last close, entry/exit costs; no stop-loss.'}
    results['settings'] = {'commission_per_side': .001, 'slippage_per_side': .0005,
        'cooldown_steps': 1, 'max_trade_duration': 0, 'max_drawdown_cap': .15,
        'gradual_drawdown': False, 'frame_stack': 8, 'deterministic': True,
        'norm_obs': False, 'norm_reward': False}
    results['dataset_sha256'] = hashlib.sha256((root / 'data/btc_usdt_1h.csv').read_bytes()).hexdigest()
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'Results saved: {out}', flush=True)


if __name__ == '__main__':
    main()
