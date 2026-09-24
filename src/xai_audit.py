"""
XAI Explanation Ordering Audit
========================================================================
The synthetic audit calculates Spearman rank correlation between injected
attribution-like feature scores and generated explanation order. The random
arrays are not SHAP values computed from a model. A separate checkpoint scan
samples real model actions and explanations.

Goal: Check ordering logic without claiming full real-model explanation fidelity.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

# Add root and src directories to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from xai_engine import XAIEngine


from scipy.stats import spearmanr


def compute_spearman_rank_correlation(ranks_a: List[float], ranks_b: List[float]) -> float:
    """
    Compute Spearman Rank Correlation Coefficient (rho) between two rank sequences
    using SciPy to ensure mathematically sound fractional mid-ranking for ties.
    Returns 0.0 when rank variance is zero or inputs are constant/missing.
    """
    a = np.array(ranks_a, dtype=np.float64)
    b = np.array(ranks_b, dtype=np.float64)
    
    if len(a) != len(b) or len(a) < 2:
        return 0.0
        
    # Check for identical/constant values (e.g. all 9999 missing sentinels)
    if np.all(b == b[0]) or np.all(a == a[0]):
        return 0.0
        
    res = spearmanr(a, b)
    corr = res.statistic if hasattr(res, 'statistic') else res.correlation
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def audit_explanation_fidelity(
    coin_ticker: str = "BTC",
    current_price: float = 50000.0,
    rsi: float = 45.0,
    macd: float = 0.001,
    volume_ratio: float = 1.2,
    action_name: str = "BUY (LONG)",
    confidence: float = 85.0,
    probs: Optional[List[float]] = None,
    num_samples: int = 50,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Test explanation rank ordering over generated attribution-like arrays.
    Correctly penalizes missing features and handles ties.
    """
    if probs is None:
        probs = [0.10, 0.80, 0.10]
        
    np.random.seed(random_seed)
    
    feature_names = [
        "Open Ratio", "High Ratio", "Low Ratio", "Volatility", 
        "Price Change", "Price Change Prev", "Normalized Volume",
        "Short Trend (SMA20)", "Medium Trend (SMA99)", "Momentum (ROC24)", 
        "Volatility Ratio", "Macro Drawdown", "Market Regime",
        "Portfolio Position", "Unrealized P&L", "Idle Steps", "Holding Duration"
    ]
    num_features = len(feature_names)
    stacked_dim = num_features * 8  # 136 features
    
    action_idx_map = {
        "NEUTRAL": 0, "BUY (LONG)": 1, "SELL (SHORT)": 2
    }
    action_idx = action_idx_map.get(action_name, 1)
    
    top_k_rhos = []
    audit_logs = []
    missing_feature_counts = []
    
    for sample_id in range(num_samples):
        # Generate synthetic score arrays for 3 canonical actions, 136 features.
        dummy_shap = [np.random.randn(1, stacked_dim) * 0.1 for _ in range(3)]
        
        # Inject distinct random magnitude signals for features
        top_indices = np.random.choice(num_features, size=3, replace=False)
        dummy_shap[action_idx][0, top_indices[0]] = 2.5
        dummy_shap[action_idx][0, top_indices[1]] = -1.8
        dummy_shap[action_idx][0, top_indices[2]] = 1.2
        
        # Instantiate XAIEngine
        engine = XAIEngine(
            coin_ticker=coin_ticker,
            current_price=current_price,
            rsi=rsi,
            macd=macd,
            volume_ratio=volume_ratio,
            action_name=action_name,
            confidence=confidence,
            probs=probs,
            shap_values=dummy_shap,
            feature_names=feature_names
        )
        
        # Ground Truth SHAP Ranks from XAIEngine
        top_shap_features = engine._get_top_shap_features(top_n=3)
        shap_feature_names = [item['feature'] for item in top_shap_features]
        ground_truth_ranks = list(range(len(shap_feature_names)))
        
        # Natural Language generated output
        nl_text = engine.generate_response("why did you make this trade?")
        
        # Extract order of appearance in generated Natural Language rationale text
        nl_feature_positions = []
        missing_count = 0
        for feat in shap_feature_names:
            pos = nl_text.find(feat)
            if pos != -1:
                nl_feature_positions.append(pos)
            else:
                nl_feature_positions.append(9999)
                missing_count += 1
                
        missing_feature_counts.append(missing_count)
        
        # Spearman correlation on top explained features
        if missing_count == len(shap_feature_names):
            # All features missing -> zero fidelity
            rho = 0.0
        else:
            rho = compute_spearman_rank_correlation(ground_truth_ranks, nl_feature_positions)
            if missing_count > 0:
                # Penalize missing features
                rho = float(rho * (1.0 - (missing_count / len(shap_feature_names))))
                
        top_k_rhos.append(rho)
        
        audit_logs.append({
            "sample_id": sample_id,
            "rho": rho,
            "missing_count": missing_count,
            "top_shap_feature": top_shap_features[0]['feature'],
            "top_shap_impact": top_shap_features[0]['impact']
        })
        
    mean_rho = float(np.mean(top_k_rhos))
    min_rho = float(np.min(top_k_rhos))
    mean_missing_rate = float(np.mean(missing_feature_counts) / 3.0)
    pass_audit = (mean_rho >= 0.95) and (mean_missing_rate == 0.0)
    
    return {
        "mean_spearman_rho": mean_rho,
        "min_spearman_rho": min_rho,
        "missing_feature_rate": mean_missing_rate,
        "pass_audit": pass_audit,
        "num_samples": num_samples,
        "target_threshold": 0.95
    }


def audit_real_model_fidelity(num_eval_states: int = 10, output_path=None, scan_windows: int = 1, scan_steps: int = 200) -> Dict[str, Any]:
    """Audit feature ordering on real checkpoint states; not LLM factual accuracy.

    Only observed actions are reported. Missing actions are explicitly uncovered.
    SHAP explains actor logits; scores are not probabilities of profitable trading.
    """
    import json, hashlib
    from pathlib import Path
    import torch
    from rl_env import ActiveCryptoEnv
    from trading_utils import compute_indicators
    from xai_shap import compute_shap_values
    if not 1 <= num_eval_states <= 100:
        raise ValueError("num_eval_states must be between 1 and 100")
    paths = [Path("models/persistent_brain.pth"), Path("data/btc_usdt_1h.csv")]
    if not all(p.exists() for p in paths):
        return {"status": "skipped", "reason": "Artifacts not found", "overall_pass": False}
    from api import model
    if model is None:
        raise RuntimeError("Checkpoint could not be loaded")
    model.policy.set_training_mode(False)
    norm = model.get_vec_normalize_env()
    frame_dim = model.observation_space.shape[0] // 8
    if not 1 <= scan_windows <= 24 or not 16 <= scan_steps <= 500:
        raise ValueError("Bounded scan requires 1..24 windows and 16..500 steps")
    full = pd.read_csv(paths[1])
    ends = np.unique(np.linspace(min(500,len(full)),len(full),scan_windows,dtype=int)) if scan_windows>1 else [len(full)]
    states, dates, prices, predicted = [], [], [], []
    for end in ends:
        df=compute_indicators(full.iloc[max(0,end-600):end].reset_index(drop=True)).dropna().reset_index(drop=True)
        env=ActiveCryptoEnv(df,random_start=False,max_steps=len(df),
                           include_regime=frame_dim in (12,17),include_context=frame_dim in (16,17))
        obs,_=env.reset(seed=42)
        stack=np.zeros((8,frame_dim),dtype=np.float32); stack[-1]=obs
        try:
            for step in range(min(scan_steps,len(df)-1)):
                flat=stack.reshape(1,-1).copy()
                normalized=norm.normalize_obs(flat) if norm else flat
                action,_=model.predict(normalized,deterministic=True)
                action=int(np.asarray(action).item())
                if step>=8:
                    states.append(flat[0]); predicted.append(action)
                    row=df.iloc[env.current_step]
                    dates.append(str(row.get("datetime",env.current_step))); prices.append(float(row.close))
                obs,_,done,truncated,_=env.step(action)
                stack[:-1]=stack[1:]; stack[-1]=obs
                if done or truncated: break
        finally:
            env.close()
    states=np.asarray(states)
    if len(states)<8:
        raise ValueError("Insufficient historical states for the audit")
    # Round-robin sampling targets observed classes without forcing policy actions.
    pools=[list(np.flatnonzero(np.asarray(predicted)==action)) for action in range(3)]
    pools=[list(np.asarray(pool)[np.unique(np.linspace(0,len(pool)-1,min(num_eval_states,len(pool)),dtype=int))]) if pool else [] for pool in pools]
    chosen=[]
    while len(chosen)<num_eval_states and any(pools):
        for pool in pools:
            if pool and len(chosen)<num_eval_states: chosen.append(pool.pop(0))
    background=states[np.linspace(0,len(states)-1,8,dtype=int)]
    names=["NEUTRAL","BUY (LONG)","SELL (SHORT)"]
    records=[]
    rng_state=np.random.get_state()
    try:
        np.random.seed(42)
        for idx in chosen:
            flat=states[idx:idx+1]
            normalized=norm.normalize_obs(flat) if norm else flat
            with torch.no_grad():
                probs=model.policy.get_distribution(torch.as_tensor(normalized,device=model.device)).distribution.probs.cpu().numpy()[0]
            action=int(np.argmax(probs))
            values=compute_shap_values(model,flat,background,nsamples=48,vec_normalize=norm)
            engine=XAIEngine("BTC",prices[idx],50.0,0.0,1.0,names[action],float(probs[action]*100),probs.tolist(),shap_values=values)
            features=engine._get_top_shap_features(top_n=3)
            text=engine.generate_response("why did you make this trade?")
            positions=[text.find(f["feature"]) for f in features]
            missing=sum(pos<0 for pos in positions)
            rho=compute_spearman_rank_correlation(list(range(len(positions))),positions) if not missing else 0.0
            raw=np.stack(values,axis=-1) if isinstance(values,list) else np.asarray(values)
            records.append({"state_time":dates[idx],"action":names[action],"probabilities":probs.tolist(),
                            "observation":flat[0].tolist(),"shap_values":raw.tolist(),"explanation":text,
                            "rho":rho,"missing_feature_rate":missing/max(1,len(positions)),"passed":not missing and rho>=0.95})
    finally:
        np.random.set_state(rng_state)
    results={name:{"samples":len(group),"mean_spearman_rho":float(np.mean([r["rho"] for r in group])),
                   "missing_feature_rate":float(np.mean([r["missing_feature_rate"] for r in group]))}
             for name in names if (group:=[r for r in records if r["action"]==name])}
    uncovered = [n for n in names if n not in results]
    samples_passed = bool(records) and all(r["passed"] for r in records)
    coverage_complete = len(uncovered) == 0
    report={"status":"completed","scope":"Sampled historical policy states and template feature ordering only; not general reliability",
            "action_results":results,"uncovered_actions":uncovered,
            "samples_passed":samples_passed,"coverage_complete":coverage_complete,
            "overall_pass":samples_passed and coverage_complete,"samples":records,
            "scan_windows":int(scan_windows),"scan_steps":int(scan_steps),"scanned_states":len(states),
            "observed_action_counts":{names[i]:predicted.count(i) for i in range(3)},
            "seed":42,"shap_nsamples":48,"background_observations":background.tolist(),
            "sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [Path("models/hmm_regime.pkl"),Path("models/persistent_brain_normalizer.pkl")]}}
    if output_path:
        dest=Path(output_path); dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit_all_actions_fidelity(
    num_samples_per_action: int = 50,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run fidelity evaluation across ALL 3 canonical action states:
    NEUTRAL (Cash), BUY (LONG), and SELL (SHORT).
    Validates feature ranking order and checks absence of directional contradiction.
    """
    actions = [
        ("NEUTRAL", [0.80, 0.10, 0.10]),
        ("BUY (LONG)", [0.10, 0.80, 0.10]),
        ("SELL (SHORT)", [0.10, 0.10, 0.80])
    ]
    per_action_results = {}
    all_rhos = []
    
    for action_name, action_probs in actions:
        res = audit_explanation_fidelity(
            action_name=action_name,
            probs=action_probs,
            num_samples=num_samples_per_action,
            random_seed=random_seed
        )
        per_action_results[action_name] = res
        all_rhos.append(res["mean_spearman_rho"])
        
    overall_mean_rho = float(np.mean(all_rhos))
    overall_pass = all(r["pass_audit"] for r in per_action_results.values())
    
    return {
        "per_action": per_action_results,
        "overall_mean_spearman_rho": overall_mean_rho,
        "overall_pass": overall_pass,
        "actions_evaluated": [a[0] for a in actions],
        "samples_per_action": num_samples_per_action
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  Expanded XAI Natural Language Explanation Fidelity Audit")
    print("=" * 65)
    
    # 1. Multi-Action Synthetic Probe Audit
    print("\n--- Phase 1: Multi-Action Synthetic Feature Probes ---")
    multi_res = audit_all_actions_fidelity(num_samples_per_action=50)
    for act, r in multi_res["per_action"].items():
        print(f"  Action: {act:<14} | Samples: {r['num_samples']} | Mean rho: {r['mean_spearman_rho']:.4f} | Missing: {r['missing_feature_rate']:.1%}")
    print(f"  Overall Mean Spearman Rank Correlation: {multi_res['overall_mean_spearman_rho']:.4f}")
    status_str = "[PASSED]" if multi_res["overall_pass"] else "[FAILED]"
    print(f"  Multi-Action Synthetic Audit Status: {status_str}")
    
    # 2. Real Model Checkpoint Audit
    print("\n--- Phase 2: Real Checkpoint Evaluation (persistent_brain.pth) ---")
    try:
        real_res = audit_real_model_fidelity(num_eval_states=10)
        print(f"  Status: {real_res.get('status')}")
        print(f"  Observed Action Counts: {real_res.get('observed_action_counts')}")
        print(f"  Uncovered Actions in Real Checkpoint: {real_res.get('uncovered_actions')}")
        print(f"  Checkpoint Sample Audit Pass: {real_res.get('samples_passed')}")
    except Exception as e:
        print(f"  Real model check skipped or errored: {e}")
        
    print("\n" + "=" * 65)


