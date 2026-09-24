"""Read-only inspection of saved Tiny PPO checkpoints and evaluation inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_project_env = Path(__file__).resolve().parents[2] / "env" / "Lib" / "site-packages"
sys.path.insert(0, str(_project_env))

import numpy as np
import pandas as pd
import torch

from run_tiny_diagnostic import PROJECT_ROOT, _vec, build_model
from trading_utils import compute_indicators


def inspect_policy(checkpoint: Path, frame, stride: int = 50) -> dict:
    env = _vec(frame, True, .25, max_steps=len(frame))
    model = build_model(env, seed=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(state["policy"], strict=True)
    model.policy.set_training_mode(False)
    obs = env.reset()
    probabilities = []
    observations = []
    done = np.array([False])
    step = 0
    while not bool(done[0]):
        if step % stride == 0:
            tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                probs = model.policy.get_distribution(tensor).distribution.probs
            probabilities.append(probs.cpu().numpy()[0])
            observations.append(obs[0].reshape(8, -1)[-1].copy())
        obs, _, done, _ = env.step(np.array([0]))
        step += 1
    env.close()
    p = np.asarray(probabilities)
    x = np.asarray(observations)
    return {
        "sampled_states": len(p),
        "mean_action_probability_cash_long_short": np.round(p.mean(axis=0), 4).tolist(),
        "median_action_probability_cash_long_short": np.round(np.median(p, axis=0), 4).tolist(),
        "max_long_short_probability": p[:, 1:].max(axis=0).tolist(),
        "mean_long_short_probability": p[:, 1:].mean(axis=0).tolist(),
        "deterministic_action_counts_cash_long_short": np.bincount(p.argmax(axis=1), minlength=3).tolist(),
        "cash_probability_range": np.round([p[:, 0].min(), p[:, 0].max()], 4).tolist(),
        "feature_std_latest_frame": np.round(x.std(axis=0), 4).tolist(),
        "feature_abs_max_latest_frame": np.round(np.abs(x).max(axis=0), 4).tolist(),
    }


def main() -> None:
    root = PROJECT_ROOT / "models/experiments/tiny_diagnostic_500k"
    output = {}
    for run in sorted(root.glob("*/manifest.json")):
        folder = run.parent
        manifest = json.loads(run.read_text())
        seed = manifest["settings"]["seed"]
        raw = pd.read_csv(PROJECT_ROOT / "data/btc_usdt_1h.csv")
        dates = pd.to_datetime(raw["datetime"] if "datetime" in raw else raw["timestamp"])
        hmm = str(folder / "hmm_regime.pkl")
        train = compute_indicators(raw[dates < "2025-04-01"].reset_index(drop=True).iloc[:-48].reset_index(drop=True), fit_hmm=False, hmm_artifact_path=hmm)
        bull = compute_indicators(raw[(dates >= "2025-04-01") & (dates < "2025-11-01")].reset_index(drop=True), fit_hmm=False, hmm_artifact_path=hmm)
        bear = compute_indicators(raw[(dates >= "2025-11-01") & (dates < "2026-06-01")].reset_index(drop=True), fit_hmm=False, hmm_artifact_path=hmm)
        output[str(seed)] = {}
        for name in ("harsh", "gradual"):
            checkpoint = folder / f"{name}_seed{seed}.pth"
            output[str(seed)][name] = {
                label: inspect_policy(checkpoint, frame)
                for label, frame in (("train", train), ("bull", bull), ("bear", bear))
            }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
