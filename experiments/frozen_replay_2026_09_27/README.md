# Frozen-policy replay: 27 September 2026

A small, auditable supplementary check of the saved demo policy. This is retrospective replay, **not an unseen test** or a reproduction of the report's historical walk-forward results. No model was trained or selected during this check.

## What was fixed before inference

`protocol.json` records the input SHA-256 hashes, original source commit (`dbf9a0d`), all simulator settings overridden by the runner, and exact row/date boundaries. All remaining simulator defaults come from the hashed `src/rl_env.py`. The last 2,160 rows were split into three consecutive 720-row windows, without selecting for return. Timestamps are retained exactly as stored; no timezone conversion is applied.

The policy, observation normaliser and HMM are frozen. Indicators are recomputed over the frozen OHLCV history. Each window resets its environment context and eight-frame observation stack. Inference uses CPU, one thread, seed 42, evaluation mode and masked deterministic argmax. Normaliser updates and reward normalisation are disabled.

Capital starts at $10,000. Commission is 0.1% and slippage 0.05% per side. The policy simulator imposes a 48-hour position limit and a one-step cooldown. Its evaluation drawdown cap is 100%, so this replay does not measure the training configuration's protective stop.

Cash earns zero. Buy-and-Hold enters at the next candle open and liquidates at the final close with the same costs, without the policy's holding-duration rule. An independent always-long simulator check agrees with the analytic baseline terminal balance within one cent in each window. Maximum drawdown includes initial capital and terminal liquidation.

## Observed results

| Window | Dates in 2026 | PPO ROI | Buy-and-Hold ROI | PPO max drawdown | B&H max drawdown |
|---|---|---:|---:|---:|---:|
| W1 | 9 June–9 July | -2.85% | +1.33% | 15.30% | 13.38% |
| W2 | 9 July–8 August | -1.46% | +2.77% | 8.38% | 6.45% |
| W3 | 8 August–7 September | +15.80% | +20.77% | 6.23% | 5.10% |

Cash return and drawdown were zero throughout. All 2,157 policy decisions were Long, with 15 entries per window as positions were closed and reopened. PPO underperformed Buy-and-Hold in all three windows. Different holding rules prevent attributing this gap solely to policy quality or transaction costs.

Two complete runs produced byte-identical `results.csv` and `policy_trace.csv`; see `repeatability.json`. This supports deterministic replay in the recorded runtime, not independent-seed robustness or performance generalisation. Training exposure to these dates is unknown. One checkpoint, one asset and three short contiguous windows cannot establish broad trading effectiveness.

## Re-run

Use a checkout containing the exact inputs named in `protocol.json`. Source commit `dbf9a0d525e9141c67052ed8f1aeb8928a62a779` identifies those inputs; later documentation-only commits do not change them. Hashes deliberately fail on modified files. The recorded Windows checkout uses CRLF for the two source files and CSV. For a fresh checkout, use `git -c core.autocrlf=true clone https://github.com/ponggosnayr/cm3070-rl-trading-system.git` so text input bytes match.

The tested interpreter was Python 3.12.14 on Windows with the package versions in `run1/runtime.json`. In particular, the run used stable-baselines3 2.9.0; the main project's requirements currently specify 2.8.0. This is **not** a fresh-install verification of those requirements. Reproduction on another platform or library version has not been checked.

From the repository root, with the recorded runtime available:

```text
python experiments/frozen_replay_2026_09_27/replay_check.py --repo . --protocol experiments/frozen_replay_2026_09_27/protocol.json --out output/frozen-replay-run1
python experiments/frozen_replay_2026_09_27/replay_check.py --repo . --protocol experiments/frozen_replay_2026_09_27/protocol.json --out output/frozen-replay-run2
```

Compare the SHA-256 hashes of each run's `results.csv` and `policy_trace.csv` against the retained outputs. Each `runtime.json` also records runner/protocol hashes, package versions, unchanged-input verification and the independent baseline check. Use normal Python execution, not `python -O`, because input and baseline checks use assertions.

The report remains a separate private submission document. This folder contains only the replay method and observed evidence.
