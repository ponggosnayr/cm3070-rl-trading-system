# Evaluation and data provenance

[Back to the project overview](../README.md)

## What the saved results show

The repository preserves several historical experiments. They are useful for inspecting behaviour and limitations, but should not be treated as one consistently controlled benchmark.

| Evidence | Observation in the saved file |
| --- | --- |
| [Walk-forward results](../walk_forward_results.csv) | Five sequential folds. Folds 3 and 4 recorded zero trades while buy-and-hold rose, showing missed opportunities when the policy remained in cash. |
| [Baseline comparison](../comparison_results.csv) | PPO and A2C recorded zero trades in this saved comparison; their zero return should not be presented as evidence of active trading skill. |
| [Cross-market results](../data/cross_market_results.csv) | Outcomes vary substantially by asset, including positive crypto returns and losses on SPY and QQQ. The table does not establish reliable transfer across markets. |

Returns, drawdowns and trade counts need to be interpreted together. A low drawdown caused by taking no positions is different from robust risk management while trading. Drawdown signs also vary between saved tables; inspect each file's convention before comparing them.

## Historical inputs versus current code

The exact saved runs **cannot be reproduced from this trimmed repository**:

- The walk-forward CSV records fold boundaries for **57,756 BTC rows with no purge gap**.
- The bundled BTC hourly CSV contains **58,479 rows**.
- The current walk-forward script defaults to a **24-row purge window**, rather than the boundaries in the historical CSV.
- Original run seeds, exact source snapshots and baseline/cross-market manifests were not retained.
- Some assets in historical cross-market outputs are not included in the current demo datasets.

The current evaluation implementation fits HMM/preprocessing within each training fold and evaluates using frozen fold parameters. That describes the current code; it does not retroactively establish the settings of the saved experiments.

## Supplementary frozen-policy replay

The [27 September 2026 replay](../experiments/frozen_replay_2026_09_27/README.md) retains fixed windows, exact input hashes, runtime versions, a runnable harness and two identical sets of metrics/traces. The saved demo policy chose Long throughout all three windows and underperformed cost-adjusted Buy-and-Hold in each. This is a repeatability check with disclosed holding-rule differences and unknown training exposure, not an unseen test or a reproduction of the historical results above.

## Market data and the dashboard

Bundled market CSVs are frozen snapshots for the local demo, not live quotes. Some dashboard paths attempt to obtain current prices or candles from external services, so network availability affects what can be shown.

When a live five-minute feed is unavailable, the chart reconstructs illustrative five-minute candles from archived hourly bars and labels them with the source dates. Those reconstructed candles are not independently observed five-minute market data.

The SPY and QQQ daily files contain one midnight UTC candle per market date. Duplicate Yahoo daily timestamps at 04:00/05:00 UTC were removed in favour of the corresponding midnight rows. Every retained row, including its OHLCV values and saved indicators, was preserved.

## What software checks establish

Automated tests check implementation behaviour and saved-result consistency. They do not establish profitability, clean-install compatibility on every platform, continuous live-feed availability or exact reproduction of historical experiments.

SHAP displays feature contributions to policy action scores. The synthetic attribution audit in `src/xai_audit.py` is a diagnostic; it does not establish explanation fidelity for every action of the trained model. Advisor text should be read alongside the underlying numbers and data dates.

## Next research steps

- Retain complete run manifests and frozen inputs for repeatable comparisons.
- Evaluate regime-balanced training and the no-trade behaviour on unseen periods.
- Validate cross-market transfer under consistent time ranges, costs and sampling.
- Test explanations against the actual policy across actions and market regimes.

This is a research and software-engineering project, not a production trading service or investment recommendation.
