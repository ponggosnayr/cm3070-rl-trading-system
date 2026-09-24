# Expanded XAI Explanation Ordering Audit

**Project Title:** Adaptive Multi-Market Reinforcement Learning Trading System & Financial Advisor Bot  
**Module:** CM3070 Computer Science Final Project  
**Date:** September 2026  
**Script Reference:** [`src/xai_audit.py`](../src/xai_audit.py)  
**Evaluation Target:** Spearman Rank Correlation ($\rho \ge 0.95$) & Action-Explanation Polarity Consistency  

---

## 1. Executive Summary & Audit Motivation

The initial project submission reported an XAI feature ordering correlation of $\rho = 1.00$ evaluated over a narrow sample of 9 BUY/LONG action states. As identified in the project assessment:
1. Evaluating only BUY/LONG states leaves NEUTRAL (Cash) and SELL (SHORT) unverified.
2. Synthetic test probes were not clearly partitioned from real model checkpoint inferences.

This expanded audit explicitly addresses these limitations by:
1. Running systematic synthetic multi-action probes across all three canonical actions: **NEUTRAL**, **BUY (LONG)**, and **SELL (SHORT)** ($N=150$ total evaluations).
2. Probing the production model checkpoint (`models/persistent_brain.pth`) across historical evaluation windows to document genuine action distribution.

---

## 2. Phase 1: Multi-Action Synthetic Feature Probes

To test explanation ordering across *all* action labels, we generated 50 random attribution-like score arrays per action ($N=150$). These are **not SHAP values computed from a trained model**; this phase tests the explanation engine's ordering logic only.

### 2.1 Quantitative Results

| Action Class | Evaluated Samples | Target Threshold | Mean Spearman $\rho$ | Min Spearman $\rho$ | Missing Feature Rate | Audit Status |
|---|---|---|---|---|---|---|
| **NEUTRAL (Cash)** | 50 | $\rho \ge 0.95$ | **0.9973** | 0.9850 | **0.00%** | **PASSED** |
| **BUY (LONG)** | 50 | $\rho \ge 0.95$ | **1.0000** | 1.0000 | **0.00%** | **PASSED** |
| **SELL (SHORT)** | 50 | $\rho \ge 0.95$ | **1.0000** | 1.0000 | **0.00%** | **PASSED** |
| **Overall Multi-Action** | **150** | $\rho \ge 0.95$ | **0.9991** | **0.9850** | **0.00%** | **PASSED** |

### 2.2 Key Findings
* **Rank Preservation:** Across all three action labels, the text presented the top three injected features in score order ($\text{Overall } \rho = 0.9991$).
* **Zero Missing Features:** All three injected top features appeared in every tested explanation.
* **Directional Context:** For `NEUTRAL` actions, feature impacts are contextualized defensively (e.g., elevated volatility or negative trend discouraging market entry). For `BUY` actions, features emphasize momentum and support. For `SELL` actions, features highlight overbought signals or regime downturns.

---

## 3. Phase 2: Real Checkpoint Evaluation (`persistent_brain.pth`)

### 3.1 Historical Market State Probing
We evaluated 200 historical hourly states across Bitcoin market data using the frozen production checkpoint via `audit_real_model_fidelity`:
* **Observed Action Distribution (192 evaluated steps after warm-up):**
  * `BUY (LONG)`: 192 states (100.0%)
  * `NEUTRAL (Cash)`: 0 states (0.0%)
  * `SELL (SHORT)`: 0 states (0.0%)
* **Uncovered Actions in Real Checkpoint:** `NEUTRAL`, `SELL (SHORT)`

### 3.2 Analysis of Real-Model Action Asymmetry
In this selected historical slice, the production checkpoint chose `BUY (LONG)` on all 192 evaluated steps, leaving Cash and Short unexercised. Other walk-forward and validation results use different policies or checkpoints and cannot fill this real-model explanation-coverage gap. Synthetic probes check software behavior when those labels are supplied; they do not establish faithful model explanations for unobserved actions.

---

## 4. Methodological Distinction: Synthetic vs. Real-Model Findings

To ensure scientific honesty in the final dissertation:

```
+-----------------------------------------------------------------------------------------+
|                               XAI EVALUATION PARTITION                                  |
|                                                                                         |
|   [Synthetic Feature Probes]                 [Real Checkpoint Inference]                |
|   * Evaluates: Template ordering logic       * Evaluates: Deployed model behavior       |
|   * Coverage: NEUTRAL, LONG, SHORT (100%)    * Coverage: 100% BUY (LONG) in this slice  |
|   * Result: rho = 0.9991 (PASSED)            * Result: NEUTRAL & SHORT unexercised here |
+-----------------------------------------------------------------------------------------+
```

1. **Synthetic Probes demonstrate:** The explanation engine's ordering logic and template substitution work correctly across all three action classes (LONG, NEUTRAL, SHORT) when triggered.
2. **Real Checkpoint Probes demonstrate:** In this 192-hour test slice, the model maintained continuous Long exposure (192 BUY, 0 NEUTRAL, 0 SHORT). Out-of-sample evaluations elsewhere demonstrated positive active returns in falling markets (Fold 2: +44.51%, 65 trades; Fold 5: +4.73%, 2 trades), persistent cash holding during rising markets (Folds 3 & 4: 0 trades), and cash-only preservation in the developer bear benchmark (0 trades, 0.00% drawdown).

---

## 5. Conclusion

This evaluation shows that synthetic ordering checks passed across Cash, Long and Short labels ($\rho = 0.9991$). The 192-step real-checkpoint slice covered Long only. Real-model explanation fidelity for Cash and Short remains unverified.
