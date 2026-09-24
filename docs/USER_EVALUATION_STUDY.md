# Interface Usability Evaluation: Cognitive Walkthrough & Proposed Empirical Study Protocol

**Project Title:** Adaptive Multi-Market Reinforcement Learning Trading System & Financial Advisor Bot  
**Module:** CM3070 Computer Science Final Project  
**Scope:** Formative Cognitive Walkthrough (Completed) & Empirical User Study Protocol (Proposed)  
**Date:** September 2026  
**Methodological Reference:** Wharton et al. (1994) Cognitive Walkthrough; Nielsen (1994) Usability Engineering  

---

## 1. Executive Summary & Study Framing

Under **Project Template Reference 4.2 (Financial Advisor Bot)**, an essential objective is delivering an interface that communicates model decisions, risk postures, and data states accessibly to non-technical retail users.

To evaluate interface learnability, error recovery, and transparency, usability analysis was partitioned into two distinct components:
1. **Formative Cognitive Walkthrough (Completed):** An author-conducted usability inspection assisted by AI tooling during development, stepping through standardized workflows using five representative retail personas to identify and resolve interaction hurdles (Wharton et al., 1994).
2. **Empirical Multi-Participant Study Protocol (Proposed Future Work):** A formal, task-based laboratory protocol designed for independent human participants ($N \ge 30\text{--}40$) to measure quantitative task completion rates, time-on-task, and standardized System Usability Scale (SUS) benchmarks.

**Academic Integrity Declaration:**  
*No independent participant study was conducted.* All empirical interface improvements documented herein were derived through author-led cognitive walkthroughs, iterative frontend development, and automated integration checks. No synthetic or simulated task times, completion percentages, or SUS questionnaire scores are presented as measured human user evidence.

---

## 2. Author-Led Formative Cognitive Walkthrough (Completed)

### 2.1 Inspection Methodology
The cognitive walkthrough evaluates the **learnability** of unfamiliar user interfaces (Wharton et al., 1994). The author evaluated the six views of the deployed web platform (`http://localhost:5173`) by stepping through three core user tasks, analyzing each step against four canonical usability questions:
1. *Will the user try to achieve the right effect?*
2. *Will the user notice that the correct action is available?*
3. *Will the user associate the correct action with the intended effect?*
4. *If the correct action is performed, will the user see that progress is being made?*

To simulate real-world usage without human participants, the inspection adopted five non-technical retail personas as analytical lenses:
* **Persona 1 (Novice Saver):** No prior financial experience; familiar only with consumer chatbots.
* **Persona 2 (Occasional ETF Investor):** Understands basic equities; unfamiliar with quantitative indicators or leverage.
* **Persona 3 (Junior Software Developer):** Technical computing background; basic machine learning knowledge; novice trader.
* **Persona 4 (Administrative Professional):** Practical retail saver; sensitive to capital loss and jargon overload.
* **Persona 5 (Active Retail Trader):** 3+ years self-directed retail trading; values speed and execution transparency.

### 2.2 Standardized Task Paths Evaluated
* **Task 1 (Advice & Stance Interpretation):** Navigate to the advisor view and determine whether the bot recommends buying, shorting, or holding cash.
* **Task 2 (Explainability & Uncertainty Comprehension):** Inspect the SHAP attribution chart and identify the primary technical indicator driving the recommendation and whether it exerts positive or negative pressure.
* **Task 3 (Data Freshness & Offline Mode Recognition):** Determine whether displayed indicators reflect live exchange feeds or cached historical data during a simulated network disconnect.

### 2.3 Qualitative Cognitive Friction Points Identified & Resolved

Through this walkthrough, three primary interaction obstacles were identified and engineered out of the frontend:

```
+-----------------------------------------------------------------------------------------+
|                         COGNITIVE WALKTHROUGH REPAIR CYCLE                              |
|                                                                                         |
|   [Identified Barrier]       --> [Cognitive Root Cause]  --> [Implemented Resolution]   |
|   U-01: Technical Feature        Raw indicator names         Plain-English aliases and  |
|         Jargon in SHAP           (e.g., momentum_rsi_14)     interactive InfoBadges     |
|                                                                                         |
|   U-02: Action Ambiguity         "NEUTRAL" could mean        Explicit "Target Cash"     |
|         for Inaction Signals     holding or liquidating      and "Stay in Cash" badges  |
|                                                                                         |
|   U-03: Subdued Stale Data       Subtle header dot           High-visibility amber      |
|         Warning Status           lacked visual prominence    "OFFLINE (CACHE)" badge    |
+-----------------------------------------------------------------------------------------+
```

1. **Defect U-01 (Feature Jargon Cognitive Barrier):**  
   * *Walkthrough Observation:* In Task 2, displaying raw variable identifiers such as `momentum_rsi_14` or `volatility_ratio` presented cognitive barriers for non-technical personas (P1, P4), creating confusion over whether red attribution bars signified a price drop or an increase in market volatility.
   * *Implemented Fix:* Built human-readable feature aliases (`Relative Strength (RSI)`, `Market Volatility Ratio`) and added dynamic natural-language attribution summaries above the SHAP waterfall chart alongside interactive `InfoBadge` tooltips.
2. **Defect U-02 (Action Clarity for Defensive Postures):**  
   * *Walkthrough Observation:* In Task 1, displaying a bare `NEUTRAL` signal left ambiguity over whether the portfolio was maintaining existing long positions or holding 100% fiat cash.
   * *Implemented Fix:* Updated the interface across all views with explicit labels: `Target Cash` in Card 1, `Stay in Cash` in plain signals, and `0 trades (100% Cash Defense)` in the backtest visualizer.
3. **Defect U-03 (System Status Visibility in Failure Modes):**  
   * *Walkthrough Observation:* In Task 3, simulated feed disconnections produced a subtle grey indicator that could easily be missed by a retail user.
   * *Implemented Fix:* Implemented a high-visibility amber status badge (`OFFLINE (CACHE)`) in market cards and an explicit `Historical Replay (Offline)` / `Cached 5m Stream` indicator on candlestick charts.

---

## 3. Proposed Empirical Usability Study Protocol (Future Work)

To enable rigorous, independent human-subject evaluation in future research, this section specifies a complete, standardized usability testing protocol.

### 3.1 Participant Recruitment Profile
* **Target Sample Size:** $N = 30\text{--}40$ independent participants, providing statistical power for quantitative benchmark comparisons (Sauro & Lewis, 2016).
* **Demographic Cohorts:** Stratified into three experience tiers:
  1. *Novice Retail Investors (40%):* No prior algorithmic trading or machine learning experience.
  2. *Intermediate Retail Traders (40%):* Self-directed retail equity/crypto investors.
  3. *Technical / Quantitative Specialists (20%):* Software engineers or data analysts.
* **Inclusion Criteria:** Age $\ge 18$; basic familiarity with web browsers; willingness to complete paper-trading scenarios.

### 3.2 Standardized Task Execution Script
Participants will be seated at a standardized terminal with the application deployed locally, using think-aloud verbalization while executing three timed tasks:

| Task ID | Task Description | Target Success Benchmark | Error Definition |
|---|---|---|---|
| **Task 1** | Identify current model recommendation (Buy / Short / Cash) and policy confidence level. | Time $\le 30$ seconds; 100% accurate identification | Misidentifying Cash as active Long; failing to locate confidence gauge. |
| **Task 2** | Inspect top SHAP feature and explain whether it promotes or discourages trading. | Time $\le 60$ seconds; $\ge 90\%$ directional comprehension | Conflating negative volatility weight with a negative price drop. |
| **Task 3** | Recognize system state when market feeds disconnect during simulated server downtime. | Time $\le 45$ seconds; 100% state awareness | Believing cached offline data is a live market feed. |

### 3.3 Standardized Measurement Instruments
1. **Time-on-Task & Success Rates:** Logged via automated screen recordings and timestamped user interaction logs.
2. **System Usability Scale (SUS):** Administered immediately following task completion using the standard 10-item Likert survey (Brooke, 1996):
   * Q1: I think that I would like to use this system frequently.
   * Q2: I found the system unnecessarily complex.
   * Q3: I thought the system was easy to use.
   * Q4: I think that I would need the support of a technical person to be able to use this system.
   * Q5: I found the various functions in this system were well integrated.
   * Q6: I thought there was too much inconsistency in this system.
   * Q7: I would imagine that most people would learn to use this system very quickly.
   * Q8: I found the system very cumbersome to use.
   * Q9: I felt very confident using the system.
   * Q10: I needed to learn a lot of things before I could get going with this system.
3. **Benchmarking Standard:** Scored using the Sauro-Lewis Curved Grading Scale (Sauro & Lewis, 2016), evaluating whether the platform achieves Grade A usability ($>80.8$ points, 90th percentile).

---

## 4. Conclusion & Methodological Boundary

1. **Clear Division of Contributions:** This document separates the **author-led formative cognitive walkthrough** completed during system development from the **empirical laboratory protocol** proposed for future human trials.
2. **Verifiable Engineering Impacts:** The completed cognitive walkthrough directly guided three concrete interface enhancements in `frontend/src/App.tsx`: plain-English attribution aliases, unambiguous cash defense labels, and explicit offline status badges.
3. **Scientific Honesty:** By eliminating synthetic timing and questionnaire figures, this report maintains strict adherence to academic research integrity.
