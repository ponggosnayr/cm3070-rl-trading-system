> Status after focused repairs: historical results are preserved; empirical DSR, explanation fidelity and latency claims require new evidence. Automated checks do not certify rubric compliance.

# University of London CM3070 Final Project — Rubric Compliance Matrix

**Project Title:** Adaptive Multi-Market Reinforcement Learning Trading System & Financial Advisor Bot  
**Module:** CM3070 Computer Science Final Project (30 Credits)  
**Programme:** BSc (Hons) Computer Science, University of London  
**Template Reference:** 4.2 — Financial Advisor Bot (Level 6 CM3020 Artificial Intelligence)  
**Academic Year:** 2025–2026  

---

## 1. Executive Summary

This document provides external markers, second examiners, and audit evaluators with an explicit, section-by-section compliance matrix cross-referencing the project deliverables against:
1. **The 6 Module Learning Outcomes (LO1–LO6)** of the CM3070 Final Project specification.
2. **The Project Template Reference 4.2 (Financial Advisor Bot)** guidelines and grade descriptors.
3. **The First-Class Honours (70–100% / 80–100% Outstanding)** criteria defined in the University of London Programme Regulations.

Every claim, software module, test suite, and evaluation result is indexed directly to source code paths and report sections.

---

## 2. CM3070 Module Learning Outcomes (LO1 – LO6) Mapping

| Learning Outcome | Syllabus Descriptor | Project Implementation & Evidence | Code & Artifact Reference | Report Section |
|---|---|---|---|---|
| **LO1: Selection of Appropriate Techniques** | Select and apply appropriate Computer Science techniques to a particular problem. | Applied deep actor-critic reinforcement learning (PPO with clipped surrogate objective), multi-head self-attention Transformer encoders for temporal time-series feature extraction, Gaussian hidden Markov models (HMM) for market regime detection, and KernelExplainer SHAP for post-hoc feature attribution. | [`src/rl_env.py`](../src/rl_env.py)<br>[`src/train_agent.py`](../src/train_agent.py)<br>[`src/xai_shap.py`](../src/xai_shap.py) | Section 2.1–2.4<br>Section 4.1–4.4 |
| **LO2: Project Proposal Development** | Develop a project proposal that can be addressed using Computer Science techniques. | Defined research questions on downside capital preservation and explainability fidelity; structured feasibility study, risk management matrix, hardware constraints (CUDA/MPS/CPU), and ethical considerations regarding financial autonomy. | [`docs/PROJECT_PLAN.md`](PROJECT_PLAN.md)<br>[`docs/PROJECT_CALENDAR.md`](PROJECT_CALENDAR.md) | Section 1.2–1.4 |
| **LO3: Literature Review & Prior Work** | Evaluate previous work in areas related to chosen project and write a literature review. | Critically appraised algorithmic trading paradigms, FinRL ensemble models (Yang et al., 2020), temporal Transformers (Vaswani et al., 2017), backtest overfitting hazards (Bailey & López de Prado, 2014), Moody's differential Sharpe ratio (Moody & Saffell, 2001), and XAI frameworks (Lundberg & Lee, 2017). | [`FINAL_PROJECT_REPORT.pdf`](FINAL_PROJECT_REPORT.pdf) (37 formal academic citations) | Section 2 (Literature Review) |
| **LO4: Substantial Software Engineering** | Design and develop a substantial piece of software matching the project brief. | Built an end-to-end full-stack system: vectorized Gymnasium market environment (`ActiveCryptoEnv`), 4-layer PyTorch Transformer policy, asynchronous FastAPI backend (`api.py`), developer training GUI (`train_gui.py`), and interactive Vite React TypeScript dashboard. | [`api.py`](../api.py)<br>[`train_gui.py`](../train_gui.py)<br>[`frontend/src/App.tsx`](../frontend/src/App.tsx) | Section 3 (System Design)<br>Section 4 (Implementation) |
| **LO5: Rigorous Testing & Evaluation** | Test and evaluate software in terms of user needs, software correctness, and efficiency. | 248 automated tests passed in the latest local run; five walk-forward folds, historical cross-market comparisons, simulations, an internal heuristic inspection and an author-led cognitive walkthrough provide further evaluation. Formal empirical DSR, independent user outcomes and real-model explanation coverage across all actions remain unverified. | [`tests/`](../tests/)<br>[`verify_submission.py`](../verify_submission.py)<br>[`docs/USER_EVALUATION_STUDY.md`](USER_EVALUATION_STUDY.md) | Section 5 (Evaluation) |
| **LO6: Comprehensive Academic Reporting** | Report results of the project in written, oral, and visual form. | Report source contains six chapters, architecture figures, results tables and critical limitations. A demonstration video is prepared, and the final PDF includes the public repository URL. | [`FINAL_PROJECT_REPORT.pdf`](../FINAL_PROJECT_REPORT.pdf)<br>[`docs/presentation_slides/`](presentation_slides/) | Complete report |

---

## 3. Template Reference 4.2 (Financial Advisor Bot) Specification Compliance

The table below cross-references each specific requirement stated in the University of London Template 4.2 project brief:

| Template 4.2 Requirement | Template Brief Specification | Implemented Architectural Solution | Code / Artifact Location |
|---|---|---|---|
| **Financial System Scope** | *"Identify which kind of financial systems the bot will advise about (e.g. stock market, crypto-currencies)..."* | System models 24/7 cryptocurrency markets (BTC, ETH, DOGE) as in-domain continuous trading, alongside zero-shot cross-market evaluation across equity index ETFs (SPY, QQQ) and commodities (WTI Crude, Spot Gold, Natural Gas). | [`data/`](../data/) (29 OHLCV datasets)<br>[`data/cross_market_results.csv`](../data/cross_market_results.csv) |
| **AI / ML Decision Paradigm** | *"Modelling the problem as 'decision making under uncertainty' would lend itself to reinforcement learning..."* | Formulated as a Markov Decision Process (MDP) solved via Proximal Policy Optimization (PPO) with a custom 4-layer multi-head self-attention Transformer feature extractor operating on an 8-step frame-stacked observation tensor (136 dimensions). | [`src/rl_env.py`](../src/rl_env.py) lines 35–150<br>[`src/train_agent.py`](../src/train_agent.py) |
| **Separation of Concerns** | *"You will probably need separate systems for training models and gathering data... We would not expect non-technical users to interact with training..."* | Enforced strict architectural decoupling: Developer Training & Validation Toolkit (`train_gui.py`, `train_agent.py`, `walk_forward_eval.py`) is physically segregated from the user-facing web dashboard (`frontend/` + `api.py`). Non-technical users interact only with pre-trained inferences. | [`train_gui.py`](../train_gui.py)<br>[`api.py`](../api.py)<br>[`frontend/`](../frontend/) |
| **User Interface for Non-Technical Users** | *"It should be possible for a non-technical user to interact with the bot to receive advice, for example, via a web interface..."* | Developed a modern Vite React TypeScript single-page application featuring 6 modular tabs: Market Overview, Dynamic Backtest Visualizer, HMM Regime Classifier, SHAP Importance Hub, Advisor Chatbot, and 'Model School' Educational Hub. | [`frontend/src/App.tsx`](../frontend/src/App.tsx)<br>[`Launch_Dashboard.bat`](../Launch_Dashboard.bat) |
| **Explainability & Grounding** | *"The bot should present its analysis and recommendations to the user with explanations... You could consider using language models..."* | Integrated post-hoc SHAP attribution (`xai_shap.py`) with an LLM conversational agent (Google Gemini with deterministic offline fallback) mediated by a rule-based polarity consistency checker (`xai_engine.py`) with limited validation; empirical reliability remains to be established. | [`src/xai_shap.py`](../src/xai_shap.py)<br>[`src/xai_engine.py`](../src/xai_engine.py)<br>[`src/xai_audit.py`](../src/xai_audit.py) |
| **Backtesting & Robust Evaluation** | *"You might need a backtesting system to test your advisor's trading strategies... present evidence that you have evaluated the advice..."* | Next-candle execution reduces same-candle look-ahead risk and models 0.1% fees plus 0.05% slippage. Evaluation includes five walk-forward folds and DQN, A2C and Buy-and-Hold comparisons. DSR/PSR code is checked with illustrative inputs; a formal trial-adjusted empirical DSR was not established. | [`src/trading_utils.py`](../src/trading_utils.py)<br>[`src/walk_forward_eval.py`](../src/walk_forward_eval.py)<br>[`src/utils/dsr.py`](../src/utils/dsr.py) |

---

## 4. University of London Grade Descriptors Alignment

### Criteria for First Class Honours (70–100%) & Outstanding (80–100%)

According to the University of London grading descriptors for the CM3070 Final Project:

1. **Exceptional Technical Achievement:**
   - Standard undergraduate submissions typically implement basic tabular Q-learning or simple Multi-Layer Perceptrons (MLPs) on synthetic or unadjusted price data.
   - This project implements a **Gymnasium environment** with next-candle execution, clipped log-return and drawdown reward terms, a **custom 4-layer PyTorch Transformer extractor**, and HMM market regime classification.
   - **248 automated tests** passed in the latest local run, covering financial calculations, API behavior and explanation checks.

2. **Creative Thought & Novel Synthesis:**
   - Addressed transaction-cost churn with an action cooldown and tested the trade-off introduced by drawdown penalties; Cash-only convergence remains a documented failure mode.
   - Synthesized **Explainable AI (SHAP)** with **conversational NLP** through a deterministic polarity validator, achieving an audited Spearman rank correlation with empirical fidelity pending verification, and testing for conflicting trade advice.

3. **Critical Evaluation & Scientific Honesty:**
   - Applied **5-fold expanding Walk-Forward Validation** and disclosed single-seed and checkpoint-selection limitations. Illustrative DSR calculations do not establish empirical trial-adjusted significance.
   - Transparently documented policy behavior during bull market regimes (Folds 3 and 4), explaining why downside risk penalties induce flat positioning and discussing regime-conditional reward adaptation as future work.

4. **Professional Software Engineering & Reproducibility:**
   - Provided automated verification (`python verify_submission.py` and `Verify_Submission.bat`); runtime depends on the machine and installed dependencies.
   - Maintained clean separation of concerns, complete documentation, strict typing, and zero unresolved `TODO` or `FIXME` tokens.

---

## 5. Verification Checklist for Markers

To verify all claims made in the dissertation and code submission:

```bash
# 1. Run the one-click automated audit (executes all 7 stages):
python verify_submission.py

# 2. Run the unit and adversarial test suite:
pytest tests/ -v

# 3. Test the XAI explanation fidelity audit script:
python src/xai_audit.py

# 4. Compile the academic report from LaTeX source:
cd docs
xelatex -interaction=nonstopmode DRAFT_FINAL_PROJECT_REPORT_V2.tex

# 5. Launch the full-stack web application:
# (Root directory) Launch_Dashboard.bat
```
