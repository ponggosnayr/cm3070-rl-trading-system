> Historical results are preserved; empirical DSR, broad explanation fidelity and latency claims still need new evidence. Automated checks do not certify rubric compliance.

# 📈 Adaptive Multi-Market RL Trading System

**CM3070 Final Project — BSc Computer Science**

A quantitative, AI-powered cryptocurrency trading system utilizing **Reinforcement Learning (PPO)** and deep temporal feature extraction (**Transformers**). The system emphasizes robust backtesting, risk management, and quantitative metrics (Sharpe Ratio, Max Drawdown) over simplistic rule-based logic.

---

## 📋 Prerequisites

- **Python 3.11+** (Python 3.12 recommended; required by pinned dependencies)
- **Node.js 20.19+ or 22.12+** & **npm** (required for building and running the Vite React frontend)

## ⚙️ Setup Instructions

### 1. Clone the Project
```bash
git clone https://github.com/ponggosnayr/cm3070-rl-trading-system.git
cd cm3070-rl-trading-system
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Run the System

To start the system, you need to run both the FastAPI backend and the Vite React frontend.

#### Start the Backend API:
```bash
python api.py
```
The backend will launch locally at `http://127.0.0.1:8000`.

#### Start the React Frontend:
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
The frontend will launch at `http://localhost:5173`.

#### Alternative Launcher (Windows)
After installing the Python dependencies and running `npm install` in `frontend/`, Windows users can double-click [`Launch_Dashboard.bat`](Launch_Dashboard.bat). It starts the local backend and frontend and opens `http://localhost:5173`.

---

## 🧪 One-Click Marker Verification & Automated Audit

For markers and evaluators verifying the submission against University of London CM3070 rubric criteria, run the automated 7-stage verification suite:

```bash
python verify_submission.py
```
*(On Windows, you can alternatively double-click [`Verify_Submission.bat`](Verify_Submission.bat). Build the frontend with `npm run build` first.)*

This script automatically executes:
1. **Environment & Dependencies:** Validates PyTorch, Stable-Baselines3, Gymnasium, and FastAPI.
2. **Codebase Compilation:** Compiles the packaged Python source files to check syntax.
3. **Checkpoints & Datasets:** Validates model weights (`models/persistent_brain.pth`) and OHLCV datasets.
4. **Automated Test Suite:** Runs the complete test suite and reports the current result.
5. **Empirical Convergence:** Cross-checks raw CSV results against report Tables 4, 5, and 6.
6. **Reproducible Math & XAI:** Checks illustrative DSR/PSR calculations and tests explanation ordering with synthetic feature-score arrays. These checks do not establish empirical trial-adjusted DSR or real-model explanation fidelity across all actions.
7. **Deliverables Check:** Verifies production frontend bundle and updated academic dissertation (`FINAL_PROJECT_REPORT.pdf`).

> [!NOTE]
> For the syllabus and grade-descriptor cross-reference, see [`docs/RUBRIC_COMPLIANCE_MATRIX.md`](docs/RUBRIC_COMPLIANCE_MATRIX.md). The matrix is a self-assessment, not a grading result.

---

## 🚀 How to Use & Platform Architecture

The web platform is organized into an institutional **Tri-Pane Interface** designed for both retail accessibility and quantitative audit:

1. **Left Navigation Rail (Asset Watchlist):**
   - Switch seamlessly between monitored crypto assets (**Bitcoin (BTC)**, **Ethereum (ETH)**, **Dogecoin (DOGE)**) and stock index ETFs (**S&P 500 (SPY)**, **NASDAQ 100 (QQQ)**).
   - View real-time price updates, 24-hour swings, sparklines, and policy stance badges (`Buy Opportunity`, `Hold / Defensive`, `Sell / Step Aside`).

2. **Center Pane (Strategy Lab & Analytical Workbench):**
   - Toggle between **🌱 Simple Mode** and **🔬 Advanced (Quant) Mode** via the segmented pill toolbar:
     - **🌱 Simple Mode (Default):** Built for retail clarity using accessible, question-led decision framing:
       - **Card 1 (What is the AI Doing Right Now?):** Clarifies live policy stance and recommended portfolio allocation (e.g. *35% Asset • 65% Cash* during pullbacks) dynamically harmonized with detected market climate.
       - **Card 2 (Why Did the AI Make This Call?):** Highlights the top **Key Decision Drivers** (live SHAP feature attributions filtered for meaningful impact, such as *Buyers Stepping In at Lows* and *20-Period Trend*).
       - **Card 3 (How Does It Handle Market Stress?):** Interactive market stress evaluator assessing account defense across historical market case studies (2022 FTX Liquidity Collapse, 2024 ETF Inflow Expansion, High-Volatility Range), with simulated ruin probability ($0.0\%$) and profitable simulation rate.
       - **Expandable Projection Drawer:** Displays the historical simulated portfolio net worth curve and 50-path Monte Carlo risk projection cones with real timestamped x-axes.
     - **🔬 Advanced (Quant) Mode:** Full institutional workbench for academic and quantitative evaluation:
       - **Section 1: Backtest Performance & Equity Curve:** Cumulative return, annualized Sharpe ratio, maximum drawdown, and chronological trade markers.
       - **Section 2: Market Climate (Unsupervised HMM):** 3-state Hidden Markov Model regime transition matrix, transition probabilities, and live volatility state.
       - **Section 3: Deflated Sharpe Ratio (DSR):** Illustrative DSR/PSR calculations; a formal empirical DSR needs the full exploratory trial history.
       - **Section 4: Local & Global SHAP Explainability:** Neural network feature attribution rankings with relative percentage impacts across all 14 state inputs.
       - **Section 5: 5-Fold Walk-Forward Validation (WFV):** Out-of-sample stress matrix across non-overlapping historical market eras (2020 Bull Run, 2021 ATH, 2022 Crypto Winter, 2023 Consolidation).
       - **Section 6: Monte Carlo Equity Projection Cones:** 50-path stochastic parameter simulation displaying median trajectories, 5th-95th percentile confidence bands, and Value-at-Risk ($VaR_{5\%}$).

3. **Right Pane (AI Financial Advisor Copilot):**
   - Multi-turn interactive chat interface powered by Google Gemini (with deterministic offline quantitative fallbacks).
   - Grounded in real-time model telemetry: policy action probabilities, top SHAP decision drivers, detected market climate, and backtest metrics.
   - Includes one-click prompt pills for instant queries (*"🎯 Why Buy Opportunity?"*, *"🛡️ Stress test scenario"*, *"📊 Explain risk grade"*).

---

## 🏗️ Project Architecture

```text
cm3070-rl-trading-system/
├── api.py                   # FastAPI quantitative backend (PPO inference, backtesting, XAI)
├── verify_submission.py     # Automated 7-stage pre-submission verification audit
├── Verify_Submission.bat    # Windows one-click runner for submission verification audit
├── Launch_Dashboard.bat     # Windows launcher for FastAPI backend & React frontend
├── Launch_Train_GUI.bat     # Windows launcher for model retraining GUI
├── Run_Tests.bat            # Windows launcher for running pytests
├── FINAL_PROJECT_REPORT.pdf # updated academic dissertation PDF (Template 4.2 compliant)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation and marker guide
├── src/                     # Shared source code for core logic
│   ├── rl_env.py            # Custom vectorized Gymnasium environment & Transformers
│   ├── train_agent.py       # Core PPO training loop with dynamic scheduling
│   ├── mc_engine.py         # Monte Carlo simulation engine
│   ├── xai_engine.py        # Explanation generation & polarity consistency validator
│   ├── xai_shap.py          # Local and global SHAP feature attribution
│   ├── walk_forward_eval.py # Walk-Forward Validation & Deflated Sharpe Ratio engine
│   ├── trading_utils.py     # Mathematical backtesting models and statistics
│   └── utils/dsr.py         # Bailey & López de Prado (2014) DSR/PSR formulations
├── frontend/                # Vite React TypeScript dashboard user interface
├── data/                    # Historical OHLCV datasets (Crypto, ETFs, Commodities)
├── models/                  # Saved policy checkpoints (.pth, .zip, .pkl normalizers)
├── docs/                    # Academic reports, rubric compliance matrix, and slides
│   ├── RUBRIC_COMPLIANCE_MATRIX.md # Detailed LO1-LO6 & Template 4.2 audit mapping
│   ├── DRAFT_FINAL_PROJECT_REPORT_V2.tex # LaTeX report source
│   └── HEURISTIC_EVALUATION_REPORT.md # Nielsen's 10-heuristic usability audit
└── tests/                   # Automated test suite
```

### System Components
| Component | Technology | Purpose |
|---|---|---|
| **RL Optimization Engine** | Stable Baselines3 (PPO) | Optimizes policy gradients for realized PnL |
| **Temporal Feature Extractor** | PyTorch (Transformers) | Processes frame-stacked momentum data |
| **Market Simulation** | Gymnasium | Vectorized backtest environment with fees and cooldowns |
| **Walk-Forward Engine** | Custom WFV + SciPy | Sequential out-of-sample testing & Deflated Sharpe Ratios |
| **Web Dashboard** | Vite + React + TypeScript | Sleek user-facing dashboard for backtests, XAI, and chatbot |
| **Backend API** | FastAPI + Uvicorn | Performs model inference, backtests, and SHAP attributions |
| **Explainable AI (XAI)** | SHAP + Local persona analyzer | Visualizes and describes AI decision attributions |
| **Live Data Pipeline** | Binance & Yahoo Finance APIs + Local CSVs | Real-time crypto (BTC, ETH, DOGE) and ETF (SPY, QQQ) live price feeds with TTL caching alongside historical OHLCV data |

---

## 🔬 Quantitative Rigor & Anti-Overfitting Design

### 1. Elimination of Look-Ahead Bias
In [`ActiveCryptoEnv`](src/rl_env.py), actions at step $T$ are executed against $\text{Open}_{T+1}$ with 0.1% commission and 0.05% slippage, reducing same-candle look-ahead risk.

### 2. Failure Mode & Policy Collapse Prevention
During initial baseline iterations:
* **Naive Profit Optimization:** Agents over-traded, accumulating negative returns due to transaction fee drag.
* **Current design:** Uses clipped log-return with drawdown and safeguard penalties plus action cooldowns. Differential Sharpe was an earlier experiment, not the current step reward.

### 3. Out-of-Sample Walk-Forward Validation
The strategy was evaluated with five sequential **Walk-Forward Validation** folds ([`src/walk_forward_eval.py`](src/walk_forward_eval.py)). This probes different historical regimes but does not eliminate model-selection bias or single-seed uncertainty.

---

## 📝 License

This project is submitted as coursework for the CM3070 Computer Science Final Project at the University of London.

