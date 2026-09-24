# CM3070 Trading System

Source code for a local research dashboard that displays saved reinforcement learning trading signals, historical backtests, risk simulations and explanations. It does not place real trades.

## Requirements

- Python 3.12
- Node.js 20.19+ or 22.12+, with npm

## Install

From a terminal:

    git clone https://github.com/ponggosnayr/cm3070-rl-trading-system.git
    cd cm3070-rl-trading-system
    python -m venv env

Activate the environment (env\Scripts\activate on Windows, or source env/bin/activate on macOS/Linux), then install the backend and frontend dependencies:

    python -m pip install -r requirements.txt
    cd frontend
    npm ci

The repository includes the saved model and sample market data needed for the local demo. A Gemini API key is optional; the advisor has a deterministic fallback when it is unavailable.

## Run the app

Open two terminals from the repository root. With the Python environment active, start the backend in one:

    python api.py

Start the frontend in the other:

    cd frontend
    npm run dev

Open http://localhost:5173. On Windows, Launch_Dashboard.bat is an alternative after installation.

## Check the code

Run the Python tests from the repository root:

    python -m pytest tests/

Check the frontend from the frontend directory:

    npm run build

## Where to look

- api.py: FastAPI backend and model inference
- frontend/: React dashboard
- src/: trading environment, training, backtesting and explanations
- models/: saved checkpoints and normalizers
- data/: historical market data
- tests/: automated tests

- FINAL_PROJECT_REPORT.pdf: project dissertation report

Results shown by the dashboard are research simulations, not investment advice.
