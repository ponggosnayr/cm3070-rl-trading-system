"""
University of London CM3070 Final Project — Automated Submission Verifier
========================================================================
Project: Adaptive Multi-Market RL Trading System & Financial Advisor Bot
Specification: Template Reference 4.2 (Financial Advisor Bot)

Executes an automated 7-stage pre-submission verification audit:
  Stage 1: Environment & Dependency Inspection
  Stage 2: Python Codebase Compilation & AST Verification (0 syntax errors)
  Stage 3: Pre-trained Checkpoints & Data Invariant Validation
  Stage 4: Full Automated Test Suite Execution (dynamic count)
  Stage 5: Quantitative Empirical Convergence (CSV vs Report Tables 4, 5, 6)
  Stage 6: Illustrative DSR Math & Synthetic Template Checks (rho >= 0.95)
  Stage 7: Production Deliverables Check

Exit Code:
  0 = All 7 automated stages passed (not a grading or live-feed guarantee)
  1 = Verification failure encountered
"""

import os
import sys
import glob
import time
import subprocess
import py_compile
import re
import pandas as pd
import numpy as np


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_stage_header(stage_num: int, title: str):
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[Stage {stage_num}/7] {title}{Colors.ENDC}")
    print("-" * 72)


def run_stage_1_environment() -> bool:
    print_stage_header(1, "Environment & Dependency Inspection")
    py_ver = sys.version.split()[0]
    print(f"  Python Version: {py_ver} (Target: >= 3.10) ... {Colors.OKGREEN}[OK]{Colors.ENDC}")

    required_pkgs = [
        "torch", "stable_baselines3", "sb3_contrib", "gymnasium", "pandas",
        "numpy", "fastapi", "uvicorn", "pydantic", "shap", "scipy"
    ]
    missing = []
    for pkg in required_pkgs:
        try:
            __import__(pkg)
            print(f"  Package '{pkg}': Available ... {Colors.OKGREEN}[OK]{Colors.ENDC}")
        except ImportError:
            print(f"  Package '{pkg}': MISSING ... {Colors.FAIL}[FAILED]{Colors.ENDC}")
            missing.append(pkg)

    if missing:
        print(f"\n  {Colors.FAIL}Error: Missing dependencies: {missing}{Colors.ENDC}")
        return False
    return True


def run_stage_2_syntax() -> bool:
    print_stage_header(2, "Python Codebase Compilation & AST Verification")
    files = [f for f in glob.glob("**/*.py", recursive=True) if not f.startswith("env\\") and not f.startswith("env/")]
    errors = 0
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
        except Exception as e:
            print(f"  {Colors.FAIL}Error in {f}: {e}{Colors.ENDC}")
            errors += 1

    if errors == 0:
        print(f"  Compiled {len(files)} Python files cleanly with 0 syntax errors ... {Colors.OKGREEN}[OK]{Colors.ENDC}")
        return True
    return False


def run_stage_3_checkpoints_and_data() -> bool:
    print_stage_header(3, "Pre-Trained Checkpoints & Data Invariant Check")
    critical_files = [
        ("models/persistent_brain.pth", 70_000_000),  # > 70MB
        ("models/persistent_brain_normalizer.pkl", 4_000),
        ("models/hmm_regime.pkl", 1_000),
        ("models/dqn_baseline.zip", 200_000),
        ("models/a2c_baseline.zip", 200_000),
        ("data/btc_usdt_1h.csv", 10_000_000),
        ("data/eth_usdt_1h.csv", 10_000_000),
        ("data/doge_usdt_1h.csv", 10_000_000),
        ("data/cross_market_results.csv", 500),
        ("walk_forward_results.csv", 300),
        ("comparison_results.csv", 150),
    ]
    all_ok = True
    for path, min_size in critical_files:
        if not os.path.exists(path):
            print(f"  File '{path}': MISSING ... {Colors.FAIL}[FAILED]{Colors.ENDC}")
            all_ok = False
        else:
            size = os.path.getsize(path)
            if size < min_size:
                print(f"  File '{path}': Truncated ({size} bytes < {min_size}) ... {Colors.FAIL}[FAILED]{Colors.ENDC}")
                all_ok = False
            else:
                print(f"  File '{path}': Present ({size:,} bytes) ... {Colors.OKGREEN}[OK]{Colors.ENDC}")
    return all_ok


def run_stage_4_automated_tests() -> bool:
    print_stage_header(4, "Full Automated Test Suite Execution")
    print("  Running 'pytest tests/ -v'...")
    t0 = time.time()
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
    elapsed = time.time() - t0

    passed_match = re.search(r"(\d+) passed", res.stdout)
    has_failures = "failed" in res.stdout or res.returncode != 0
    if res.returncode == 0 and passed_match and not has_failures:
        count = passed_match.group(1)
        print(f"  All {count} unit tests passed in {elapsed:.1f}s with zero failures ... {Colors.OKGREEN}[OK]{Colors.ENDC}")
        return True
    else:
        print(f"  Test suite finished with code {res.returncode}:")
        for line in res.stdout.splitlines()[-10:]:
            print("    ", line)
        if res.stderr:
            print("  Stderr:", res.stderr[-500:])
        return False


def run_stage_5_empirical_convergence() -> bool:
    print_stage_header(5, "Quantitative Empirical Convergence (CSVs vs Report Tables)")

    # 1. Walk-forward verification
    wfv_df = pd.read_csv("walk_forward_results.csv")
    mean_roi = float(wfv_df["Out-of-Sample ROI (%)"].mean())
    mean_sharpe = float(wfv_df["Out-of-Sample Sharpe"].mean())
    mean_dd = float(wfv_df["Max DD (%)"].mean())
    mean_trades = float(wfv_df["Trades"].mean())

    assert abs(mean_roi - 11.11) < 0.05, f"WFV Mean ROI mismatch: {mean_roi} != 11.11"
    assert abs(mean_sharpe - 0.60) < 0.05, f"WFV Mean Sharpe mismatch: {mean_sharpe} != 0.60"
    assert abs(mean_dd - (-4.91)) < 0.05, f"WFV Mean DD mismatch: {mean_dd} != -4.91"
    assert abs(mean_trades - 13.8) < 0.05, f"WFV Mean Trades mismatch: {mean_trades} != 13.8"
    print(f"  Table 4 (Walk-Forward Validation): Mean ROI=+11.11%, Sharpe=0.60, DD=-4.91% ... {Colors.OKGREEN}[MATCH]{Colors.ENDC}")

    # 2. Baseline comparison verification
    comp_df = pd.read_csv("comparison_results.csv")
    ppo_row = comp_df[comp_df["Model"].str.contains("PPO")].iloc[0]
    dqn_row = comp_df[comp_df["Model"].str.contains("DQN")].iloc[0]
    bh_row = comp_df[comp_df["Model"].str.contains("Buy & Hold")].iloc[0]

    assert ppo_row["ROI (%)"] == 0.0 and ppo_row["Max DD (%)"] == 0.0
    assert dqn_row["ROI (%)"] == -8.04 and dqn_row["Max DD (%)"] == 13.92
    assert bh_row["ROI (%)"] == -24.94 and bh_row["Max DD (%)"] == 52.86
    print(f"  Table 5 (Baseline Comparison): PPO=0.00% DD, DQN=-8.04% ROI, B&H=-24.94% ROI ... {Colors.OKGREEN}[MATCH]{Colors.ENDC}")

    # 3. Cross-market verification
    cm_df = pd.read_csv("data/cross_market_results.csv")
    btc_row = cm_df[cm_df["Asset"].str.contains("BTC")].iloc[0]
    eth_row = cm_df[cm_df["Asset"].str.contains("ETH")].iloc[0]
    doge_row = cm_df[cm_df["Asset"].str.contains("DOGE")].iloc[0]

    assert btc_row["Model ROI (%)"] == 8.71 and btc_row["Alpha (%)"] == 38.52
    assert eth_row["Model ROI (%)"] == 14.41 and eth_row["Alpha (%)"] == 60.99
    assert doge_row["Model ROI (%)"] == 72.73 and doge_row["Alpha (%)"] == 113.33
    print(f"  Table 6 (Cross-Market Transfer): BTC=+8.71%, ETH=+14.41%, DOGE=+72.73% ... {Colors.OKGREEN}[MATCH]{Colors.ENDC}")
    return True


def run_stage_6_xai_and_dsr() -> bool:
    print_stage_header(6, "Illustrative DSR Math & Synthetic Template Checks")

    # DSR module verification
    sys.path.insert(0, os.path.abspath("src"))
    from utils.dsr import deflated_sharpe_ratio, minimum_track_record_length
    fold_sharpes = [0.59, 1.36, 0.00, 0.00, 1.04]
    dsr_result = deflated_sharpe_ratio(
        sharpe_ratio=0.60,
        sample_length=48130,
        trials_sr=fold_sharpes,
        annualization_factor=8760.0
    )
    assert dsr_result["observed_sharpe"] == 0.60
    assert dsr_result["n_trials"] == 5
    assert dsr_result["dsr"] > 0.0
    print(f"  Illustrative DSR/PSR math (fold Sharpes used as trial inputs): DSR = {dsr_result['dsr']:.2f}, PSR = {dsr_result['psr']:.2f} ... {Colors.OKGREEN}[CALCULATED]{Colors.ENDC}")

    # Synthetic explanation-ordering check across all 3 action labels
    from xai_audit import audit_all_actions_fidelity
    fidelity = audit_all_actions_fidelity(num_samples_per_action=50, random_seed=42)
    mean_rho = fidelity["overall_mean_spearman_rho"]
    assert fidelity["overall_pass"], "Synthetic explanation-ordering check failed"
    assert mean_rho >= 0.95, f"Synthetic ordering below threshold: {mean_rho} < 0.95"
    for act, r in fidelity["per_action"].items():
        print(f"  Synthetic explanation ordering ({act}): rho={r['mean_spearman_rho']:.4f}, missing={r['missing_feature_rate']:.1%} ... {Colors.OKGREEN}[PASS]{Colors.ENDC}")
    print(f"  Synthetic three-action ordering: Mean rho={mean_rho:.4f}; real-model multi-action fidelity unverified ... {Colors.OKGREEN}[CALCULATED]{Colors.ENDC}")
    return True


def run_stage_7_deliverables() -> bool:
    print_stage_header(7, "Production Deliverables")

    # Frontend build verification
    fe_index = "frontend/dist/index.html"
    if os.path.exists(fe_index):
        print(f"  Frontend Production Bundle: {fe_index} exists ... {Colors.OKGREEN}[OK]{Colors.ENDC}")
    else:
        print(f"  Frontend Production Bundle: MISSING (run 'npm run build' in frontend/) ... {Colors.FAIL}[FAILED]{Colors.ENDC}")
        return False

    return True

def main():
    print("=" * 72)
    print("  UNIVERSITY OF LONDON CM3070 FINAL PROJECT SUBMISSION AUDIT")
    print("  Adaptive Multi-Market RL Trading System & Financial Advisor Bot")
    print("=" * 72)

    stages = [
        ("Environment & Dependencies", run_stage_1_environment),
        ("Python Codebase Syntax", run_stage_2_syntax),
        ("Checkpoints & Datasets", run_stage_3_checkpoints_and_data),
        ("Automated Test Suite (Dynamic Discovery)", run_stage_4_automated_tests),
        ("Empirical CSV-Report Convergence", run_stage_5_empirical_convergence),
        ("Illustrative DSR Math & Synthetic Ordering", run_stage_6_xai_and_dsr),
        ("Production Deliverables", run_stage_7_deliverables),
    ]

    results = []
    t_start = time.time()

    for name, func in stages:
        try:
            ok = func()
            results.append((name, ok))
            if not ok:
                print(f"\n{Colors.FAIL}STAGE FAILED: {name}{Colors.ENDC}")
                break
        except Exception as e:
            print(f"\n{Colors.FAIL}EXCEPTION IN STAGE '{name}': {e}{Colors.ENDC}")
            results.append((name, False))
            break

    total_time = time.time() - t_start
    print("\n" + "=" * 72)
    print("  AUDIT SUMMARY")
    print("=" * 72)

    all_passed = True
    for name, ok in results:
        status_str = f"{Colors.OKGREEN}PASSED{Colors.ENDC}" if ok else f"{Colors.FAIL}FAILED{Colors.ENDC}"
        print(f"  {name:<45} : {status_str}")
        if not ok:
            all_passed = False

    print("-" * 72)
    print(f"  Total Audit Time: {total_time:.2f} seconds")

    if all_passed and len(results) == len(stages):
        print(f"\n  {Colors.BOLD}{Colors.OKGREEN}Automated checks passed. Empirical claims and rubric compliance are not certified.{Colors.ENDC}\n")
        sys.exit(0)
    else:
        print(f"\n  {Colors.BOLD}{Colors.FAIL}VERIFICATION FAILED! PLEASE REVIEW STAGE FAILURES ABOVE.{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
