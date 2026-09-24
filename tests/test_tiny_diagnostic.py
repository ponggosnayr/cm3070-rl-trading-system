import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
import run_tiny_diagnostic as d


def frame(values):
    return d.synthetic_frame(values)


def test_rising_long_has_positive_return_and_accounting(tmp_path):
    result = d.evaluate_policy(d.FixedPolicy(1), frame(np.linspace(100, 120, 20)), label="rise",
                               trace_path=tmp_path / "rise.csv")
    assert result["horizon"] == "full"
    assert result["roi_pct"] > 0
    assert result["policy_counts"]["1"] == result["steps"]
    assert len(pd.read_csv(tmp_path / "rise.csv")) == result["steps"] + 1


def test_falling_long_loses_short_gains_and_cash_is_flat():
    falling = d.evaluate_policy(d.FixedPolicy(1), frame(np.linspace(120, 100, 20)), label="fall")
    cash = d.evaluate_policy(d.FixedPolicy(0), frame(np.linspace(120, 100, 20)), label="cash")
    assert falling["roi_pct"] < 0
    short = d.evaluate_policy(d.FixedPolicy(2), frame(np.linspace(120, 100, 20)), label="short")
    assert short["roi_pct"] > 0
    assert cash["roi_pct"] == 0
    assert cash["trades"] == 0


def test_flat_long_costs_are_reflected():
    result = d.evaluate_policy(d.FixedPolicy(1), frame(np.full(20, 100.0)), label="flat")
    assert result["roi_pct"] < 0
    assert result["buy_hold_roi_pct"] == 0
    expected = ((1 - .001) ** 2 * (1 - .0005) / (1 + .0005) - 1) * 100
    assert abs(result['roi_pct'] - expected) < .0001
    assert abs(result['buy_hold_net_costs_roi_pct'] - expected) < .0001


def test_smoke_main_writes_manifest(tmp_path):
    d.main(["--smoke", "--out", str(tmp_path)])
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    assert '"mode": "smoke"' in manifests[0].read_text()
