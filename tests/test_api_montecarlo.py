import sys
import os
import pytest
from fastapi import HTTPException

# Add project root directory to path so we can import api.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import get_montecarlo_data

def test_api_montecarlo_success():
    """Verify that the Monte Carlo simulation logic returns correct structure and values."""
    # We call the function directly as a standard Python function
    data = get_montecarlo_data(asset="Bitcoin (BTC)", timeframe="1H", simulations=5)
    
    # Check high-level fields
    assert "summary" in data
    assert "percentiles" in data
    assert "sample_paths" in data
    assert "step_count" in data
    assert data["step_count"] == 150
    
    # Check summary metrics keys
    summary = data["summary"]
    expected_summary_keys = [
        "mean_roi", "std_roi", "min_roi", "max_roi", 
        "var_5th", "mean_sharpe", "mean_drawdown", "success_rate"
    ]
    for key in expected_summary_keys:
        assert key in summary
        assert isinstance(summary[key], float)
        
    # Check percentiles lists
    percentiles = data["percentiles"]
    for p_key in ["5", "25", "50", "75", "95"]:
        assert p_key in percentiles
        assert len(percentiles[p_key]) == 150
        for val in percentiles[p_key]:
            assert isinstance(val, (int, float))
            
    # Check sample paths
    sample_paths = data["sample_paths"]
    assert len(sample_paths) == 5
    for path in sample_paths:
        assert len(path) == 150
        for val in path:
            assert isinstance(val, (int, float))

def test_api_montecarlo_invalid_asset():
    """Verify that a 404 HTTPException is raised for a non-existent asset."""
    with pytest.raises(HTTPException) as exc_info:
        get_montecarlo_data(asset="FakeAssetCoin", timeframe="1H", simulations=5)
    assert exc_info.value.status_code == 404
    assert "Asset data not found" in exc_info.value.detail

def test_api_montecarlo_custom_parameters():
    """Verify that custom noise and black swan parameters run successfully."""
    data = get_montecarlo_data(
        asset="Bitcoin (BTC)", timeframe="1H", simulations=10, noise_std=0.005, inject_swan=True
    )
    assert data["step_count"] == 150
    assert len(data["sample_paths"]) == 5


@pytest.mark.parametrize("simulations", [0, -1, 501, 1_000_000])
def test_api_montecarlo_rejects_unbounded_work(simulations):
    with pytest.raises(HTTPException) as exc_info:
        get_montecarlo_data(asset="Bitcoin (BTC)", simulations=simulations)
    assert exc_info.value.status_code == 422


@pytest.mark.parametrize("noise_std", [-0.1, 0.21, float("nan")])
def test_api_montecarlo_rejects_invalid_noise(noise_std):
    with pytest.raises(HTTPException) as exc_info:
        get_montecarlo_data(asset="Bitcoin (BTC)", simulations=3, noise_std=noise_std)
    assert exc_info.value.status_code == 422
