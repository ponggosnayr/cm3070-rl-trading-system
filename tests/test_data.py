import pytest
import pandas as pd
import numpy as np
import os
import sys
from typing import List

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)

EXPECTED_FILES = [
    'btc_usdt_1h.csv',
    'eth_usdt_1h.csv',
    'doge_usdt_1h.csv',
]

REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
INDICATOR_COLUMNS = [
    'sma_7', 'sma_25', 'sma_99', 
    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
    'volume_sma_20', 'roc_12', 'market_regime'
]

@pytest.fixture(scope="session")
def btc_df() -> pd.DataFrame:
    """Load the BTC dataset for testing integrity."""
    filepath = os.path.join(DATA_DIR, 'btc_usdt_1h.csv')
    if not os.path.exists(filepath):
        pytest.skip(f"BTC data file not found at {filepath}")
    return pd.read_csv(filepath)

class TestDataAvailability:
    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_exists(self, filename: str) -> None:
        """Verify that essential data files are present in the data directory."""
        filepath = os.path.join(DATA_DIR, filename)
        assert os.path.exists(filepath), f"Required data file {filename} is missing."

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_content(self, filename: str) -> None:
        """Verify that data files are not empty and have a valid header."""
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip("File missing")
        df = pd.read_csv(filepath, nrows=5)
        assert not df.empty, f"Data file {filename} is empty."
        for col in REQUIRED_COLUMNS + INDICATOR_COLUMNS:
            assert col in df.columns, f"Column {col} missing in {filename}."

class TestDataIntegrity:
    def test_column_types(self, btc_df: pd.DataFrame) -> None:
        """Ensure OHLCV columns are numeric."""
        for col in REQUIRED_COLUMNS:
            assert pd.api.types.is_numeric_dtype(btc_df[col]), f"Column {col} is not numeric."

    def test_price_consistency(self, btc_df: pd.DataFrame) -> None:
        """Validate basic price logic: high >= low, high >= open, high >= close."""
        assert (btc_df['high'] >= btc_df['low']).all(), "High price found below low price."
        assert (btc_df['high'] >= btc_df['open']).all(), "High price found below open price."
        assert (btc_df['high'] >= btc_df['close']).all(), "High price found below close price."
        assert (btc_df['low'] <= btc_df['open']).all(), "Low price found above open price."
        assert (btc_df['low'] <= btc_df['close']).all(), "Low price found above close price."

    def test_no_nulls_in_critical_columns(self, btc_df: pd.DataFrame) -> None:
        """Critical columns must not contain any missing values."""
        for col in REQUIRED_COLUMNS:
            assert btc_df[col].isna().sum() == 0, f"NaN values detected in critical column {col}."
        
        # Indicators might have NaNs at the beginning due to rolling windows,
        # so we check after the first 100 rows.
        for col in INDICATOR_COLUMNS:
            assert btc_df[col].iloc[100:].isna().sum() == 0, f"NaN values detected in indicator column {col}."

    def test_rsi_range(self, btc_df: pd.DataFrame) -> None:
        """RSI indicator should be within the standard 0-100 range."""
        if 'rsi_14' in btc_df.columns:
            rsi = btc_df['rsi_14'].dropna()
            assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI values outside [0, 100]."

    def test_temporal_consistency(self, btc_df: pd.DataFrame) -> None:
        """Ensure no duplicate timestamps if a time column exists."""
        time_cols = [c for c in btc_df.columns if 'time' in c.lower() or 'date' in c.lower()]
        for col in time_cols:
            valid_times = btc_df[col].dropna()
            assert valid_times.duplicated().sum() == 0, f"Duplicate timestamps found in {col}."


class TestCrossMarketConsistency:
    def test_schema_uniformity(self) -> None:
        """All market datasets should have identical column schemas."""
        schemas = []
        for filename in EXPECTED_FILES:
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                schemas.append(set(pd.read_csv(path, nrows=0).columns))
        
        if len(schemas) > 1:
            first_schema = schemas[0]
            for i, schema in enumerate(schemas[1:], 1):
                assert schema == first_schema, f"Schema mismatch for {EXPECTED_FILES[i]}."
