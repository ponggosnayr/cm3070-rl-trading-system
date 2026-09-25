"""Guard the ETF daily datasets used by the dashboard."""

from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("asset", ["spy", "qqq"])
def test_etf_daily_data_has_one_candle_per_market_date(asset):
    data = pd.read_csv(DATA_DIR / f"{asset}_daily.csv")
    dates = pd.to_datetime(data["datetime"], errors="raise")

    assert dates.is_monotonic_increasing
    assert dates.dt.normalize().is_unique
    assert (dates.dt.strftime("%H:%M:%S") == "00:00:00").all()
    unix_seconds = (dates - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    assert (unix_seconds == data["timestamp"]).all()

    indicator_columns = {
        "sma_7", "sma_25", "sma_99", "rsi_14", "macd", "macd_signal",
        "macd_hist", "volume_sma_20", "roc_12", "market_regime",
    }
    assert indicator_columns.issubset(data.columns)
