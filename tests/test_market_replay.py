"""Regression check that fallback candles remain tied to archived market data."""

import pandas as pd
import pytest

from api import generate_5m_candles_from_1h


def test_offline_replay_keeps_source_dates_and_prices(tmp_path):
    hours = pd.date_range("2026-09-07 00:00:00", periods=24, freq="h", tz="UTC")
    source = pd.DataFrame({
        "datetime": hours.tz_localize(None).astype(str),
        "open": [100.0 + i for i in range(24)],
        "high": [102.0 + i for i in range(24)],
        "low": [99.0 + i for i in range(24)],
        "close": [101.0 + i for i in range(24)],
    })
    path = tmp_path / "archived_hourly.csv"
    source.to_csv(path, index=False)

    candles = generate_5m_candles_from_1h(str(path))

    assert len(candles) == 24 * 12
    assert candles[0]["time"] == "2026-09-07 00:00 UTC"
    assert candles[-1]["time"] == "2026-09-07 23:55 UTC"
    assert candles[-1]["close"] == pytest.approx(124.0, abs=0.01)
    assert candles == generate_5m_candles_from_1h(str(path))
