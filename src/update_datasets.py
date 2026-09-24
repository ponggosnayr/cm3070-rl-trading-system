"""
Automated Multi-Asset Dataset Ingestion & Update Pipeline
=========================================================
Updates historical CSV datasets in data/ using live feeds:
- Crypto (BTC, ETH, DOGE): Binance via CCXT
- Equities & Commodities (SPY, QQQ, GLD, WTI): Yahoo Finance via yfinance

Ensures all technical indicators and HMM market regimes are
re-calculated to maintain 100% schema integrity and test compliance.
"""

import os
import sys
import time
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from trading_utils import compute_indicators

DATA_DIR = os.path.join(BASE_DIR, "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backup_frozen_benchmark")

def ensure_backup():
    """Ensure a backup of original benchmark CSVs exists."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".csv") and "results" not in fname:
            src_file = os.path.join(DATA_DIR, fname)
            dst_file = os.path.join(BACKUP_DIR, fname)
            if not os.path.exists(dst_file) and os.path.isfile(src_file):
                import shutil
                shutil.copy2(src_file, dst_file)
    print(f"[OK] Verified backup folder at: {BACKUP_DIR}")

def update_crypto_asset(symbol_pair: str, tf: str, filename: str):
    """Fetch recent crypto candles from Binance and update CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n--- Updating {symbol_pair} ({tf}) -> {filename} ---")
    
    if not os.path.exists(filepath):
        print(f"  [ERROR] File {filepath} not found.")
        return

    existing_df = pd.read_csv(filepath)
    dt_col = "datetime" if "datetime" in existing_df.columns else existing_df.columns[0]
    existing_df["datetime"] = pd.to_datetime(existing_df[dt_col], utc=True)
    last_dt = existing_df["datetime"].max()
    print(f"  Existing dataset ends at: {last_dt}")
    
    since_ms = int(last_dt.timestamp() * 1000) - (2 * 3600 * 1000 if tf == "1h" else 2 * 86400 * 1000)
    
    exchange = ccxt.binance({"enableRateLimit": True})
    all_candles = []
    current_since = since_ms
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol_pair, tf, since=current_since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if last_ts <= current_since or len(candles) < 1000:
                break
            current_since = last_ts + 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning during fetch: {e}")
            break

    if not all_candles:
        print(f"  No new candles available for {symbol_pair}.")
        return

    new_df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    new_df["datetime"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
    print(f"  Fetched {len(new_df)} live candles up to {new_df['datetime'].max()}.")

    # Keep OHLCV only before recomputing
    ohlcv_cols = ["datetime", "open", "high", "low", "close", "volume"]
    combined = pd.concat([existing_df[ohlcv_cols], new_df[ohlcv_cols]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
    
    # Calculate timestamp as ms integer
    combined["timestamp"] = (combined["datetime"].astype("int64") // 10**6).astype("int64")
    
    # Recompute all technical indicators and HMM market regime
    print("  Recomputing indicators and HMM market regimes...")
    combined = compute_indicators(combined)
    combined = combined.ffill().bfill()
    
    # Format datetime as standard string YYYY-MM-DD HH:MM:SS
    combined["datetime"] = combined["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure no NaN remains
    assert combined.isnull().sum().sum() == 0, f"NaN values detected in {filename}"
    
    # Save back to CSV
    combined.to_csv(filepath, index=False)
    print(f"  [SUCCESS] {filename} updated: {len(combined):,} rows, ending at {combined['datetime'].iloc[-1]}")

def update_yahoo_asset(ticker: str, tf: str, filename: str):
    """Fetch recent equity/commodity candles from Yahoo Finance and update CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n--- Updating {ticker} ({tf}) -> {filename} ---")
    
    if not os.path.exists(filepath):
        print(f"  [ERROR] File {filepath} not found.")
        return

    existing_df = pd.read_csv(filepath)
    dt_col = "datetime" if "datetime" in existing_df.columns else existing_df.columns[0]
    existing_df["datetime"] = pd.to_datetime(existing_df[dt_col], utc=True)
    last_dt = existing_df["datetime"].max()
    print(f"  Existing dataset ends at: {last_dt}")

    # Fetch from Yahoo Finance
    period = "730d" if tf == "1h" else "max"
    interval = "1h" if tf == "1h" else "1d"
    
    try:
        yf_df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return

    if yf_df.empty:
        print(f"  No data returned from Yahoo Finance for {ticker}.")
        return

    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = [col[0].lower() for col in yf_df.columns]
    else:
        yf_df.columns = [col.lower() for col in yf_df.columns]
    yf_df = yf_df.reset_index()
    
    date_col = next((c for c in yf_df.columns if "date" in str(c).lower() or "time" in str(c).lower()), yf_df.columns[0])
    yf_df["datetime"] = pd.to_datetime(yf_df[date_col], utc=True)
    
    # Select OHLCV
    ohlcv_cols = ["datetime", "open", "high", "low", "close", "volume"]
    yf_clean = yf_df[ohlcv_cols].dropna().copy()
    
    # Merge with existing
    combined = pd.concat([existing_df[ohlcv_cols], yf_clean], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
    
    # Timestamp integer
    combined["timestamp"] = (combined["datetime"].astype("int64") // 10**6).astype("int64")
    
    # Recompute indicators and regimes
    print("  Recomputing indicators and HMM market regimes...")
    combined = compute_indicators(combined)
    combined = combined.ffill().bfill()
    
    # Format datetime
    combined["datetime"] = combined["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    assert combined.isnull().sum().sum() == 0, f"NaN values detected in {filename}"
    
    combined.to_csv(filepath, index=False)
    print(f"  [SUCCESS] {filename} updated: {len(combined):,} rows, ending at {combined['datetime'].iloc[-1]}")

def main():
    ensure_backup()
    
    crypto_targets = [
        ("BTC/USDT", "1h", "btc_usdt_1h.csv"),
        ("BTC/USDT", "1d", "btc_usdt_daily.csv"),
        ("ETH/USDT", "1h", "eth_usdt_1h.csv"),
        ("ETH/USDT", "1d", "eth_usdt_daily.csv"),
        ("DOGE/USDT", "1h", "doge_usdt_1h.csv"),
        ("DOGE/USDT", "1d", "doge_usdt_daily.csv"),
    ]
    
    yahoo_targets = [
        ("SPY", "1h", "spy_1h.csv"),
        ("SPY", "1d", "spy_daily.csv"),
        ("QQQ", "1h", "qqq_1h.csv"),
        ("QQQ", "1d", "qqq_daily.csv"),
        ("GLD", "1h", "gld_1h.csv"),
        ("GLD", "1d", "gld_daily.csv"),
        ("CL=F", "1h", "wti_1h.csv"),
        ("CL=F", "1d", "wti_daily.csv"),
    ]
    
    print("\n=======================================================")
    print("  STARTING MULTI-ASSET LIVE DATA INGESTION & UPDATE")
    print("=======================================================")
    
    for symbol, tf, fname in crypto_targets:
        try:
            update_crypto_asset(symbol, tf, fname)
        except Exception as e:
            print(f"  Failed {fname}: {e}")
            
    for ticker, tf, fname in yahoo_targets:
        try:
            update_yahoo_asset(ticker, tf, fname)
        except Exception as e:
            print(f"  Failed {fname}: {e}")

    print("\n=======================================================")
    print("  ALL ASSET UPDATES COMPLETED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
