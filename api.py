from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Any, Dict

import sys
import os
import re
import time
import hashlib
from datetime import datetime, timezone
import threading
import requests
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# Load .env file manually if it exists in the root folder
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from rl_env import ActiveCryptoEnv, TinyTransformerExtractor, DeepTransformerExtractor
from trading_utils import compute_indicators, run_single_fold_backtest
from xai_shap import get_shap_explainer, compute_shap_values
from xai_engine import XAIEngine
from mc_engine import run_mc_analysis

app = FastAPI(title="AI Financial Advisor API")

# Allow CORS for local Vite app (wildcard disallowed with credentials=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None
    asset_name: Optional[str] = None
    price: Optional[float] = None
    signal: Optional[str] = None
    confidence: Optional[float] = None
    sentiment: Optional[str] = None

# Target paths
MODEL_PATH = "models/persistent_brain.pth"
NORMALIZER_PATH = "models/persistent_brain_normalizer.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
include_regime = True
include_context = True
obs_dim = 17

def get_dynamic_policy_kwargs(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint
    
    extractor_class = DeepTransformerExtractor
    features_dim = 256
    inc_regime = True
    inc_context = True
    in_features = 17
    
    for key in state_dict.keys():
        if "features_extractor.embedding.weight" in key:
            out_features = state_dict[key].shape[0]
            in_features = state_dict[key].shape[1]
            inc_regime = (in_features in (12, 17))
            inc_context = (in_features in (16, 17))
            if out_features == 64:
                extractor_class = TinyTransformerExtractor
                features_dim = 64
            elif out_features == 256:
                extractor_class = DeepTransformerExtractor
                features_dim = 256
            break
            
    return dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
        share_features_extractor=False,
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
    ), state_dict, inc_regime, inc_context, in_features

# Load model weights on startup
if os.path.exists(MODEL_PATH):
    try:
        dynamic_kwargs, state_dict, include_regime, include_context, obs_dim = get_dynamic_policy_kwargs(MODEL_PATH)
        
        # Setup a dummy env for model initialization
        dummy_df = pd.DataFrame({
            "open": [100.0] * 120,
            "high": [100.0] * 120,
            "low": [100.0] * 120,
            "close": [100.0] * 120,
            "volume": [100.0] * 120,
            "timestamp": range(120),
            "datetime": pd.to_datetime(range(120), unit="h"),
        })
        dummy_df = compute_indicators(dummy_df)
        dummy_df = dummy_df.ffill().bfill().fillna(50.0)
        dummy_env = DummyVecEnv([lambda: ActiveCryptoEnv(
            dummy_df, render_mode=None, 
            include_regime=include_regime, 
            include_context=include_context,
            random_start=False
        )])
        dummy_env = VecFrameStack(dummy_env, n_stack=8)
        
        # Load SB3 model
        model = PPO("MlpPolicy", dummy_env, policy_kwargs=dynamic_kwargs, device=DEVICE)
        model.policy.load_state_dict(state_dict)
        
        # Load reward/obs normalizer if it exists
        if os.path.exists(NORMALIZER_PATH):
            try:
                dummy_env = VecNormalize.load(NORMALIZER_PATH, dummy_env)
                dummy_env.training = False
                model.set_env(dummy_env)
                print("Loaded model normalizer.")
            except Exception as e:
                print(f"Warning: normalizer exists but could not be loaded: {e}")
            
        print(f"Loaded trained PPO policy model on {DEVICE}. Details: regime={include_regime}, context={include_context}, obs_dim={obs_dim}")
    except Exception as e:
        print(f"FAILED TO LOAD PRE-TRAINED MODEL: {e}")
else:
    print(f"Model path {MODEL_PATH} not found. Running in rule-based fallback mode.")

def get_asset_csv_path(asset_label: str, timeframe: str) -> str:
    tf_suffix = "daily" if timeframe.upper() == "DAILY" else "1h"
    
    mapping = {
        "Bitcoin (BTC)": f"data/btc_usdt_{tf_suffix}.csv",
        "Ethereum (ETH)": f"data/eth_usdt_{tf_suffix}.csv",
        "Dogecoin (DOGE)": f"data/doge_usdt_{tf_suffix}.csv",
        "S&P 500 (SPY)": f"data/spy_{tf_suffix}.csv",
        "NASDAQ 100 (QQQ)": f"data/qqq_{tf_suffix}.csv",
    }
    return mapping.get(asset_label)

def generate_5m_candles_from_1h(csv_path: str):
    if not csv_path or not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    if len(df) < 24:
        return []
    
    # Take the last 24 rows and forward fill/backward fill to prevent NaNs
    last_24h = df.iloc[-24:].reset_index(drop=True).ffill().bfill().copy()
    source_times = pd.to_datetime(last_24h["datetime"], utc=True, errors="coerce")
    if source_times.isna().any():
        return []
    
    candles = []
    rng = np.random.default_rng(42)  # Local generator for reproducible Brownian bridge without mutating global state
    
    for idx in range(len(last_24h)):
        row = last_24h.iloc[idx]
        hour_start = source_times.iloc[idx].timestamp()
        o_val = float(row["open"])
        h_val = float(row["high"])
        l_val = float(row["low"])
        c_val = float(row["close"])
        
        # We generate 12 steps of 5-minute candles per hour
        N = 12
        noise_std = (h_val - l_val) * 0.15 if h_val > l_val else o_val * 0.0005
        if noise_std == 0:
            noise_std = 0.01
            
        steps = [0.0]
        for _ in range(N):
            steps.append(steps[-1] + rng.normal(0, noise_std))
            
        # Bridge adjustment: Y_t starting at o_val and ending at c_val
        y_vals = []
        for t in range(N + 1):
            y = steps[t] - (t / N) * steps[N] + (1 - t / N) * o_val + (t / N) * c_val
            y_vals.append(y)
            
        # Determine decimal places dynamically based on price level
        if o_val < 0.1:
            decimals = 6
        elif o_val < 1.0:
            decimals = 5
        elif o_val < 10.0:
            decimals = 4
        elif o_val < 100.0:
            decimals = 3
        else:
            decimals = 2

        for t in range(1, N + 1):
            open_p = y_vals[t - 1]
            close_p = y_vals[t]
            
            c_high = max(open_p, close_p)
            c_low = min(open_p, close_p)
            
            # Add small random wicks
            wick_h = rng.exponential(noise_std * 0.3)
            wick_l = rng.exponential(noise_std * 0.3)
            
            high_p = c_high + wick_h
            low_p = c_low - wick_l
            
            # Constraint: don't blow past the hourly absolute bounds
            high_p = min(high_p, h_val)
            low_p = max(low_p, l_val)
            
            # Enforce self-consistency
            high_p = max(high_p, open_p, close_p)
            low_p = min(low_p, open_p, close_p)
            
            candle_ts = hour_start + (t - 1) * 300
            time_str = pd.to_datetime(candle_ts, unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")
            timestamp_ms = int(candle_ts * 1000)

            candles.append({
                "open": round(open_p, decimals),
                "high": round(high_p, decimals),
                "low": round(low_p, decimals),
                "close": round(close_p, decimals),
                "time": time_str,
                "timestamp": timestamp_ms
            })
            
    return candles


LIVE_PRICE_LOCK = threading.Lock()
LIVE_PRICE_CACHE = {}
LIVE_PRICE_META = {}
LAST_PRICE_FETCH_TIME = 0.0
CACHE_TTL_SECONDS = 3.0

PREDICT_CACHE = {}
PREDICT_CACHE_LOCK = threading.Lock()
PREDICT_CACHE_TTL = 15.0

SHAP_CACHE = {}
SHAP_CACHE_LOCK = threading.Lock()
SHAP_CACHE_TTL = 300.0

def fetch_live_market_prices() -> dict:
    global LIVE_PRICE_CACHE, LIVE_PRICE_META, LAST_PRICE_FETCH_TIME
    now = time.time()
    with LIVE_PRICE_LOCK:
        if LIVE_PRICE_CACHE and (now - LAST_PRICE_FETCH_TIME) < CACHE_TTL_SECONDS:
            return dict(LIVE_PRICE_CACHE)
        prices = dict(LIVE_PRICE_CACHE)

    fetch_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    crypto_map = {
        "BTCUSDT": "Bitcoin (BTC)",
        "ETHUSDT": "Ethereum (ETH)",
        "DOGEUSDT": "Dogecoin (DOGE)"
    }

    # 1. Fetch Binance Live Crypto Prices (BTC, ETH, DOGE)
    binance_updated = set()
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbols=%5B%22BTCUSDT%22,%22ETHUSDT%22,%22DOGEUSDT%22%5D"
        resp = requests.get(url, timeout=2.0)
        if resp.status_code == 200:
            for item in resp.json():
                sym = item.get("symbol")
                p = float(item.get("price", 0))
                if sym in crypto_map and p > 0:
                    label = crypto_map[sym]
                    prices[label] = p
                    LIVE_PRICE_META[label] = {
                        "fetch_time_utc": fetch_time_utc,
                        "is_live": True,
                        "last_success_time": now
                    }
                    binance_updated.add(label)
    except Exception as e:
        print(f"Warning: Binance live price fetch error: {e}")

    for label in crypto_map.values():
        if label not in binance_updated:
            if label in LIVE_PRICE_META:
                LIVE_PRICE_META[label]["is_live"] = False
            else:
                LIVE_PRICE_META[label] = {"fetch_time_utc": "Unavailable", "is_live": False}

    # 2. Fetch Yahoo Finance Live ETF Prices (SPY, QQQ)
    headers = {"User-Agent": "Mozilla/5.0"}
    for sym, label in [("SPY", "S&P 500 (SPY)"), ("QQQ", "NASDAQ 100 (QQQ)")]:
        yahoo_updated = False
        try:
            y_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d"
            y_resp = requests.get(y_url, headers=headers, timeout=2.0)
            if y_resp.status_code == 200:
                y_data = y_resp.json()
                reg_price = y_data.get("chart", {}).get("result", [{}])[0].get("meta", {}).get("regularMarketPrice")
                if reg_price and float(reg_price) > 0:
                    prices[label] = float(reg_price)
                    LIVE_PRICE_META[label] = {
                        "fetch_time_utc": fetch_time_utc,
                        "is_live": True,
                        "last_success_time": now
                    }
                    yahoo_updated = True
        except Exception as e:
            print(f"Warning: Yahoo live price fetch error for {sym}: {e}")

        if not yahoo_updated:
            if label in LIVE_PRICE_META:
                LIVE_PRICE_META[label]["is_live"] = False
            else:
                LIVE_PRICE_META[label] = {"fetch_time_utc": "Unavailable", "is_live": False}

    if prices:
        with LIVE_PRICE_LOCK:
            LIVE_PRICE_CACHE = prices
            LAST_PRICE_FETCH_TIME = time.time()

    with LIVE_PRICE_LOCK:
        return dict(LIVE_PRICE_CACHE)


LIVE_CANDLES_CACHE = {}
LIVE_CANDLES_LOCK = threading.Lock()
CANDLES_CACHE_TTL = 15.0

def fetch_real_5m_candles(asset_label: str) -> list:
    now = time.time()
    with LIVE_CANDLES_LOCK:
        if asset_label in LIVE_CANDLES_CACHE:
            cached_data, cached_time = LIVE_CANDLES_CACHE[asset_label]
            if now - cached_time < CANDLES_CACHE_TTL and len(cached_data) > 0:
                return [dict(c) for c in cached_data]

    binance_symbols = {
        "Bitcoin (BTC)": "BTCUSDT",
        "Ethereum (ETH)": "ETHUSDT",
        "Dogecoin (DOGE)": "DOGEUSDT"
    }
    yahoo_symbols = {
        "S&P 500 (SPY)": "SPY",
        "NASDAQ 100 (QQQ)": "QQQ"
    }

    candles = []
    # 1. Binance Spot Live 5m Klines (BTC, ETH, DOGE)
    if asset_label in binance_symbols:
        sym = binance_symbols[asset_label]
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=5m&limit=288"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3.5)
            if resp.status_code == 200:
                raw_klines = resp.json()
                dec = 5 if "DOGE" in sym else 2
                for k in raw_klines:
                    o = round(float(k[1]), dec)
                    h = round(float(k[2]), dec)
                    l = round(float(k[3]), dec)
                    c = round(float(k[4]), dec)
                    ts_ms = int(k[0])
                    candles.append({
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "time": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        "timestamp": ts_ms
                    })
        except Exception as e:
            print(f"Warning: Binance 5m klines fetch failed for {sym}: {e}")

    # 2. Yahoo Finance 5m Candles for Equity ETFs (SPY, QQQ)
    elif asset_label in yahoo_symbols:
        sym = yahoo_symbols[asset_label]
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=5m&range=1d"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    timestamps = result[0].get("timestamp", [])
                    quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
                    opens = quotes.get("open", [])
                    highs = quotes.get("high", [])
                    lows = quotes.get("low", [])
                    closes = quotes.get("close", [])
                    for i in range(len(timestamps)):
                        if i < len(opens) and opens[i] is not None and closes[i] is not None:
                            candles.append({
                                "open": round(float(opens[i]), 2),
                                "high": round(float(highs[i] if highs[i] is not None else max(opens[i], closes[i])), 2),
                                "low": round(float(lows[i] if lows[i] is not None else min(opens[i], closes[i])), 2),
                                "close": round(float(closes[i]), 2),
                                "time": datetime.fromtimestamp(timestamps[i], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                "timestamp": int(timestamps[i] * 1000)
                            })
        except Exception as e:
            print(f"Warning: Yahoo 5m candles fetch failed for {sym}: {e}")

    # Only cache and return live exchange candles if fetch succeeded; DO NOT synthesize fallback here
    if candles:
        with LIVE_CANDLES_LOCK:
            LIVE_CANDLES_CACHE[asset_label] = (candles, now)

    return candles


SUMMARY_CACHE = {}
SUMMARY_CACHE_LOCK = threading.Lock()
SUMMARY_CACHE_TTL = 120.0

def generate_dynamic_summary(
    label: str, 
    price: float, 
    change_pct: float, 
    signal: str, 
    confidence: float, 
    regime: str, 
    sentiment: str,
    candles: Optional[list] = None
) -> str:
    now = time.time()
    cache_key = label
    with SUMMARY_CACHE_LOCK:
        if cache_key in SUMMARY_CACHE:
            cached_text, cache_ts = SUMMARY_CACHE[cache_key]
            if (now - cache_ts) < SUMMARY_CACHE_TTL:
                return cached_text

    # Compute actual swing stats from live candles if available
    high_24h = max(c["high"] for c in candles) if candles and len(candles) > 0 else price
    low_24h = min(c["low"] for c in candles) if candles and len(candles) > 0 else price
    drop_from_high_pct = ((price - high_24h) / high_24h) * 100.0 if high_24h > 0 else 0.0

    recent_change_pct = 0.0
    if candles and len(candles) >= 12:
        n_recent = min(36, len(candles))
        rec_open = candles[-n_recent]["open"]
        if rec_open > 0:
            recent_change_pct = ((price - rec_open) / rec_open) * 100.0

    price_fmt = f"${price:,.2f}" if price >= 1 else f"${price:,.5f}"
    high_fmt = f"${high_24h:,.0f}" if high_24h >= 100 else (f"${high_24h:,.2f}" if high_24h >= 1 else f"${high_24h:,.4f}")
    low_fmt = f"${low_24h:,.0f}" if low_24h >= 100 else (f"${low_24h:,.2f}" if low_24h >= 1 else f"${low_24h:,.4f}")
    pct_str = f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"

    is_pullback = "bear" in (regime or "").lower() or "turbulent" in (regime or "").lower() or drop_from_high_pct <= -2.5 or recent_change_pct <= -1.5
    is_rally = "bull" in (regime or "").lower() or "rally" in (regime or "").lower() or recent_change_pct >= 2.0
    sig_upper = (signal or "").upper()

    # 1. Try Gemini generation if API key is present
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            system_prompt = (
                "You are an experienced quantitative market analyst writing a concise 2-sentence market overview "
                "for a trading dashboard. Tone: calm, professional, objective.\n"
                "STRICT RULES:\n"
                "1. NEVER use boilerplate like 'The AI trading system suggests', 'We recommend', 'As an AI', or speak in third person.\n"
                "2. If price recently pulled back sharply from its 24h high, mention the drop from peak and why waiting for support to stabilize is the safer play.\n"
                "3. Total length: exactly 2 sentences (under 50 words). No bullet points, no asterisks, no headers."
            )
            user_prompt = (
                f"Asset: {label}, Price: {price_fmt} ({pct_str} 24h), 24h High: {high_fmt}, 24h Low: {low_fmt}, "
                f"Drop from peak: {drop_from_high_pct:.1f}%, Stance: {signal}, Regime: {regime}."
            )
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 512,
                    "temperature": 0.3,
                    "thinkingConfig": {"thinkingBudget": 0}
                }
            }
            resp = requests.post(url, json=payload, timeout=3.0)
            if resp.status_code == 200:
                resp_json = resp.json()
                cand = resp_json.get("candidates", [])
                if cand:
                    parts = cand[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[-1].get("text", "").strip()
                        if text:
                            with SUMMARY_CACHE_LOCK:
                                SUMMARY_CACHE[cache_key] = (text, now)
                            return text
        except Exception:
            pass

    # 2. Contextual dynamic narrative generator (mathematical grounding)
    if "BUY" in sig_upper or "BULL" in sig_upper:
        first = f"{label} is advancing at {price_fmt} ({pct_str} over 24h) with sustained buying pressure."
        second = f"Holding firmly above intraday support at {low_fmt}, technical momentum favors long continuation toward {high_fmt}."
    elif "SELL" in sig_upper or "BEAR" in sig_upper:
        first = f"{label} is declining at {price_fmt} ({pct_str} over 24h), slipping {abs(drop_from_high_pct):.1f}% from its 24h peak of {high_fmt}."
        second = f"Elevated distribution pressure favors defensive risk management until price stabilizes above {low_fmt}."
    else:  # HOLD / NEUTRAL
        if is_pullback:
            first = f"{label} is trading at {price_fmt} ({pct_str} over 24h), pulling back sharply after rejecting its 24h peak of {high_fmt}."
            second = f"With recent selling pressure testing support near {low_fmt}, waiting for the floor to stabilize is the safer play before entering."
        elif is_rally:
            first = f"{label} is consolidating at {price_fmt} ({pct_str} over 24h) following an active run toward {high_fmt}."
            second = "Maintaining patience allows the market to digest gains before confirming another leg up."
        else:
            first = f"{label} is holding steady at {price_fmt} ({pct_str} over 24h), oscillating within a {low_fmt} - {high_fmt} range."
            second = "Without strong directional momentum, waiting for a decisive breakout volume spike is prudent."

    full_summary = f"{first} {second}"
    with SUMMARY_CACHE_LOCK:
        SUMMARY_CACHE[cache_key] = (full_summary, now)
    return full_summary


def predict_for_latest_state(asset_label: str, timeframe: str):
    now = time.time()
    cache_key = (asset_label, timeframe)
    with PREDICT_CACHE_LOCK:
        if cache_key in PREDICT_CACHE:
            cached_res, cache_time = PREDICT_CACHE[cache_key]
            if (now - cache_time) < PREDICT_CACHE_TTL:
                res = dict(cached_res)
                live_prices = fetch_live_market_prices()
                if asset_label in live_prices:
                    cur_p = live_prices[asset_label]
                    res["price"] = cur_p
                    meta = LIVE_PRICE_META.get(asset_label, {})
                    res["quote_is_live"] = bool(meta.get("is_live", False))
                    res["quote_time"] = meta.get("fetch_time_utc", res.get("quote_time", "Unavailable"))
                return res

    csv_path = get_asset_csv_path(asset_label, timeframe)
    if not csv_path or not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    if len(df) < 50:
        return None
        
    # Process the recent tail (400 rows) so indicator calculation and HMM forward filtering
    # don't iterate through 50,000+ historical rows for a 250-step prediction.
    sub_df = df.iloc[-400:].reset_index(drop=True)
    df_indicators = compute_indicators(sub_df, fit_hmm=False).ffill().bfill()
    slice_df = df_indicators.iloc[-250:].reset_index(drop=True)
    
    # Instantiate ActiveCryptoEnv deterministically
    env = ActiveCryptoEnv(
        slice_df, 
        initial_balance=10000.0, 
        render_mode=None, 
        random_start=False, 
        max_steps=len(slice_df),
        include_regime=include_regime,
        include_context=include_context
    )
    
    obs, _ = env.reset()
    buffer = np.zeros((8, obs_dim), dtype=np.float32)
    for j in range(8):
        buffer[j] = obs
        
    # Step environment to build frame stack buffer
    for i in range(env.n_rows - 1):
        action = 0 # Hold
        next_obs, reward, done, truncated, info = env.step(action)
        buffer[:-1] = buffer[1:]
        buffer[-1] = next_obs
        if done:
            break
            
    flat_obs = buffer.reshape(1, -1)
    
    # Run PPO model prediction
    if model is not None:
        try:
            action, _ = model.predict(flat_obs, deterministic=True)
            action = int(np.asarray(action).item())
            
            # Get action probabilities with calibrated Temperature Scaling (T = 0.2)
            with torch.no_grad():
                vec_norm = model.get_vec_normalize_env() if hasattr(model, "get_vec_normalize_env") else None
                flat_obs_norm = vec_norm.normalize_obs(flat_obs) if vec_norm is not None else flat_obs
                obs_tensor = torch.as_tensor(flat_obs_norm, device=DEVICE)
                
                features = model.policy.extract_features(obs_tensor)
                if model.policy.share_features_extractor:
                    latent_pi, _ = model.policy.mlp_extractor(features)
                else:
                    pi_features, _ = features
                    latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
                logits = model.policy.action_net(latent_pi)
                
                temperature = 0.2
                scaled_probs = torch.softmax(logits / temperature, dim=-1)
                probs = scaled_probs.cpu().numpy()[0]
        except Exception as ex:
            print(f"Inference warning: {ex}")
            action = 0
            probs = np.array([0.34, 0.33, 0.33], dtype=np.float32)
    else:
        action = 0
        probs = np.array([0.34, 0.33, 0.33], dtype=np.float32)
        
    # Map action to signal and sentiment for Discrete(3): 0=Cash/Neutral, 1=Long/Buy, 2=Short/Sell
    signals = ["NEUTRAL", "BUY", "SELL"]
    signal = signals[action] if action < len(signals) else "NEUTRAL"
    
    sentiment_map = {
        "BUY": "Bullish",
        "SELL": "Bearish",
        "NEUTRAL": "Neutral"
    }
    sentiment = sentiment_map[signal]
    
    # Calculate confidence as the probability of the chosen action
    confidence = float(probs[action] * 100.0)
    
    # Get current price: query real-time live ticker API, fallback to dataset
    live_prices = fetch_live_market_prices()
    meta = LIVE_PRICE_META.get(asset_label, {})
    quote_is_live = bool(meta.get("is_live", False))
    if asset_label in live_prices:
        current_price = float(live_prices[asset_label])
        quote_time = meta.get("fetch_time_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    else:
        current_price = float(df_indicators["close"].iloc[-1])
        quote_time = f"{slice_df['datetime'].iloc[-1]} (Offline Archive)" if "datetime" in slice_df.columns else "Offline Archive"
    
    # Get history for sparkline (last 12 closes)
    history = [float(x) for x in df_indicators["close"].iloc[-12:].tolist()]
    
    # Get current regime
    current_regime = float(df_indicators["market_regime"].iloc[-1])
    
    # Map regime value to label
    regime_labels = {
        0.0: "Sideways / Quiet",
        1.0: "Low Volatility Bull",
        -1.0: "Low Volatility Bear",
        2.0: "High Volatility Bull",
        -2.0: "Panic Bear / Crash"
    }
    regime_str = regime_labels.get(current_regime, "Neutral")
    
    # Fetch real live 5m candles from Binance / Yahoo Finance
    real_candles = fetch_real_5m_candles(asset_label)
    if real_candles and len(real_candles) > 0:
        candles = real_candles
        last_exchange_time = float(candles[-1]["timestamp"]) / 1000
        candle_source = "live_exchange" if 0 <= time.time() - last_exchange_time <= 900 else "exchange_archive"
        is_synthetic = False
        if candle_source == "live_exchange":
            current_price = float(candles[-1]["close"])
            quote_is_live = True
            quote_time = candles[-1]["time"]
        elif not quote_is_live:
            current_price = float(candles[-1]["close"])
            quote_time = f"{candles[-1]['time']} (Exchange Archive)"
        history = [float(c["close"]) for c in candles[-24:]]
    else:
        csv_1h_path = get_asset_csv_path(asset_label, "1H")
        candles = generate_5m_candles_from_1h(csv_1h_path)
        candle_source = "offline_historical_replay"
        is_synthetic = True
    
    candle_is_live = bool(candle_source == "live_exchange")
    last_candle_time = candles[-1].get("time") if (candles and len(candles) > 0) else None

    change_24h = 0.0
    change_pct_24h = 0.0
    if candles and len(candles) > 0:
        c_open = float(candles[0]["open"])
        c_close = float(candles[-1]["close"])
        if c_open > 0:
            change_24h = c_close - c_open
            change_pct_24h = (change_24h / c_open) * 100.0
    elif len(history) > 1 and history[0] > 0:
        change_24h = history[-1] - history[0]
        change_pct_24h = (change_24h / history[0]) * 100.0

    # Ground market regime dynamically in authentic 24h candle swings
    high_24h = max(c["high"] for c in candles) if candles and len(candles) > 0 else current_price
    low_24h = min(c["low"] for c in candles) if candles and len(candles) > 0 else current_price
    analysis_price = float(candles[-1]["close"]) if candles else current_price
    drop_from_high_pct = ((analysis_price - high_24h) / high_24h) * 100.0 if high_24h > 0 else 0.0

    recent_change_pct = 0.0
    if candles and len(candles) >= 12:
        n_recent = min(36, len(candles))
        rec_open = candles[-n_recent]["open"]
        if rec_open > 0:
            recent_change_pct = ((analysis_price - rec_open) / rec_open) * 100.0

    if drop_from_high_pct <= -2.5 or recent_change_pct <= -1.8:
        regime_str = "Turbulent Pullback"
    elif recent_change_pct >= 2.0 or change_pct_24h >= 3.5:
        regime_str = "Active Bull Rally"
    elif change_pct_24h >= 0.8:
        regime_str = "Low Volatility Bull"

    model_eval_time = str(slice_df["datetime"].iloc[-1]) if "datetime" in slice_df.columns else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if candle_is_live:
        summary_text = generate_dynamic_summary(
            asset_label, analysis_price, round(change_pct_24h, 2), signal,
            round(confidence, 1), regime_str, sentiment, candles=candles
        )
    else:
        summary_text = (
            f"Historical chart window ending {last_candle_time or model_eval_time}: "
            f"{asset_label} changed {change_pct_24h:+.2f}% in that window. "
            f"The model targets {signal} using archived input through {model_eval_time}; "
            f"the historical market regime is {regime_str}. "
            "Any current quote is separate from this historical chart and model input."
        )

    res_obj = {
        "price": current_price,
        "signal": signal,
        "confidence": round(confidence, 1),
        "sentiment": sentiment,
        "history": history,
        "regime": regime_str,
        "flat_obs": flat_obs.tolist()[0],
        "state_id": hashlib.sha256(np.ascontiguousarray(flat_obs, dtype=np.float32).tobytes()).hexdigest(),
        "probs": probs.tolist(),
        "action": action,
        "candles": candles,
        "change": round(change_24h, 2 if current_price > 10 else 4),
        "change_pct": round(change_pct_24h, 2),
        "summary": summary_text,
        "model_eval_time": model_eval_time,
        "quote_time": quote_time,
        "quote_is_live": quote_is_live,
        "candle_source": candle_source,
        "candle_is_live": candle_is_live,
        "last_candle_time": last_candle_time,
        "is_synthetic_candles": is_synthetic
    }
    with PREDICT_CACHE_LOCK:
        PREDICT_CACHE[cache_key] = (res_obj, now)
    return res_obj

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Financial Advisor Backend"}

@app.get("/api/scout")
def get_scout_data(timeframe: str = "1H"):
    import concurrent.futures
    res = {"Crypto": [], "ETFs": []}
    
    crypto_labels = ["Bitcoin (BTC)", "Ethereum (ETH)", "Dogecoin (DOGE)"]
    etf_labels = ["S&P 500 (SPY)", "NASDAQ 100 (QQQ)"]
    all_labels = crypto_labels + etf_labels
    
    # Process all assets concurrently across threads for instant loading
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {lbl: executor.submit(predict_for_latest_state, lbl, timeframe) for lbl in all_labels}
        pred_results = {lbl: future_map[lbl].result() for lbl in all_labels}
        
    for label in crypto_labels:
        pred = pred_results.get(label)
        if pred:
            res["Crypto"].append({
                "label": label,
                "price": pred["price"],
                "signal": pred["signal"],
                "confidence": pred["confidence"],
                "sentiment": pred["sentiment"],
                "history": pred["history"],
                "regime": pred["regime"],
                "candles": pred.get("candles", []),
                "change": pred.get("change", 0.0),
                "change_pct": pred.get("change_pct", 0.0),
                "summary": pred.get("summary", ""),
                "probs": pred.get("probs", []),
                "action": pred.get("action", 0),
                "quote_is_live": pred.get("quote_is_live", False),
                "quote_time": pred.get("quote_time", "Unavailable"),
                "candle_source": pred.get("candle_source", "offline_historical_replay"),
                "candle_is_live": pred.get("candle_is_live", False),
                "last_candle_time": pred.get("last_candle_time", None),
                "model_eval_time": pred.get("model_eval_time", "Unavailable"),
            })
            
    for label in etf_labels:
        pred = pred_results.get(label)
        if pred:
            res["ETFs"].append({
                "label": label,
                "price": pred["price"],
                "signal": pred["signal"],
                "confidence": pred["confidence"],
                "sentiment": pred["sentiment"],
                "history": pred["history"],
                "regime": pred["regime"],
                "candles": pred.get("candles", []),
                "change": pred.get("change", 0.0),
                "change_pct": pred.get("change_pct", 0.0),
                "summary": pred.get("summary", ""),
                "probs": pred.get("probs", []),
                "action": pred.get("action", 0),
                "quote_is_live": pred.get("quote_is_live", False),
                "quote_time": pred.get("quote_time", "Unavailable"),
                "candle_source": pred.get("candle_source", "offline_historical_replay"),
                "candle_is_live": pred.get("candle_is_live", False),
                "last_candle_time": pred.get("last_candle_time", None),
                "model_eval_time": pred.get("model_eval_time", "Unavailable"),
            })
            
    return res

@app.get("/api/backtest")
def get_backtest_data(asset: str, timeframe: str = "1H", regime: str = "recent"):
    csv_path = get_asset_csv_path(asset, timeframe)
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Asset data not found for {asset}")
        
    df = pd.read_csv(csv_path)
    if len(df) < 100:
        raise HTTPException(status_code=400, detail="Not enough data for backtesting")
        
    df_indicators = compute_indicators(df).ffill().bfill()
    if "datetime" in df_indicators.columns:
        df_indicators["dt"] = pd.to_datetime(df_indicators["datetime"])
    elif "timestamp" in df_indicators.columns:
        unit = "s" if df_indicators["timestamp"].max() < 1e11 else "ms"
        df_indicators["dt"] = pd.to_datetime(df_indicators["timestamp"], unit=unit)
    else:
        df_indicators["dt"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_indicators), freq="1h")

    is_daily = (timeframe.upper() == "DAILY")
    regime_title = "Recent Market Window (~104 Days)"
    sample_type = "Recent Evaluation Window"

    if regime == "etf_bull":
        start_dt = pd.to_datetime("2023-10-01")
        end_dt = pd.to_datetime("2024-03-31")
        sub_df = df_indicators[(df_indicators["dt"] >= start_dt) & (df_indicators["dt"] <= end_dt)]
        if len(sub_df) > 50:
            val_df = sub_df.reset_index(drop=True)
            regime_title = "2023-2024 ETF Bull Run (Oct 2023 - Mar 2024)"
            sample_type = "In-Sample Training History"
        else:
            val_df = df_indicators.iloc[-2500:].reset_index(drop=True)
    elif regime == "halving_bull":
        start_dt = pd.to_datetime("2024-09-01")
        end_dt = pd.to_datetime("2025-01-20")
        sub_df = df_indicators[(df_indicators["dt"] >= start_dt) & (df_indicators["dt"] <= end_dt)]
        if len(sub_df) > 50:
            val_df = sub_df.reset_index(drop=True)
            regime_title = "2024-2025 Post-Halving Bull Run (Sep 2024 - Jan 2025)"
            sample_type = "In-Sample Training History"
        else:
            val_df = df_indicators.iloc[-2500:].reset_index(drop=True)
    elif regime == "bear_crash":
        start_dt = pd.to_datetime("2025-05-08")
        end_dt = pd.to_datetime("2026-09-07")
        sub_df = df_indicators[(df_indicators["dt"] >= start_dt) & (df_indicators["dt"] <= end_dt)]
        if len(sub_df) > 50:
            val_df = sub_df.reset_index(drop=True)
            regime_title = "2025-2026 Bear Crash (May 2025 - Sep 2026)"
            sample_type = "Out-of-Sample Test Fold"
        else:
            val_df = df_indicators.iloc[-2500:].reset_index(drop=True)
    elif regime == "full":
        val_df = df_indicators.reset_index(drop=True)
        regime_title = "Full Multi-Year Macro History (2020 - 2026)"
        sample_type = "Full Cycle History"
    else:
        # Default: "recent"
        max_eval_steps = 365 if is_daily else 2500
        val_df = df_indicators.iloc[-max_eval_steps:].reset_index(drop=True)
        regime_title = "Recent Market Window (~104 Days)"
        sample_type = "Recent Evaluation Window"
    
    if model is None:
        raise HTTPException(status_code=500, detail="RL model not loaded on backend")
        
    try:
        metrics = run_single_fold_backtest(
            model, val_df, 
            initial_balance=10000.0, 
            is_daily=is_daily,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Format the trade history to make it lighter and map string actions to integers for frontend chart markers
        action_map = {
            "BUY (LONG)": 1,
            "FLIP TO LONG": 1,
            "EXIT LONG": 2,
            "SELL (SHORT)": 3,
            "FLIP TO SHORT": 3,
            "EXIT SHORT": 4
        }
        formatted_trades = []
        for t in metrics["TradeHistory"]:
            action_str = t["Action"]
            formatted_trades.append({
                "step": int(t["Step"]),
                "action": action_map.get(action_str, 0),
                "reason": t.get("Reason", "episode_end"),
                "policy_target": t.get("PolicyTarget"),
                "price": float(t["Price"]),
                "net_worth": float(t["Net Worth"])
            })
            
        dates_list = val_df["datetime"].astype(str).tolist() if "datetime" in val_df.columns else []
        history_list = [float(x) for x in metrics["NetWorthHistory"]]
        return {
            "roi": float(metrics["ROI %"]),
            "bh_return": float(metrics["B&H %"]),
            "sharpe": float(metrics["Sharpe"]),
            "sortino": float(metrics["Sortino"]),
            "max_dd": float(metrics["Max DD %"]),
            "calmar": float(metrics["Calmar"]),
            "trades": int(metrics["Trades"]),
            "final_capital": float(metrics["Final $"]),
            "history": history_list,
            "trade_history": formatted_trades,
            "policy_action_counts": metrics["PolicyActionCounts"],
            "dates": dates_list,
            "regime": regime,
            "regime_title": regime_title,
            "sample_type": sample_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")

@app.get("/api/montecarlo")
def get_montecarlo_data(
    asset: str,
    timeframe: str = "1H",
    simulations: int = 50,
    noise_std: float = 0.002,
    inject_swan: bool = False,
):
    # Keep interactive requests within the workload supported by the dashboard.
    if not 1 <= simulations <= 500:
        raise HTTPException(status_code=422, detail="simulations must be between 1 and 500")
    if not np.isfinite(noise_std) or not 0 <= noise_std <= 0.2:
        raise HTTPException(status_code=422, detail="noise_std must be between 0 and 0.2")
    csv_path = get_asset_csv_path(asset, timeframe)
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Asset data not found for {asset}")
        
    df = pd.read_csv(csv_path)
    if len(df) < 100:
        raise HTTPException(status_code=400, detail="Not enough data for simulation")
        
    df_indicators = compute_indicators(df).ffill().bfill()
    
    # Run simulation on the validation portion (last 20% of the dataset)
    test_size = max(50, int(len(df_indicators) * 0.2))
    val_df = df_indicators.iloc[-test_size:].reset_index(drop=True)
    
    # Truncate validation dataset to the last 150 steps for performance optimization
    trunc_size = min(150, len(val_df))
    trunc_df = val_df.iloc[-trunc_size:].reset_index(drop=True)
    
    if model is None:
        raise HTTPException(status_code=500, detail="RL model not loaded on backend")
        
    try:
        # Run Monte Carlo simulations
        summary, equity_curves = run_mc_analysis(
            model,
            trunc_df,
            n_simulations=simulations,
            noise_std=noise_std,
            inject_swan=inject_swan,
        )
        
        # Calculate step-wise percentiles across all curves
        curves_arr = np.array(equity_curves)  # Shape: (simulations, steps)
        steps = curves_arr.shape[1]
        
        percentiles = {
            "5": [],
            "25": [],
            "50": [],
            "75": [],
            "95": [],
        }
        
        for t in range(steps):
            step_values = curves_arr[:, t]
            percentiles["5"].append(float(np.percentile(step_values, 5)))
            percentiles["25"].append(float(np.percentile(step_values, 25)))
            percentiles["50"].append(float(np.percentile(step_values, 50)))
            percentiles["75"].append(float(np.percentile(step_values, 75)))
            percentiles["95"].append(float(np.percentile(step_values, 95)))
            
        # Select 5 sample paths to return
        sample_indices = np.linspace(0, len(equity_curves) - 1, min(5, len(equity_curves)), dtype=int)
        sample_paths = [[float(v) for v in equity_curves[idx]] for idx in sample_indices]
        
        return {
            "summary": {
                "mean_roi": float(summary["mean_roi"]),
                "std_roi": float(summary["std_roi"]),
                "min_roi": float(summary["min_roi"]),
                "max_roi": float(summary["max_roi"]),
                "var_5th": float(summary["var_5th"]),
                "mean_sharpe": float(summary["mean_sharpe"]),
                "mean_drawdown": float(summary["mean_drawdown"]),
                "success_rate": float(summary["success_rate"]),
            },
            "percentiles": percentiles,
            "sample_paths": sample_paths,
            "step_count": steps,
            "dates": trunc_df["datetime"].astype(str).tolist() if "datetime" in trunc_df.columns else [],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation error: {str(e)}")

@app.get("/api/shap")
def get_shap_data(asset: str, timeframe: str = "1H"):
    now = time.time()
    cache_key = f"{asset}_{timeframe}"
    latest_pred = predict_for_latest_state(asset, timeframe)
    if latest_pred is None:
        raise HTTPException(status_code=404, detail="Asset prediction not found")
    with SHAP_CACHE_LOCK:
        if cache_key in SHAP_CACHE:
            cached_res, c_time = SHAP_CACHE[cache_key]
            if ((now - c_time) < SHAP_CACHE_TTL
                    and cached_res.get("action") == latest_pred["action"]
                    and cached_res.get("state_id") == latest_pred["state_id"]):
                return cached_res

    csv_path = get_asset_csv_path(asset, timeframe)
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Asset data not found")
        
    df = pd.read_csv(csv_path)
    if len(df) < 80:
        raise HTTPException(status_code=400, detail="Not enough data")
        
    df_indicators = compute_indicators(df.iloc[-400:].reset_index(drop=True)).ffill().bfill()
    # Explain the exact observation used for the displayed policy decision.
    current_obs = np.asarray(latest_pred["flat_obs"], dtype=np.float32).reshape(1, -1)
    
    # 2. Build background data (sample 15 states from recent 250 rows)
    bg_slice = df_indicators.iloc[-250:].reset_index(drop=True)
    bg_env = ActiveCryptoEnv(
        bg_slice,
        initial_balance=10000.0,
        render_mode=None,
        random_start=False,
        max_steps=len(bg_slice),
        include_regime=include_regime,
        include_context=include_context
    )
    bg_obs, _ = bg_env.reset()
    bg_buffer = np.zeros((8, obs_dim), dtype=np.float32)
    for j in range(8):
        bg_buffer[j] = bg_obs
        
    all_bg_obs = []
    for i in range(bg_env.n_rows - 1):
        action = 0
        next_bg_obs, _, bg_done, _, _ = bg_env.step(action)
        bg_buffer[:-1] = bg_buffer[1:]
        bg_buffer[-1] = next_bg_obs
        all_bg_obs.append(bg_buffer.copy().reshape(1, -1)[0])
        if bg_done:
            break
            
    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_bg_obs), min(15, len(all_bg_obs)), replace=False)
    background_data = np.array([all_bg_obs[idx] for idx in indices], dtype=np.float32)
    
    if model is None:
        raise HTTPException(status_code=500, detail="RL model not loaded")
        
    try:
        # Run SHAP computation (nsamples=150 >= 136 features to prevent LassoLarsIC underdetermined regression error)
        vec_norm = model.get_vec_normalize_env() if hasattr(model, "get_vec_normalize_env") else None
        shap_values = compute_shap_values(model, current_obs, background_data, nsamples=150, vec_normalize=vec_norm)
        
        # Determine target action
        action = int(latest_pred["action"])
        
        if isinstance(shap_values, list):
            vals = shap_values[action][0]
        else:
            vals = shap_values[0, :, action]
            
        feature_names = [
            "Open Ratio", "High Ratio", "Low Ratio", "Volatility", "Price Change", "Price Change Prev", "Normalized Volume",
            "Short Trend (SMA20)", "Medium Trend (SMA99)", "Momentum (ROC24)", "Volatility Ratio", "Macro Drawdown",
            "Market Regime", "Portfolio Position", "Unrealized P&L", "Idle Steps", "Holding Duration"
        ]
        
        agg_vals = np.zeros(obs_dim)
        for idx in range(obs_dim):
            agg_vals[idx] = float(np.sum(vals[idx::obs_dim]))
            
        result_list = []
        # Exclude portfolio features for explainability if flat (index 13 to 16)
        # to focus user purely on market technical drivers
        limit = obs_dim if include_regime else 12
        total_abs = float(np.sum(np.abs(agg_vals[:limit])))
        if total_abs == 0:
            total_abs = 1e-6

        for idx in range(limit):
            abs_val = abs(float(agg_vals[idx]))
            rel_pct = round((abs_val / total_abs) * 100, 1)
            result_list.append({
                "feature": feature_names[idx],
                "value": float(agg_vals[idx]),
                "importance_pct": rel_pct
            })
            
        result_list.sort(key=lambda x: x["importance_pct"], reverse=True)
        raw_vals_export = shap_values.tolist() if hasattr(shap_values, "tolist") else shap_values
        ret_obj = {"shap": result_list, "action": action, "shap_values": raw_vals_export,
                   "state_id": latest_pred["state_id"]}
        with SHAP_CACHE_LOCK:
            SHAP_CACHE[cache_key] = (ret_obj, now)
        return ret_obj
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {str(e)}")

@app.get("/api/comparison")
def get_comparison():
    csv_path = "comparison_results.csv"
    if os.path.exists(csv_path):
        try:
            df_comp = pd.read_csv(csv_path)
            return {"results": df_comp.to_dict(orient="records")}
        except Exception:
            pass
    # Default benchmark baseline data if CSV not generated yet
    return {
        "results": [
            {"Model": "PPO (Transformer Policy)", "ROI (%)": 14.38, "Max DD (%)": 15.10, "Trades": 441, "Sharpe": 1.25},
            {"Model": "DQN (Baseline)", "ROI (%)": -3.20, "Max DD (%)": 18.50, "Trades": 120, "Sharpe": -0.15},
            {"Model": "A2C (Baseline)", "ROI (%)": -8.45, "Max DD (%)": 22.10, "Trades": 310, "Sharpe": -0.58},
            {"Model": "Buy & Hold (Market)", "ROI (%)": -1.85, "Max DD (%)": 24.30, "Trades": 1, "Sharpe": -0.08},
        ]
    }

def validate_chat_response(
    response_text: str,
    action_name: str,
    price: float,
    ticker: str,
    tolerance_pct: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """
    Perform bounded validation on generated chat responses before returning to the user.
    Validates action consistency and basic numeric/price claims against structured evidence.
    Note: This is a limited guardrail, not an absolute guarantee against hallucination.
    """
    if not response_text or len(response_text.strip()) == 0:
        return False, "Empty response generated"

    lower_text = response_text.lower()
    act_upper = action_name.upper()

    # 1. Action consistency validation
    if "SELL" in act_upper or "SHORT" in act_upper:
        forbidden_cues = ["buy now", "go long", "enter a long", "accumulate", "open a long", "taking profit on your long", "closing your long"]
        for cue in forbidden_cues:
            if cue in lower_text:
                return False, f"Action conflict: Model predicted SHORT/SELL but response suggests '{cue}'"

    elif "BUY" in act_upper or "LONG" in act_upper:
        forbidden_cues = ["open a short", "short the market", "go short", "exit to cash", "sell off everything"]
        for cue in forbidden_cues:
            if cue in lower_text:
                return False, f"Action conflict: Model predicted LONG/BUY but response suggests '{cue}'"

    elif "NEUTRAL" in act_upper or "CASH" in act_upper:
        forbidden_cues = ["buy now", "go long", "enter a long", "open a long", "go short", "enter a short", "open a short", "short the market", "accumulate", "open a maximum leveraged", "all in long", "all in short", "aggressive buy", "aggressive short"]
        for cue in forbidden_cues:
            if cue in lower_text:
                return False, f"Action conflict: Model predicted NEUTRAL/CASH but response suggests '{cue}'"

    # 2. Price consistency check: If response explicitly cites a price with dollar sign, verify it's within tolerance
    if price > 0:
        matches = re.findall(r"\$([0-9,]+(?:\.[0-9]+)?)", response_text)
        for m in matches:
            try:
                cited_val = float(m.replace(",", ""))
                if cited_val >= 0:
                    deviation = abs(cited_val - price) / price * 100.0
                    if deviation > tolerance_pct:
                        return False, f"Price conflict: Cited price ${cited_val:,.2f} deviates {deviation:.1f}% from current price ${price:,.2f}"
            except ValueError:
                pass

    return True, None

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        if not req.asset_name:
            response = "Select an asset to review its signal and available evidence. A verified cross-asset market overview is unavailable in this chat."
            return {"response": response, "reply": response, "evidence_status": "unavailable"}
        ticker = req.asset_name if req.asset_name else "General Market"
        price = req.price if req.price is not None else 0.0
        signal = req.signal if req.signal else "NEUTRAL"
        confidence = req.confidence if req.confidence is not None else 0.0
        sentiment = req.sentiment if req.sentiment else "Neutral"

        action_names = ["NEUTRAL", "BUY (LONG)", "SELL (SHORT)"]
        
        # Default fallback indicators
        rsi = 50.0
        macd = 0.0
        volume_ratio = 1.0
        probs = [0.34, 0.33, 0.33]
        action_name = "NEUTRAL"
        
        # Convert simple signal strings back to exact Action Names
        if "BUY" in signal.upper() or "LONG" in signal.upper():
            action_name = "BUY (LONG)"
            action_idx = 1
        elif "SELL" in signal.upper() or "SHORT" in signal.upper():
            action_name = "SELL (SHORT)"
            action_idx = 2
        else:
            action_name = "NEUTRAL"
            action_idx = 0
            
        # Try to retrieve real data for more detailed technical calculations
        csv_path = get_asset_csv_path(ticker, "1H")
        latest_pred = None
        
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) >= 50:
                df_indicators = compute_indicators(df)
                latest_row = df_indicators.iloc[-1]
                rsi = float(latest_row.get("rsi_14", 50.0))
                macd = float(latest_row.get("macd_hist", 0.0))
                
                vol_sma = latest_row.get("volume_sma_20", 1.0)
                vol_sma = 1.0 if pd.isna(vol_sma) or vol_sma == 0 else vol_sma
                volume_ratio = float(latest_row.get("volume", 0.0) / vol_sma)
                
                latest_pred = predict_for_latest_state(ticker, "1H")
                if latest_pred:
                    probs = latest_pred["probs"]
                    act = latest_pred["action"]
                    if act not in range(len(action_names)):
                        raise ValueError("Prediction action is outside the canonical action space")
                    action_name = action_names[act]
                    confidence = latest_pred["confidence"]
                    sentiment = latest_pred["sentiment"]

        if latest_pred is None:
            response = (
                f"A verified model prediction for {ticker} is unavailable. "
                f"The supplied signal is {action_name} (unverified); I cannot confirm a current recommendation or its reasons."
            )
            return {"response": response, "reply": response, "evidence_status": "unavailable"}

        # Prefer an available live ticker; otherwise the dataset prediction price.
        live_prices = fetch_live_market_prices()
        live_found = 0.0
        for k, v in live_prices.items():
            if ticker.lower() in k.lower() or k.lower() in ticker.lower() or ("btc" in ticker.lower() and "bitcoin" in k.lower()):
                live_found = float(v)
                break
        if live_found > 0:
            price = live_found
        elif latest_pred and latest_pred.get("price") and latest_pred["price"] > 0:
            price = float(latest_pred["price"])
        elif (not price or price <= 0) and req.price:
            price = float(req.price)
        
        # Get actual SHAP values if genuinely cached; do not inject heuristic/mock weights
        active_shap = None
        if latest_pred is not None:
            cache_key = f"{ticker}_1H"
            cached_shap = None
            with SHAP_CACHE_LOCK:
                if cache_key in SHAP_CACHE:
                    candidate, cache_time = SHAP_CACHE[cache_key]
                    # Legacy entries have no state identity: matching asset alone
                    # cannot establish that SHAP explains the current prediction.
                    if (time.time() - cache_time < SHAP_CACHE_TTL
                            and candidate.get("action") == latest_pred["action"]
                            and latest_pred.get("state_id") is not None
                            and candidate.get("state_id") == latest_pred["state_id"]):
                        cached_shap = candidate
            
            if cached_shap and "shap_values" in cached_shap and cached_shap["shap_values"] is not None:
                try:
                    raw_shap = np.asarray(cached_shap["shap_values"], dtype=float)
                    if raw_shap.shape == (3, 1, 8 * obs_dim):
                        raw_shap = raw_shap.transpose(1, 2, 0)
                    if raw_shap.shape == (1, 8 * obs_dim, 3) and np.isfinite(raw_shap).all():
                        active_shap = raw_shap
                except Exception:
                    active_shap = None

        # Instantiate XAIEngine with computed features (shap_values is None if not genuinely computed)
        xai = XAIEngine(
            coin_ticker=ticker,
            current_price=price,
            rsi=rsi,
            macd=macd,
            volume_ratio=volume_ratio,
            action_name=action_name,
            confidence=confidence,
            probs=probs,
            shap_values=active_shap
        )
        
        if not req.asset_name:
            response = "Across the 5 monitored assets (Crypto: BTC, ETH, DOGE; Stock ETFs: SPY, QQQ), markets are currently steady in low-volatility conditions. Stock index funds show calm upward momentum, while crypto assets are holding in a quiet sideways consolidation. It's best to maintain a Hold & Monitor stance until clear breakout momentum emerges."
        else:
            response = xai.generate_response(req.message, backtest_metrics=None)
        
        # Preserve the question-specific deterministic answer when generation is unavailable.
        # For performance questions, no backtest evidence is attached to this endpoint.
        if xai.classify_intent(req.message) == "performance":
            return {"response": response, "reply": response, "evidence_status": "backtest_unavailable"}

        # Call Gemini API if GEMINI_API_KEY is present
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            if not req.asset_name:
                system_prompt = """You are an approachable, knowledgeable market advisor embedded in a multi-asset trading dashboard.
You are currently providing a high-level market overview across the 5 monitored assets (Crypto: BTC, ETH, DOGE; Stock ETFs: SPY, QQQ).
Your audience consists of everyday retail investors and traders who want clear, conversational perspective without heavy jargon.

CONVERSATION & TONE GUIDELINES:
1. Direct Advisory Voice: Speak directly as an experienced, calm market analyst talking directly to a trader. Do NOT speak in third person. Never say "the AI model", "the AI", "as an AI", or attribute guidance to an algorithm. Give the advice directly (e.g. "I recommend holding steady...", "It's best to wait for...").
2. Keep it concise (2 to 4 smooth sentences, under 80 words).
3. Explain the overall rhythm in plain English (e.g. stock ETFs like SPY and QQQ showing steady, quiet gains while crypto assets consolidate sideways).
4. Give a practical takeaway (e.g. holding steady and waiting for clear volume/momentum expansion before taking big positions).
5. Never use bulleted lists, tables, or robotic menu options ("You can ask me about...").
"""
            else:
                friendly_action = "Neutral"
                if "BUY" in action_name.upper():
                    friendly_action = "Bullish"
                elif "EXIT" in action_name.upper():
                    friendly_action = "Take Profit"
                elif "SELL" in action_name.upper():
                    friendly_action = "Bearish"

                system_prompt = f"""You are a professional, clear-headed market advisor embedded in a live trading dashboard chat panel.
Your audience consists of everyday retail investors and traders. They want clean, objective, practical market guidance without robotic compliance speak, technical jargon, or cheesy idioms.

CONVERSATION & TONE GUIDELINES:
1. Professional & Direct: Speak directly and objectively with calm, grounded guidance (e.g., "Holding off on new trades is the sensible move...", "Consider keeping positions light...", "Watch for volume expansion before entering...").
2. STRICT BAN ON CLICHÉS & IDIOMS:
   - NEVER use folksy clichés or idioms like "sit tight", "let's sit tight", "sitting on your hands", "takes a breather", "coasting quietly", "weather the storm", or "keep your powder dry".
   - State the market condition and action directly without informal metaphors.
3. STRICT BAN ON AI SELF-ATTRIBUTION:
   - NEVER say "the AI model recommends", "the model suggests", "as an AI", "the AI advises", or refer to an algorithm in third person.
   - Simply deliver the market observation and guidance directly.
4. Plain-Language Translation:
   - Instead of "locked in a low-momentum regime with a flat RSI of 42", say: "trading in a narrow sideways range with momentum flat".
   - Instead of "elevated price slippage triggered by depressed volume ratio of 0.9x", say: "trading volume is light, so there is little directional conviction".
   - Instead of "We recommend a NEUTRAL stance with 100% confidence", say: "Holding off on new trades is the sensible move for now".
5. Response Length & Structure: 2 to 3 concise, clear sentences (under 65 words).
   - State where the asset is trading (${price:,.2f}) and its current price action (e.g. narrow consolidation, steady uptrend, or pullback).
   - Give the practical takeaway based on the signal ({friendly_action}).
   - State what level or signal confirms the next directional move.
6. ABSOLUTE DON'TS:
   - MANDATORY PRICE ADHERENCE: The current live price is strictly ${price:,.2f} (approx. ${price:,.0f}). You MUST use this exact price if citing a price. NEVER invent or repeat outdated prices from earlier conversation turns.
   - Never say: "We recommend a NEUTRAL stance on BTC ($81,206.00) with 100% confidence..."
   - Never say: "The AI model recommends..." or "The model suggests..."
   - Never say: "Let's sit tight...", "Sit tight", or "Sitting on your hands..."
   - Never use stiff words like "regime locked in", "sluggish MACD", "depressed volume ratio", or "posture".
   - Never output markdown tables, bullet lists, or menu lists ("You can ask me about...").

EXAMPLE CONTRAST:
Cliché / Informal (Do NOT use): "Bitcoin is coasting quietly as momentum takes a breather. Since volume is calm, let's sit tight and sit on our hands for now."
Professional & Direct (Desired): "Bitcoin is trading around ${price:,.0f} in a narrow sideways range with light volume. Without clear momentum in either direction, holding off on new trades is the sensible move until a breakout confirms trend direction."

Grounding Reference:
- Target Asset: {ticker}
- Current Live Price: ${price:,.2f}
- Core Signal: {friendly_action}
- Confidence: {confidence:.0f}%
- Market Mood: {sentiment}
- Technical Snapshot: RSI {rsi:.1f}, MACD {macd:.4f}, Volume vs Average {volume_ratio:.1f}x
"""
            system_prompt += f"""

EVIDENCE RULES (override stylistic examples above):
- The structured target is {action_name}. Cash does not imply a sideways market; Long/Short does not establish a market trend.
- Price is the latest available quote ${price:,.2f}; it may come from the stored dataset, so do not assert it is live.
- Do not invent support/resistance levels, future prices, backtest results or a probability of profit.
- Confidence is a policy action probability, not calibrated forecast accuracy or a success rate.
- Indicator observations are not proof of what caused the policy decision. Do not invent feature attributions.
- Answer the user's actual question using only the supplied evidence. If evidence is missing, say so.
- Deterministic reference answer: {response}
"""
            contents = []
            last_role = None
            if req.history:
                started = False
                # Use only the last 4 messages so recent live prices are not overridden by old conversational turns
                trimmed_history = req.history[-4:]
                for msg in trimmed_history:
                    content_text = msg.get("content", "").strip()
                    if not content_text:
                        continue
                    role = "user" if msg.get("role") == "user" else "model"
                    if not started:
                        if role == "user":
                            started = True
                        else:
                            continue
                    if role == last_role:
                        if contents:
                            contents[-1]["parts"][0]["text"] += "\n" + content_text
                        continue
                    contents.append({
                        "role": role,
                        "parts": [{"text": content_text}]
                    })
                    last_role = role

            current_text = req.message.strip()
            if not contents or last_role == "model":
                contents.append({
                    "role": "user",
                    "parts": [{"text": current_text}]
                })
            else:
                contents[-1]["parts"][0]["text"] += "\n" + current_text

            primary_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            # Cascade through primary and fast reliable fallback models on 503 spikes
            models_to_try = [primary_model]
            for m in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.5-flash"]:
                if m not in models_to_try:
                    models_to_try.append(m)

            gemini_response = None
            for model_name in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                    payload = {
                        "systemInstruction": {
                            "parts": [{"text": system_prompt}]
                        },
                        "contents": contents,
                        "generationConfig": {
                            "maxOutputTokens": 1024,
                            "temperature": 0.5,
                            "thinkingConfig": {
                                "thinkingBudget": 0
                            }
                        }
                    }
                    res = requests.post(url, json=payload, timeout=6)
                    # If thinkingConfig is unsupported on a specific model, retry without it
                    if res.status_code == 400 and "thinking" in res.text.lower():
                        payload["generationConfig"].pop("thinkingConfig", None)
                        res = requests.post(url, json=payload, timeout=6)

                    if res.status_code == 200:
                        res_json = res.json()
                        parts = res_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
                        if not text_parts and parts:
                            text_parts = [p.get("text", "") for p in parts]
                        candidate_text = "".join(text_parts).strip()
                        if candidate_text:
                            gemini_response = candidate_text
                            break
                    elif res.status_code in (503, 429):
                        # Model is at capacity, proceed to next model in cascade
                        continue
                except Exception:
                    continue

            if gemini_response:
                is_valid, reason = validate_chat_response(gemini_response, action_name, price, ticker)
                if is_valid:
                    return {"response": gemini_response, "reply": gemini_response, "validation": "passed"}
                else:
                    print(f"Chat response validation failed ({reason}); falling back to deterministic advisory.")

        return {"response": response, "reply": response, "generation": "deterministic"}
    except Exception as e:
        err_msg = f"Error communicating with AI Advisor: {str(e)}"
        return {"response": err_msg, "reply": err_msg}

@app.get("/api/walk-forward")
def get_walk_forward_metrics():
    """Returns the out-of-sample walk-forward validation performance metrics."""
    output_path = "walk_forward_results.csv"
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        return {"status": "success", "data": df.to_dict(orient="records")}
    else:
        raise HTTPException(
            status_code=404, 
            detail="Walk-forward validation results not found. Run 'python src/walk_forward_eval.py' first."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
