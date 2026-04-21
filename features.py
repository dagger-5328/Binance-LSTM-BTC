"""
features.py — Leak-Free Feature Engineering (V2)
--------------------------------------------------
V2 changes: Uses OHLC data, adds ATR, Bollinger position,
multi-scale returns, volume ratio. Drops coin_id.
"""
import os
import time

import numpy as np
import pandas as pd
import requests
from config import INTERVAL, HORIZONS, CACHE_DIR

BINANCE_BASE_URLS = [
    url.strip()
    for url in os.getenv(
        "BINANCE_BASE_URLS",
        "https://api.binance.com,https://api1.binance.com,https://api3.binance.com",
    ).split(",")
    if url.strip()
]
REQUEST_HEADERS = {"User-Agent": "crypto-pulse/1.0"}

# Proxy configuration for geo-blocked deployments
PROXIES = None
proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
if proxy_url:
    PROXIES = {"http": proxy_url, "https": proxy_url}

def fetch_data(symbol, total_candles=4000):
    """Fetch hourly OHLCV from Binance with pagination + caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{total_candles}_ohlcv.csv")

    allow_stale_cache = os.getenv("ALLOW_STALE_CACHE", "false").lower() == "true"

    if os.path.exists(cache_file):
        age_hrs = (time.time() - os.path.getmtime(cache_file)) / 3600
        # Use cached data if fresh (<12h) OR if Binance is likely blocked (older cache is better than no data)
        # OR if ALLOW_STALE_CACHE is enabled
        if age_hrs < 12 or allow_stale_cache:
            return pd.read_csv(cache_file, parse_dates=['Open time'])
        elif age_hrs < 168:  # 7 days
            print(f"  Using stale cache for {symbol} (age: {age_hrs:.1f}h, Binance likely blocked)")
            return pd.read_csv(cache_file, parse_dates=['Open time'])

    all_rows, end_time, remaining = [], None, total_candles
    last_error = None
    while remaining > 0:
        batch = min(1000, remaining)
        data = None

        for base_url in BINANCE_BASE_URLS:
            url = f"{base_url}/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit={batch}"
            if end_time:
                url += f"&endTime={end_time}"

            try:
                resp = requests.get(url, timeout=15, headers=REQUEST_HEADERS, proxies=PROXIES)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    raise RuntimeError(data.get("msg", f"Unexpected response from {base_url}"))
                break
            except Exception as exc:
                last_error = f"{base_url}: {exc}"

        if data is None:
            print(f"  Warning: API error for {symbol}: {last_error}")
            break
        if not data:
            break
        all_rows = data + all_rows
        end_time = data[0][0] - 1
        remaining -= len(data)
        if len(data) < batch:
            break

    if not all_rows:
        # Check if we have cached data and the error is geo-blocking (451)
        print(f"DEBUG: last_error='{last_error}', cache_exists={os.path.exists(cache_file)}")
        if os.path.exists(cache_file) and last_error and "451" in last_error:
            print(f"  Using cached data for {symbol} (Binance geo-blocked)")
            return pd.read_csv(cache_file, parse_dates=['Open time'])
        
        empty_df = pd.DataFrame()
        if last_error:
            empty_df.attrs["fetch_error"] = f"Binance request failed for {symbol}: {last_error}"
        return empty_df

    seen = set()
    unique = [r for r in all_rows if r[0] not in seen and not seen.add(r[0])]
    df = pd.DataFrame(unique, columns=[
        'time','Open','High','Low','Close','Volume','ct','q','n','tb','tq','i'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'n', 'tb']:
        df[col] = df[col].astype(float)
    df['Open time'] = pd.to_datetime(df['time'], unit='ms')
    df.rename(columns={'n': 'Num_Trades', 'tb': 'Taker_Base_Vol'}, inplace=True)
    result = df[['Open time','Open','High','Low','Close','Volume','Num_Trades','Taker_Base_Vol']].reset_index(drop=True)
    result.to_csv(cache_file, index=False)
    return result

def engineer_features(df):
    """Build all features from OHLCV data. Every feature is backward-looking only."""
    df = df.copy().reset_index(drop=True)

    # Microstructure Features
    df['Trade_Density'] = df['Volume'] / (df['Num_Trades'] + 1e-9)
    # Scale Trade_Density to make it neural-net friendly
    df['Trade_Density'] = (df['Trade_Density'] - df['Trade_Density'].rolling(72).mean()) / (df['Trade_Density'].rolling(72).std() + 1e-9)
    df['Taker_Ratio'] = df['Taker_Base_Vol'] / (df['Volume'] + 1e-9)

    # Multi-scale returns
    df['Ret_4h'] = df['Close'].pct_change(4)
    df['Ret_12h'] = df['Close'].pct_change(12)
    df['Ret_24h'] = df['Close'].pct_change(24)

    # Normalized price & volume (24h)
    df['Price_Z'] = (df['Close'] - df['Close'].rolling(24).mean()) / (df['Close'].rolling(24).std() + 1e-9)
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(24).mean()) / (df['Volume'].rolling(24).std() + 1e-9)
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(24).mean() + 1e-9)

    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # EMA crossover
    df['EMA_diff'] = (df['Close'].ewm(span=5).mean() - df['Close'].ewm(span=10).mean()) / (df['Close'] + 1e-9)

    # MACD
    df['MACD'] = (df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()) / (df['Close'] + 1e-9)

    # ATR — Average True Range (needs High/Low/Close)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean() / (df['Close'] + 1e-9)

    # Bollinger Band position (-1 to +1 range)
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_pos'] = (df['Close'] - sma20) / (2 * std20 + 1e-9)

    # Trend
    df['Up_Trend'] = (df['Close'].diff() > 0).rolling(5).mean()

    # Direction lags (shifted by 1)
    df['Lag_1'] = (df['Close'] > df['Close'].shift(1)).shift(1).astype(float)
    df['Lag_2'] = (df['Close'] > df['Close'].shift(4)).shift(1).astype(float)
    df['Lag_3'] = (df['Close'] > df['Close'].shift(24)).shift(1).astype(float)

    # Cyclical time
    hours = df['Open time'].dt.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * hours / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * hours / 24)

    # TARGETS — Outperformance vs Rolling Median (Strictly balanced)
    for label, shift_hrs in HORIZONS.items():
        fut = (df['Close'].shift(-shift_hrs) - df['Close']) / df['Close']
        realized = (df['Close'] - df['Close'].shift(shift_hrs)) / df['Close'].shift(shift_hrs)
        # Target: Will the upcoming return be higher than the immediately preceding return? (Acceleration)
        df[f'Target_{label}'] = (fut > realized).astype(float)

    return df.dropna().reset_index(drop=True)

def create_sequences(df, n_timesteps, features, target_cols):
    """Convert tabular data into LSTM-ready 3D arrays."""
    data = df[features].values
    targets = df[target_cols].values
    X, y = [], []
    for i in range(len(data) - n_timesteps):
        X.append(data[i : i + n_timesteps])
        y.append(targets[i + n_timesteps - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
