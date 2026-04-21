#!/usr/bin/env python3
"""
cache_data.py — Pre-fetch and cache Binance data for deployment
-----------------------------------------------------------------
This script fetches historical data for all configured coins and caches it
locally. Run this before building the Docker image in environments where
Binance API access is restricted.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import COINS
from features import fetch_data

def main():
    print("Pre-fetching and caching data for deployment...")
    for symbol in COINS:
        print(f"Fetching data for {symbol}...")
        try:
            df = fetch_data(symbol, total_candles=4000)
            if not df.empty:
                print(f"  ✓ Cached {len(df)} candles for {symbol}")
            else:
                print(f"  ✗ Failed to fetch data for {symbol}")
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")

    print("Data caching complete.")

if __name__ == "__main__":
    main()