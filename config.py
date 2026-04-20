"""
config.py — Institutional Quant Configuration (V3)
---------------------------------------
Key changes:
  - Microstructure features: Taker Buy Ratio, Trade Density
  - Target: Relative outperformance vs rolling median (forces strict 50% baseline)
"""

# --- Assets ---
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
INTERVAL = '1h'

# --- Features ---
N_TIMESTEPS = 48  # 2 days of hourly data (was 24)

FEATURES = [
    'Ret_4h', 'Ret_12h', 'Ret_24h',     # Multi-scale returns
    'Price_Z', 'Vol_Z', 'Vol_Ratio',     # Normalized price/volume
    'Taker_Ratio', 'Trade_Density',      # [NEW] Microstructure (Aggressor & Size)
    'RSI', 'EMA_diff', 'MACD',           # Momentum indicators
    'ATR', 'BB_pos',                      # OHLC-derived volatility/position
    'Up_Trend',                          # Trend strength
    'Lag_1', 'Lag_2', 'Lag_3',           # Historical direction lags
    'Hour_Sin', 'Hour_Cos',              # Cyclical time encoding
]

# Prediction horizons (hours ahead)
HORIZONS = {'3d': 72, '7d': 168}
TARGET_COLS = ['Target_3d', 'Target_7d']

# --- Model Architecture ---
LSTM_UNITS = [32]
DENSE_UNITS = 16
DROPOUT = 0.4
L2_REG = 0.002

# --- Training ---
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0008  # Slightly lower for stability with attention
PATIENCE_STOP = 20
PATIENCE_LR = 8

# --- Walk-Forward ---
N_FOLDS = 3
PURGE_GAP = 192  # 8 days

# --- Paths ---
MODEL_PATH = 'models/model.h5'
SCALER_PATH = 'models/scaler.pkl'
METRICS_PATH = 'models/metrics.json'
BACKTEST_PATH = 'models/backtest.json'
CACHE_DIR = 'data/cache'

# --- Friction Scenarios (Round-trip penalties) ---
FEE_SCENARIOS = {
    'No Friction': 0.0,
    'Realistic': 0.003,      # 0.1% entry fee + 0.1% exit fee + 0.05% slippage each way
    'High Stress': 0.005     # 0.5% total round trip
}
