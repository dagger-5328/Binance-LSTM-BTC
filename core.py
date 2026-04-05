import os
import requests
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --- PRO-LEVEL GLOBAL CONFIG ---
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
INTERVAL = '1h'
N_TIMESTEPS = 24 

# Features focus on Local Momentum + Global Market Context
FEATURES = [
    'Price_Z', 'Vol_Z', 'RSI', 'EMA_diff', 'Mom_5', 'Volatility',
    'RS_Market', 'Hour_Sin', 'Hour_Cos', 'Up_Trend', 'Lag_1', 'Lag_2', 'Lag_3', 'coin_id'
]

# --- DATA FETCHING ---
def fetch_data(symbol, limit=2000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=['time','Open','High','Low','Close','Volume', 'ct','q','n','tb','tq','i'])
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df['Open time'] = pd.to_datetime(df['time'], unit='ms')
        return df[['Open time', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

# --- MARKET CONTEXT ENGINE ---
def get_market_pulse(limit=100):
    """Calculates the average global return across our 4-coin basket."""
    returns = []
    for coin in COINS:
        df = fetch_data(coin, limit=limit)
        if not df.empty:
            returns.append(df['Close'].pct_change())
    
    if returns:
        return pd.concat(returns, axis=1).mean(axis=1).fillna(0)
    return pd.Series([0] * limit)

def apply_market_features(df, coin_id, market_returns=None):
    """
    Fixed feature engineering pipeline to remove Look-ahead Bias.
    """
    df = df.copy().reset_index(drop=True)
    
    # 1. Price/Vol Z-Scores
    df['Price_Z'] = (df['Close'] - df['Close'].rolling(24).mean()) / (df['Close'].rolling(24).std() + 1e-9)
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(24).mean()) / (df['Volume'].rolling(24).std() + 1e-9)

    # 2. Market Sentiment
    if market_returns is not None:
        coin_return = df['Close'].pct_change()
        m_ret = market_returns.iloc[-len(df):].reset_index(drop=True)
        df['RS_Market'] = coin_return - m_ret
    else:
        df['RS_Market'] = 0

    # 3. Cyclical Time
    hours = df['Open time'].dt.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * hours / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * hours / 24)

    # 4. Standard Technicals
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['EMA_diff'] = (df['Close'].ewm(span=5).mean() - df['Close'].ewm(span=10).mean()) / (df['Close'] + 1e-9)
    df['Mom_5'] = df['Close'].pct_change(5)
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    
    # Trend Indicator
    df['Up_Trend'] = (df['Close'].diff() > 0).rolling(5).mean()

    # 5. Multi-Horizon Targets (FIXED - Balanced)

    ret_3d = (df['Close'].shift(-72) - df['Close']) / df['Close']
    ret_7d = (df['Close'].shift(-168) - df['Close']) / df['Close']

    # Remove NaNs temporarily for threshold calculation
    valid_3d = ret_3d.dropna()
    valid_7d = ret_7d.dropna()

    # Median split → guarantees ~50/50 balance
    thresh_3d = valid_3d.median()
    thresh_7d = valid_7d.median()

    df['Target_3d'] = (ret_3d > thresh_3d).astype(int)
    df['Target_7d'] = (ret_7d > thresh_7d).astype(int)

    # 6. HISTORICAL LAGS
    df['Lag_1'] = (df['Close'] > df['Close'].shift(1)).shift(1).astype(float)
    df['Lag_2'] = (df['Close'] > df['Close'].shift(4)).shift(1).astype(float)
    df['Lag_3'] = (df['Close'] > df['Close'].shift(24)).shift(1).astype(float)
    
    df['coin_id'] = coin_id
    
    return df.dropna()

# --- PREDICTOR ENGINE ---
class ModelEngine:
    def __init__(self, model_path='models/model.h5', scaler_path='models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model, self.scaler, self.ready = None, None, False
        self._load()

    def _load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.ready = True

    def predict(self, symbol):
        if not self.ready: return {"error": "Model not trained."}
        
        df_raw = fetch_data(symbol, limit=200)
        market_pulse = get_market_pulse(limit=200)
        df = apply_market_features(df_raw, COINS.index(symbol) if symbol in COINS else 0, market_pulse)
        
        if len(df) < N_TIMESTEPS: return {"error": "Insufficient data."}
        
        df_last = df.tail(N_TIMESTEPS).copy()
        df_last[FEATURES] = self.scaler.transform(df_last[FEATURES])
        X = df_last[FEATURES].values.reshape(1, N_TIMESTEPS, len(FEATURES))
        
        # Multi-output prediction [acc_3d, acc_7d]
        raw_pred = self.model.predict(X, verbose=0)[0]
        
        rsi_val = float(df['RSI'].iloc[-1])
        ema_val = float(df['EMA_diff'].iloc[-1])
        trend_val = float(df['Up_Trend'].iloc[-1])
        
        indicators = {
            "rsi": {
                "value": round(rsi_val, 2),
                "label": "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            },
            "ema_diff": {
                "value": round(ema_val, 4),
                "label": "Bullish" if ema_val > 0 else "Bearish"
            },
            "momentum": round(float(df['Mom_5'].iloc[-1]) * 100, 2),
            "up_trend": round(trend_val * 100, 2)
        }
        
        horizons = ["3d", "7d"]
        predictions = {}
        for idx, horizon in enumerate(horizons):
            prob = float(raw_pred[idx])
            predictions[horizon] = {
                "direction": "UP" if prob > 0.50 else "DOWN",
                "confidence": round((prob if prob > 0.50 else (1 - prob)) * 100, 2)
            }
        
        return {
            "symbol": symbol,
            "latest_price": round(float(df['Close'].iloc[-1]), 2),
            "predictions": predictions,
            "indicators": indicators,
            "history": df.tail(72)['Close'].tolist() 
        }