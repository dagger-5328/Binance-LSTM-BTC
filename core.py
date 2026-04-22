import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from model import TemporalAttention

# --- PRO-LEVEL GLOBAL CONFIG ---
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
INTERVAL = '1h'
N_TIMESTEPS = 48  # Increased for better temporal context

# Comprehensive feature set (20+ features)
FEATURES = [
    'Price_Z', 'Vol_Z', 'RS_Market',  # Price/volume normalization
    'Hour_Sin', 'Hour_Cos',  # Cyclical time encoding
    'RSI', 'EMA_diff', 'Mom_5', 'Volatility',  # Technical indicators
    'Up_Trend',  # Trend strength
    'Lag_1', 'Lag_2', 'Lag_3',  # Historical lags
    'ret_1', 'ret_2', 'ret_3',  # Lagged returns
    'EMA_diff_10_20',  # EMA 10-20 difference
    'Rolling_Vol_10'  # Rolling volatility (10-period)
]

# --- DATA FETCHING ---
def fetch_data(symbol, limit=2000, months_back_start=None, months_back_end=None):
    """
    Fetch data from Binance with an automatic Fallback to Yahoo Finance 
    if the connection is blocked (Common on US Cloud servers).
    """
    from datetime import datetime, timedelta
    
    # 1. TRY BINANCE FIRST
    endpoints = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
    for base_url in endpoints:
        try:
            if months_back_start and months_back_end:
                now = datetime.utcnow()
                start_ms = int((now - timedelta(days=months_back_start * 30)).timestamp() * 1000)
                end_ms = int((now - timedelta(days=months_back_end * 30)).timestamp() * 1000)
                url = f"{base_url}/api/v3/klines?symbol={symbol}&interval={INTERVAL}&startTime={start_ms}&endTime={end_ms}&limit=1000"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
            else:
                url = f"{base_url}/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit={limit}"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
            
            df = pd.DataFrame(data, columns=['time','Open','High','Low','Close','Volume', 'ct','q','n','tb','tq','i'])
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            df['Open time'] = pd.to_datetime(df['time'], unit='ms')
            return df[['Open time', 'Close', 'Volume']]
        except Exception:
            continue

    # 2. FALLBACK TO YAHOO FINANCE (Reliable on US Cloud Servers)
    try:
        print(f"[*] Binance blocked. Falling back to Yahoo Finance for {symbol}...")
        import yfinance as yf
        # Map Binance symbols to Yahoo symbols (e.g. BTCUSDT -> BTC-USD)
        yf_symbol = symbol.replace("USDT", "-USD")
        yf_df = yf.download(yf_symbol, period="30d", interval="1h", progress=False)
        
        if not yf_df.empty:
            yf_df = yf_df.reset_index()
            yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
            df = pd.DataFrame()
            df['Open time'] = yf_df['Datetime']
            df['Close'] = yf_df['Close'].astype(float)
            df['Volume'] = yf_df['Volume'].astype(float)
            return df.tail(limit)
    except Exception as e:
        print(f"[!] All data sources failed: {e}")
    
    return pd.DataFrame()

def apply_market_features(df, coin_id, market_returns=None, is_training=False):
    """
    Fixed feature engineering pipeline with empty-data safety checks.
    """
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()

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
    
    # 5. Lagged Returns
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_2'] = df['Close'].pct_change(2)
    df['ret_3'] = df['Close'].pct_change(3)
    
    # 6. EMA Difference (10-period vs 20-period)
    df['EMA_diff_10_20'] = (df['Close'].ewm(span=10).mean() - df['Close'].ewm(span=20).mean()) / (df['Close'] + 1e-9)
    
    # 7. Rolling Volatility
    df['Rolling_Vol_10'] = df['Close'].pct_change().rolling(10).std()

    # 8. Historical Lags (Directional)
    df['Lag_1'] = (df['Close'] > df['Close'].shift(1)).shift(1).astype(float)
    df['Lag_2'] = (df['Close'] > df['Close'].shift(4)).shift(1).astype(float)
    df['Lag_3'] = (df['Close'] > df['Close'].shift(24)).shift(1).astype(float)
    
    df['coin_id'] = coin_id

    # 9. Targets (ONLY for training)
    if is_training:
        ret_3d = (df['Close'].shift(-72) - df['Close']) / df['Close']
        ret_7d = (df['Close'].shift(-168) - df['Close']) / df['Close']
        df['Target_3d'] = (ret_3d > 0.005).astype(int)
        df['Target_7d'] = (ret_7d > 0.005).astype(int)
        return df.dropna()
    
    # For inference, only drop rows that don't have enough history for indicators
    return df.dropna(subset=FEATURES)

# --- PREDICTOR ENGINE ---
class ModelEngine:
    def __init__(self, model_path='models/model.keras', scaler_path='models/scaler.pkl', metrics_path='models/metrics.json'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metrics_path = metrics_path
        self.model, self.scaler, self.threshold, self.ready = None, None, 0.5, False
        self.last_error = "No error recorded."
        self._load()

    def _load(self):
        # Use absolute paths to prevent CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(base_dir, 'models', 'model.weights.h5')
        self.scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
        self.metrics_path = os.path.join(base_dir, 'models', 'metrics.json')

        if os.path.exists(self.weights_path) and os.path.exists(self.scaler_path):
            try:
                # 1. Build the model architecture directly from code
                # This bypasses all JSON/Keras serialization bugs
                from model import build_model
                self.model = build_model(input_shape=(N_TIMESTEPS, len(FEATURES)))
                
                # 2. Load the weights into the fresh architecture
                self.model.load_weights(self.weights_path)
                
                self.scaler = joblib.load(self.scaler_path)
                
                if os.path.exists(self.metrics_path):
                    with open(self.metrics_path, 'r') as mf:
                        metrics_data = json.load(mf)
                        self.threshold = metrics_data.get('threshold', 0.5)
                
                self.ready = True
                print(f"[+] Model engine initialized (Code-built Architecture + Weights)")
            except Exception as e:
                self.last_error = str(e)
                print(f"[!] Error loading model artifacts: {e}")
                self.ready = False
        else:
            self.last_error = f"Files missing: {self.weights_path} or {self.scaler_path}"
            print(f"[!] Model artifacts missing")
            self.ready = False

    def predict(self, symbol):
        if not self.ready: return {"error": f"Model initialization failed: {self.last_error}"}
        
        df_raw = fetch_data(symbol, limit=200)
        market_pulse = get_market_pulse(limit=200)
        df = apply_market_features(df_raw, COINS.index(symbol) if symbol in COINS else 0, market_pulse)
        
        if len(df) < N_TIMESTEPS: return {"error": "Insufficient data."}
        
        # Validate all required features exist
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            return {"error": f"Missing features: {missing_features}"}
        
        df_last = df.tail(N_TIMESTEPS).copy()
        
        # Ensure features are in exact order (critical for LSTM)
        df_last = df_last[FEATURES]
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
                "direction": "UP" if prob > self.threshold else "DOWN",
                "confidence": round((prob if prob > self.threshold else (1 - prob)) * 100, 2)
            }
        
        return {
            "symbol": symbol,
            "latest_price": round(float(df['Close'].iloc[-1]), 2),
            "predictions": predictions,
            "indicators": indicators,
            "history": df.tail(72)['Close'].tolist() 
        }




