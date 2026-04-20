"""
core.py — Prediction Engine (V2)
----------------------------------
Loads the trained model, scaler, and optimized thresholds.
Provides predictions for the live dashboard.
"""

import os, json
import joblib
from tensorflow.keras.models import load_model

from config import FEATURES, HORIZONS, N_TIMESTEPS, MODEL_PATH, SCALER_PATH, METRICS_PATH
from features import fetch_data, engineer_features
from model import TemporalAttention


class ModelEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.thresholds = {}
        self.ready = False
        self.last_error = None
        self._load()

    def _load(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            self.last_error = "Model artifacts not found. Run: python train.py"
            return

        try:
            custom_objects = {'TemporalAttention': TemporalAttention}
            self.model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            self.scaler = joblib.load(SCALER_PATH)

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, 'r') as f:
                    metrics = json.load(f)
                    self.thresholds = metrics.get('thresholds', {})

            self.ready = True
            self.last_error = None
        except Exception as exc:
            self.model = None
            self.scaler = None
            self.ready = False
            self.last_error = f"Model load failed: {exc}"

    def predict(self, symbol):
        """Run prediction for a coin."""
        if not self.ready:
            return {"error": self.last_error or "Model not trained. Run: python train.py"}

        try:
            raw = fetch_data(symbol, total_candles=500)
            if raw.empty:
                return {"error": f"Could not fetch data for {symbol}"}

            df = engineer_features(raw)

            if len(df) < N_TIMESTEPS:
                return {"error": "Insufficient data after feature engineering."}

            df_last = df.tail(N_TIMESTEPS).copy()
            df_last[FEATURES] = self.scaler.transform(df_last[FEATURES])
            X = df_last[FEATURES].values.reshape(1, N_TIMESTEPS, len(FEATURES))

            raw_pred = self.model.predict(X, verbose=0)[0]

            rsi_val = float(df['RSI'].iloc[-1])
            ema_val = float(df['EMA_diff'].iloc[-1])
            trend_val = float(df['Up_Trend'].iloc[-1])
            bb_pos = float(df['BB_pos'].iloc[-1])

            indicators = {
                "rsi": {
                    "value": round(rsi_val, 2),
                    "label": "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                },
                "ema_diff": {
                    "value": round(ema_val, 4),
                    "label": "Bullish" if ema_val > 0 else "Bearish" if ema_val < 0 else "Flat"
                },
                "bb_pos": {
                    "value": round(bb_pos, 2),
                    "label": "High" if bb_pos > 1 else "Low" if bb_pos < -1 else "Mid"
                },
                "momentum": round(float(df['Ret_24h'].iloc[-1]) * 100, 2),
                "up_trend": round(trend_val * 100, 2)
            }

            horizons = list(HORIZONS.keys())
            predictions = {}
            for idx, h in enumerate(horizons):
                prob = float(raw_pred[idx])
                thresh = self.thresholds.get(h, 0.5)
                direction = "UP" if prob > thresh else "DOWN"

                if prob > thresh:
                    conf = 0.5 + 0.5 * ((prob - thresh) / (1.0 - thresh))
                else:
                    conf = 0.5 + 0.5 * ((thresh - prob) / thresh)

                predictions[h] = {
                    "direction": direction,
                    "confidence": round(conf * 100, 2),
                    "raw_prob": round(prob * 100, 2),
                    "threshold": round(thresh * 100, 2)
                }

            return {
                "symbol": symbol,
                "latest_price": round(float(df['Close'].iloc[-1]), 2),
                "predictions": predictions,
                "indicators": indicators,
                "history": df.tail(72)['Close'].tolist()
            }
        except Exception as exc:
            return {"error": f"Prediction failed for {symbol}: {exc}"}
