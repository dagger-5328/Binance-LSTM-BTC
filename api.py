"""
Professional Crypto Intelligence API
--------------------------------------
FastAPI backend serving predictions and model performance data.
"""

import json, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from core import ModelEngine
from config import COINS, METRICS_PATH, BACKTEST_PATH

app = FastAPI(
    title="Crypto Pulse API",
    description="Walk-forward validated LSTM directional forecasting.",
    version="5.0.0"
)

engine = ModelEngine()


class HorizonPrediction(BaseModel):
    direction: str
    confidence: float

class IndicatorInfo(BaseModel):
    value: float
    label: str

class IndicatorObject(BaseModel):
    rsi: IndicatorInfo
    ema_diff: IndicatorInfo
    momentum: float
    up_trend: float

class CryptoResponse(BaseModel):
    coin: str
    latest_price: float
    predictions: Dict[str, HorizonPrediction]
    indicators: IndicatorObject


@app.get("/")
def health():
    return {
        "status": "operational",
        "model_loaded": engine.ready,
        "assets": COINS
    }


@app.get("/predict/{symbol}", response_model=CryptoResponse)
def get_prediction(symbol: str):
    symbol = symbol.upper()
    if symbol not in COINS:
        raise HTTPException(400, f"Unsupported. Use: {COINS}")
    if not engine.ready:
        raise HTTPException(503, "Model not trained. Run: python train.py")

    res = engine.predict(symbol)
    if "error" in res:
        raise HTTPException(400, res["error"])

    return {
        "coin": symbol,
        "latest_price": res["latest_price"],
        "predictions": res["predictions"],
        "indicators": res["indicators"]
    }


@app.get("/metrics")
def get_metrics():
    """Return walk-forward validation results."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(404, "No metrics. Run train.py first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/backtest")
def get_backtest():
    """Return financial backtest results."""
    if not os.path.exists(BACKTEST_PATH):
        raise HTTPException(404, "No backtest. Run backtest.py first.")
    with open(BACKTEST_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
