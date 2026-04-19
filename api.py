"""
Professional Crypto Intelligence API
--------------------------------------
FastAPI backend serving predictions and model performance data.
"""

import json, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HorizonPrediction(BaseModel):
    direction: str
    confidence: float

class IndicatorInfo(BaseModel):
    value: float
    label: str

class IndicatorObject(BaseModel):
    rsi: IndicatorInfo
    ema_diff: IndicatorInfo
    bb_pos: IndicatorInfo
    momentum: float
    up_trend: float

class CryptoResponse(BaseModel):
    coin: str
    latest_price: float
    predictions: Dict[str, HorizonPrediction]
    indicators: IndicatorObject
    history: list[float]


@app.get("/")
def health():
    return {
        "status": "operational",
        "model_loaded": engine.ready,
        "assets": COINS,
        "error": engine.last_error
    }


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "api"
    }


@app.get("/readyz")
def readyz():
    if not engine.ready:
        raise HTTPException(503, engine.last_error or "Model not ready")
    return {
        "status": "ready",
        "service": "api",
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
        "indicators": res["indicators"],
        "history": res["history"]
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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
