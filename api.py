"""
Professional Crypto Momentum API (V4)
--------------------------------------
Lightweight FastAPI backend serving Multi-Horizon (3d, 7d) predictions 
and detailed relative strength indicators.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from core import ModelEngine, COINS

app = FastAPI(
    title="Crypto Weekly Intelligence API",
    description="Multi-Horizon LSTM Trend Forecasting for cryptocurrency assets.",
    version="4.0.0"
)

# Shared Predictor Engine
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
        "assets_tracked": COINS
    }

@app.get("/predict/{symbol}", response_model=CryptoResponse)
def get_prediction(symbol: str):
    """
    Returns high-conviction 3-day and 7-day direction forecasts 
    for the selected ticker symbol.
    """
    symbol = symbol.upper()
    if symbol not in COINS:
        raise HTTPException(status_code=400, detail=f"Unsupported asset. Supported: {COINS}")
    
    if not engine.ready:
        raise HTTPException(status_code=533, detail="Predictor engine initialization pending. Check train.py status.")

    res = engine.predict(symbol)
    
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
        
    return {
        "coin": symbol,
        "latest_price": res["latest_price"],
        "predictions": res["predictions"],
        "indicators": res["indicators"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
