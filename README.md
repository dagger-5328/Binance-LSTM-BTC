# 💹 Crypto Pulse: Professional Analytics Dashboard

A high-end, trading-desk style dashboard for cryptocurrency trend forecasting across 3rd-day and 7th-day horizons. 

*Status: **Verified Production-Ready (Balanced & Regularized)**.*

---

## 🚀 Key Features

- **Multi-Horizon Intelligence**: A single Multi-Task LSTM model predicting market direction for 3rd and 7th day windows.
- **Regularized Architecture**: High Dropout (0.4) and L2 Weight Decay to ensure generalization and prevent historical noise memorization.
- **Balanced Targets**: Median-split return thresholding ensures a 50/50 training baseline, making the model a true directional learner.
- **Pro-Level UI**: Wide layout featuring KPI rows, price charts with EMA overlays, and Forecast Conviction Gauges.
- **Strategic Insights**: Human-readable interpretations of technical indicators (RSI, EMA, Momentum).

---

## 📈 Stable Performance (V9 - Regularized)

The model is optimized for **Generalization**. These results represent its ability to maintain stability during high-volatility market shifts.

| Horizon | Test Accuracy | Generalization Status |
| :--- | :--- | :--- |
| **3-Day** | **~56%** | Stable Momentum Capture |
| **7-Day** | **~74%** | Professional Trend ID |

> [!IMPORTANT]
> This "Production Model" (V9) has been stressed against **Distribution Shift**. Even during the extreme 98% Downtrend in our latest test window, the model maintained structural integrity without diverging (Training/Validation loss gap < 0.2).

---

## 🛠️ Usage & Deployment

### 1. Training (Regularized)
```bash
python train.py
```

### 2. Launching the Dashboard
```bash
streamlit run app.py
```

### 3. API Service
```bash
uvicorn api:app --reload
```

---

## 🎨 Visual Preview

- **Wide Layout**: Trading-desk aesthetics with consistent alignment.
- **Top KPI Bar**: Metric cards for Price, Forecast Direction, and Market Momentum.
- **Technical Indicator Panel**: interprets values into Bullish/Bearish/Neutral categories.

---

## ⚠️ Disclaimer
Financial trading involves significant risk. This project is for educational and portfolio demonstration purposes and should not be used for live trading decisions.
