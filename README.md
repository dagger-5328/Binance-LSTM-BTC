# 💹 Crypto Pulse: Professional Multi-Horizon Analytics Dashboard

A high-performance quantitative framework for cryptocurrency trend forecasting. This project utilizes a Multi-Task LSTM architecture with Temporal Attention to predict market direction across 3-day and 7-day horizons.

---

## 🚀 Features

- **Multi-Task Learning**: Predicts both 3-day and 7-day market direction using a single, unified neural network.
- **Temporal Attention Mechanism**: Custom LSTM layer that learns to focus on the most relevant historical patterns.
- **Institutional-Grade Feature Engineering**: 20+ technical indicators including RSI, Z-scored Price/Volume, EMA Deltas, and Market Relative Strength.
- **Leakage-Free Pipeline**: Strict chronological splitting and walk-forward validation ensure realistic performance metrics.
- **Interactive Dashboard**: Streamlit-based UI for real-time visualization of predictions, technical indicators, and historical price action.
- **REST API**: FastAPI backend for integrating predictions into other trading systems.

---

## 📁 Project Structure

- `app.py`: Streamlit dashboard application.
- `train.py`: Training pipeline for the LSTM model.
- `core.py`: Core logic for data fetching, feature engineering, and prediction engine.
- `model.py`: Neural network architecture and custom layers.
- `api.py`: FastAPI implementation for prediction serving.
- `requirements.txt`: Project dependencies.
- `models/`: Directory for saved model artifacts (excluded from Git).

---

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model
```bash
# Fetch data from Binance and train the multi-horizon model
python train.py
```

### 3. Launching the Dashboard
```bash
# Start the interactive Streamlit UI
streamlit run app.py
```

### 4. Running the API
```bash
# Serve predictions via REST API
uvicorn api.py:app --reload
```

---

## 📊 Model Performance

The model is optimized for **Generalization** over raw accuracy. It uses high dropout (0.4) and L2 regularization to prevent overfitting to market noise.

| Horizon | Prediction Target | Status |
| :--- | :--- | :--- |
| **3-Day** | Directional Up/Down | Optimized for Momentum |
| **7-Day** | Directional Up/Down | Optimized for Trend ID |

---

## ⚠️ Disclaimer
Financial trading involves significant risk. This project is for educational and portfolio demonstration purposes and should not be used for live trading decisions.

---
**Author:** Institutional-grade AI Analytics Suite.
