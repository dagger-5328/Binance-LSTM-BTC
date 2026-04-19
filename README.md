# 💹 Crypto Pulse — Short-Term Crypto Signal Framework

A machine learning project focused on identifying short-term trading signals in cryptocurrency markets.

Instead of predicting simple price direction (which is often misleading due to market trends), this project focuses on detecting changes in momentum using engineered features and realistic validation techniques.

---

## 📊 Snapshot Results
*   **Accuracy:** ~53–55% on a balanced target
*   **Sharpe Ratio:** ~1.5–1.8 *(with realistic transaction costs)*
*   **Max Drawdown:** ~ -6%
*   **Trades:** ~15 high-confidence signals

> **The model prioritizes fewer, higher-quality signals rather than frequent predictions.**

---

## 🧠 Key Ideas
*   Predicting “price up/down” directly can be biased due to overall market trends.
*   Reframing the problem improves reliability.
*   Price data alone is noisy — additional signals can help.
*   Real-world factors like fees and slippage significantly affect performance.

---

## ⚙️ Methodology

### 1. Problem Framing
Instead of predicting direction, the model predicts whether:
*   **Future short-term returns are stronger than recent past returns.**

This helps reduce bias from long-term trends and creates a more balanced learning problem.

### 2. Feature Engineering
The model uses a mix of price-based and trading activity signals:
*   Short-term returns and volatility measures
*   Time-based patterns (cyclical features)
*   **Trading activity indicators**, such as:
    *   Aggressive buying vs. selling pressure
    *   Average trade size (to capture larger market moves)

### 3. Model
*   **LSTM-based sequence model** with an attention layer.
*   Looks at a rolling **48-hour window**.
*   Outputs probabilities for short-term signal strength.

*The model is used as a tool for capturing temporal patterns — the core focus of the project is on problem design and validation.*

---

## 🧪 Validation & Robustness
A key focus of this project is ensuring results are realistic and not overfit.

### Walk-Forward Validation
*   Model is tested on completely unseen future data.
*   Avoids random splits and temporal data leakage.

### Transaction Costs
Performance is tested under different friction scenarios:
1.  **No costs** (baseline)
2.  **Realistic costs** (~0.3% per round-trip trade)
3.  **High-stress costs** (~0.5% per round-trip trade)

> **Result:** The strategy remains profitable even under elevated transaction costs, indicating the signal is not purely an artifact of overtrading.

### Feature Importance
*   Permutation-based ablation method used to evaluate feature impact.
*   Helps identify which inputs contribute most to predictions.
*(Note: Permutation importance was used for efficient ablation, though it can occasionally underestimate importance when features are correlated).*

### Market Conditions
*   Performance is analyzed across different market regimes (Bull, Bear, Sideways).
*   The model performs best in sideways or choppy markets, where simple naive strategies typically struggle.

---

## 📈 Results Summary
*   Consistently outperforms a random baseline on a rigorously balanced target.
*   Maintains positive risk-adjusted returns after transaction costs are applied.
*   Shows strictly controlled drawdowns compared to naive Buy & Hold strategies.

*While the quantitative edge is modest, it is explicitly designed to be structurally realistic and robust rather than over-optimized.*

---

## 💡 Takeaways
This project highlights that:
1.  **Problem formulation** is as important as model architecture choice.
2.  Small, mathematically consistent edges matter more than high raw (biased) accuracy.
3.  Validation under **realistic constraints** is critical in financial ML.

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the model**
```bash
python train.py
```

**3. Run backtest**
```bash
python backtest.py
```

**4. Launch dashboard**
```bash
streamlit run app.py
```

---
> **⚠️ Disclaimer**
> *This project is for educational and portfolio demonstration purposes only. Cryptocurrency markets are highly volatile and unpredictable. This is NOT financial advice.*