"""
💹 Crypto Pulse: Professional Weekly Analytics Dashboard (V4)
------------------------------------------------------------
A high-end, trading-desk style dashboard for cryptocurrency trend forecasting 
across 3-day and 7-day horizons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Core logic imports (Unified Version)
from core import ModelEngine, COINS

# Initialization
engine = ModelEngine()

# Load optimized threshold from metrics.json
threshold = 0.5  # default fallback
metrics_path = os.path.join("models", "metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r') as mf:
            metrics_data = json.load(mf)
            threshold = metrics_data.get('threshold', 0.5)
    except Exception as e:
        st.warning(f"Could not load metrics.json: {e}")

# Verify engine is using same features as training
if engine.ready:
    from core import FEATURES, N_TIMESTEPS
    st.session_state.features_info = {
        "num_features": len(FEATURES),
        "timesteps": N_TIMESTEPS,
        "feature_list": FEATURES
    }

# --- Page Config & Theme ---
st.set_page_config(
    page_title="Crypto Pulse Analytics Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #238636; color: white; border: none; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { background-color: #2ea043; }
    h1, h2, h3 { color: #58a6ff; font-weight: 700; }
    .prediction-card { padding: 30px; background-color: #1c2128; border-radius: 15px; border-left: 10px solid #58a6ff; }
    .indicator-card { padding: 15px; background-color: #161b22; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 10px; }
    .up-text { color: #3fb950; font-weight: bold; font-size: 1.2rem; }
    .down-text { color: #f85149; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- Top Header Bar ---
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("💹 Crypto Pulse: Weekly Trend Analytics")
    st.markdown("AI-powered 3-day and 7-day directional forecasting with relative strength intelligence.")

with header_col2:
    status_label = "✅ Model Loaded" if engine.ready else "⏳ Engine Initializing..."
    st.info(status_label)

st.write("---")

# --- Top KPI Row (Price, Direction, Conf, Market Trend) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Placeholder results until action
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# --- Asset Selection Sidebar ---
with st.sidebar:
    st.header("⚙️ Strategy Controls")
    symbol = st.selectbox("Select Asset Pair", COINS, index=0)
    # Horizon selector now supports 3/7 as requested
    horizon_sel = st.selectbox("Forecast Horizon", ["3 Days", "7 Days"], index=1)
    
    st.markdown("---")
    st.subheader("📊 Market Parameters")
    st.caption("Interval: 1-hour Candlesticks")
    st.caption("Architecture: Multi-Horizon Stacked LSTM (V4)")
    st.caption(f"Decision Threshold: {threshold:.2f}")
    
    st.markdown("---")
    st.subheader("Model Limitations")
    st.caption("- Predictions are probabilistic forecasts.")
    st.caption("- Markets contain high stochastic noise.")
    st.caption("- NOT financial advice.")
    
    st.markdown("---")
    st.subheader("🔧 Debug Info")
    if 'features_info' in st.session_state:
        info = st.session_state.features_info
        st.caption(f"Features: {info['num_features']} | Timesteps: {info['timesteps']}")
        with st.expander("View feature list"):
            st.code(str(info['feature_list']), language='python')
    
    run_button = st.button("🚀 Run Analysis", use_container_width=True)

# Mapping
horizon_map = {"3 Days": "3d", "7 Days": "7d"}
horizon_key = horizon_map[horizon_sel]

# Trigger Analytics
if run_button or st.session_state.prediction_data is None:
    with st.spinner(f"Decoding market patterns for {symbol}..."):
        st.session_state.prediction_data = engine.predict(symbol)

# --- Layout Rendering ---
res = st.session_state.prediction_data

if res and "error" not in res:
    p_info = res["predictions"][horizon_key]
    dir_val = p_info["direction"]
    conf_val = p_info["confidence"]
    
    kpi1.metric("Current Price", f"${res['latest_price']:,.2f}")
    kpi2.metric("Target Horizon", horizon_sel)
    
    color = "normal" if dir_val == "UP" else "inverse"
    kpi3.metric("Trend Forecast", dir_val, delta=f"{conf_val:.1f}% Conf.", delta_color=color)
    
    m_trend = "Bullish" if res["indicators"]["ema_diff"]["value"] > 0 else "Bearish"
    kpi4.metric("Consensus Sentiment", m_trend)

    st.write("---")

    col_left, col_right = st.columns([2.5, 1.2])

    with col_left:
        st.subheader(f"📈 Historical Index: {symbol}")
        chart_data = pd.DataFrame(res["history"], columns=["Price"])
        chart_data["EMA_Fast"] = chart_data["Price"].ewm(span=12).mean()
        chart_data["EMA_Slow"] = chart_data["Price"].ewm(span=26).mean()
        
        st.line_chart(chart_data, color=["#58a6ff", "#3fb950", "#f85149"], height=400)
        st.caption("Historical price action with exponential moving average (12/26) cross-analysis.")

        st.write("---")
        st.subheader("💡 Strategic Insight")
        
        rsi_label = res["indicators"]["rsi"]["label"]
        ema_label = res["indicators"]["ema_diff"]["label"]
        
        insight_text = ""
        if dir_val == "UP":
            insight_text = f"The model detects a **bullish continuation** for the next {horizon_sel}. "
            if rsi_label == "Oversold":
                insight_text += "Extremely oversold conditions suggest a high-conviction recovery."
            elif ema_label == "Bullish":
                insight_text += "EMA momentum is aligned with the upward forecast."
        else:
            insight_text = f"Indicators point to a **bearish correction** over the {horizon_sel} window. "
            if rsi_label == "Overbought":
                insight_text += "Overbought signals indicate a potential immediate rejection."
            elif ema_label == "Bearish":
                insight_text += "Trend divergence confirms an unfavorable structural outlook."

        st.success(f"**What this means:** {insight_text}")

    with col_right:
        st.subheader("🎯 Model Precision")
        
        style_class = "up-text" if dir_val == "UP" else "down-text"
        st.markdown(f"""
        <div class="prediction-card">
            <h4>Next {horizon_sel} Outlook</h4>
            <span class="{style_class}">{dir_val}</span>
            <p style="margin-top:10px;">The LSTM v4 engine analyzes 12+ technical indicators and 4 major assets to evaluate this signal.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.write("**Forecast Conviction Gauge**")
        st.progress(conf_val / 100)
        
        st.write("---")
        st.subheader("🛠️ Technical Metrics")
        
        metrics = [
            ("RSI (Relative Strength)", res["indicators"]["rsi"]["value"], res["indicators"]["rsi"]["label"]),
            ("EMA Delta Pulse", res["indicators"]["ema_diff"]["value"], res["indicators"]["ema_diff"]["label"]),
            ("Momentum Velocity", res["indicators"]["momentum"], f"{res['indicators']['momentum']:.2f}%"),
            ("Structural Up_Trend", res["indicators"]["up_trend"], f"{res['indicators']['up_trend']:.1f}%")
        ]
        
        for name, val, label in metrics:
            with st.container():
                st.markdown(f"""
                <div class="indicator-card">
                    <span style="font-size:0.8rem; color:#8b949e;">{name}</span><br/>
                    <span style="font-size:1.1rem; font-weight:bold;">{val}</span> 
                    <span style="font-size:0.8rem; margin-left:10px;">({label})</span>
                </div>
                """, unsafe_allow_html=True)

else:
    if res and "error" in res:
        st.error(res["error"])
        st.warning("Ensure the V4 model has finished training.")
    else:
        st.info("Select an asset pair and update analytics to start.")
