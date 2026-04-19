"""
💹 Crypto Pulse — Professional Analytics Dashboard (V2)
-------------------------------------------------------
Tabs: Live Predictions | Model Performance | Backtest Results
Uses custom Threshold-based Signal Generation.
"""

import json, os
import streamlit as st
import pandas as pd
import numpy as np

from core import ModelEngine
from config import COINS, METRICS_PATH, BACKTEST_PATH

engine = ModelEngine()

st.set_page_config(
    page_title="Crypto Pulse Analytics V2",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #238636; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #2ea043; }
    h1, h2, h3 { color: #58a6ff; font-weight: 700; }
    .up-text { color: #3fb950; font-weight: bold; font-size: 1.2rem; }
    .down-text { color: #f85149; font-weight: bold; font-size: 1.2rem; }
    .metric-card { padding: 15px; background: #161b22; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 10px; }
    .report-box { padding: 20px; background: #1c2128; border-radius: 12px; border-left: 4px solid #58a6ff; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

h1, h2 = st.columns([4, 1])
with h1:
    st.title("💹 Crypto Pulse Analytics (V3)")
    st.caption("LSTM Attention + LayerNorm | Relative Outperformance Classifier")
with h2:
    st.info("✅ Engine Online" if engine.ready else "⏳ Train model first")

tab1, tab2, tab3 = st.tabs(["🔮 Live Tracker", "📊 Edge Detection", "💰 Portfolio Backtest"])

# ==================== TAB 1: LIVE ====================
with tab1:
    st.write("---")
    with st.sidebar:
        st.header("⚙️ Market Controls")
        symbol = st.selectbox("Asset Pair", COINS, index=0)
        horizon_sel = st.selectbox("Forecast Horizon", ["3 Days", "7 Days"], index=1)
        st.markdown("---")
        st.caption("Interval: 1H | Lookback: 48H")
        st.caption("Model: TemporalAttention LSTM")
        run_btn = st.button("🔥 Scan Market")

    h_key = {"3 Days": "3d", "7 Days": "7d"}[horizon_sel]

    if 'pred_data' not in st.session_state: st.session_state.pred_data = None
    if run_btn or st.session_state.pred_data is None:
        if engine.ready:
            with st.spinner(f"Decoding {symbol}..."):
                st.session_state.pred_data = engine.predict(symbol)

    res = st.session_state.pred_data
    if res and "error" not in res:
        p = res["predictions"][h_key]
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current Price", f"${res['latest_price']:,.2f}")
        k2.metric("Horizon", horizon_sel)
        color = "normal" if p["direction"] == "UP" else "inverse"
        
        # Display raw prob against optimized threshold
        raw_prob_val = f"{p['raw_prob']:.1f}% vs Thr. {p['threshold']:.1f}%"
        k3.metric("Direction", p["direction"], delta=raw_prob_val, delta_color=color)
        
        trend = "Bullish" if res["indicators"]["up_trend"] > 50 else "Bearish"
        k4.metric("Micro-Trend", trend)

        st.write("---")
        c1, c2 = st.columns([2.5, 1.2])
        with c1:
            st.subheader(f"📈 72H Trend: {symbol}")
            chart = pd.DataFrame(res["history"], columns=["Price"])
            chart["SMA_20"] = chart["Price"].rolling(20).mean()
            st.line_chart(chart, color=["#58a6ff", "#f85149"], height=380)

            st.write("---")
            st.subheader("💡 Strategic Insight")
            bb_lvl = res["indicators"]["bb_pos"]["label"]
            msg = f"Attention layers isolate **{'bullish' if p['direction']=='UP' else 'bearish'} structural signals** for {horizon_sel}. "
            if bb_lvl in ["High", "Low"]:
                msg += f"Current condition is near Bollinger boundaries ({bb_lvl})."
            st.success(msg)

        with c2:
            st.subheader("🎯 Signal Conviction")
            st.progress(p["confidence"] / 100)
            st.markdown(f'<span class="{"up-text" if p["direction"]=="UP" else "down-text"}">'
                        f'{p["direction"]}</span> (Relative Conf.)', unsafe_allow_html=True)
            
            st.write("---")
            st.subheader("📋 Alpha Factors")
            for n, v, l in [
                ("RSI", res["indicators"]["rsi"]["value"], res["indicators"]["rsi"]["label"]),
                ("Bollinger Z", res["indicators"]["bb_pos"]["value"], res["indicators"]["bb_pos"]["label"]),
                ("24H Return", res["indicators"]["momentum"], f'{res["indicators"]["momentum"]:.1f}%'),
                ("Up-Trend Density", res["indicators"]["up_trend"], ""),
            ]:
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:0.8rem;color:#8b949e">{n}</span><br>
                    <b>{v}</b> <span style="font-size:0.8rem;margin-left:8px">({l})</span>
                </div>""", unsafe_allow_html=True)
    elif res and "error" in res:
        st.error(res["error"])


# ==================== TAB 2: MODEL HEALTH ====================
with tab2:
    st.write("---")
    st.subheader("📊 Relative Edge Assessment (V3)")
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            report = json.load(f)
            
        st.markdown(f"""
        <div class="report-box">
            <b>Setup:</b> {report['n_folds']}-fold Walk-Forward | <b>Gap:</b> {report['purge_gap']}h <br>
            <b>Model:</b> LSTM + TemporalAttention + LayerNorm <br>
            <b>Inputs:</b> {report.get('n_timesteps', 48)}h window | {report.get('n_features', 20)} features <br>
            <b>Target:</b> Strictly balanced relative outperformance (vs rolling median)
        </div>""", unsafe_allow_html=True)
        st.write("")
        
        rows = []
        for h, m in report['summary'].items():
            if 'precision' in m:
                rows.append({
                    'Horizon': h,
                    'ROC-AUC': f"{m['auc']['mean']:.3f} (±{m['auc']['std']:.3f})",
                    'F1 Score': f"{m['f1']['mean']:.3f} (±{m['f1']['std']:.3f})",
                    'Accuracy': f"{m['accuracy']['mean']*100:.1f}%",
                    'Precision': f"{m['precision']['mean']:.3f}",
                    'Recall': f"{m['recall']['mean']:.3f}",
                    'Dyn. Thresh': f"{m['threshold']['mean']:.3f}",
                    'Baseline (UP)': f"{m['baseline']['mean']*100:.1f}%"
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
        st.write("---")
        st.subheader("📈 Diagnostics Per Fold")
        for i, fold in enumerate(report['folds']):
            with st.expander(f"Fold #{i+1} Details"):
                f_rows = []
                for h, m in fold.items():
                    f_rows.append({
                        'Horizon': h, 'AUC': m['auc'], 'F1': m['f1'],
                        'Prec': m.get('precision', 0), 'Rec': m.get('recall', 0),
                        'Dyn. Thresh': m.get('threshold', 0.5),
                        'TN/FP/FN/TP': str(m.get('confusion_matrix', [[0,0],[0,0]]))
                    })
                st.dataframe(pd.DataFrame(f_rows), hide_index=True)
    else:
        st.warning("No metrics found. Run 'python train.py'.")

# ==================== TAB 3: BACKTEST ====================
with tab3:
    st.write("---")
    st.subheader("💰 Strategy Simulation")
    st.caption("Logic: Go LONG if raw probability > Dynamic Threshold. Otherwise cash (FLAT).")
    
    if os.path.exists(BACKTEST_PATH):
        with open(BACKTEST_PATH) as f:
            bt = json.load(f)
            
        for h, data in bt.items():
            st.markdown(f"### Strategy: {h}")
            b1, b2, b3, b4 = st.columns(4)
            r_col = "normal" if data['cumulative_return'] > 0 else "inverse"
            b1.metric("Net Accumulation", f"{data['cumulative_return']:+.2f}%", 
                      delta=f"vs B&H {data['buy_hold_return']:+.2f}%", delta_color=r_col)
            b2.metric("Sharpe", f"{data['sharpe_ratio']:.2f}")
            b3.metric("Hit Rate", f"{data['win_rate']:.1f}%")
            b4.metric("Max Trailing DD", f"{data['max_drawdown']:.2f}%")
            
            if data.get('equity_curve'):
                ec = pd.DataFrame({'Strategy': data['equity_curve']})
                bh_drift = data['buy_hold_return']/100 / len(data['equity_curve'])
                ec['Buy&Hold Avg'] = np.cumprod(1 + np.full(len(data['equity_curve']), bh_drift))
                st.line_chart(ec, color=["#00C805", "#505050"], height=320)
                
            st.markdown(f"<div class='report-box'>Trades Taken: {data['n_trades']} ({data['n_long']} LONG)</div>", unsafe_allow_html=True)
            st.write("---")
    else:
        st.warning("No backtest run. Execute 'python backtest.py'.")
