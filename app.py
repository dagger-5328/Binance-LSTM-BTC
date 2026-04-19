"""
Streamlit dashboard for live predictions, validation metrics, and backtests.
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from config import BACKTEST_PATH, COINS, METRICS_PATH
from core import ModelEngine


@st.cache_resource
def get_engine():
    return ModelEngine()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def render_live_tab(engine):
    st.write("---")
    with st.sidebar:
        st.header("Market Controls")
        symbol = st.selectbox("Asset Pair", COINS, index=0)
        horizon_label = st.selectbox("Forecast Horizon", ["3 Days", "7 Days"], index=1)
        st.markdown("---")
        st.caption("Interval: 1H | Lookback: 48H")
        st.caption("Model: TemporalAttention LSTM")
        run_btn = st.button("Scan Market")

    horizon_key = {"3 Days": "3d", "7 Days": "7d"}[horizon_label]

    if "pred_data" not in st.session_state:
        st.session_state.pred_data = None

    if run_btn or st.session_state.pred_data is None:
        if engine.ready:
            with st.spinner(f"Decoding {symbol}..."):
                st.session_state.pred_data = engine.predict(symbol)

    result = st.session_state.pred_data
    if not result:
        return
    if "error" in result:
        st.error(result["error"])
        return

    prediction = result["predictions"][horizon_key]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Price", f"${result['latest_price']:,.2f}")
    k2.metric("Horizon", horizon_label)
    k3.metric(
        "Direction",
        prediction["direction"],
        delta=f"{prediction['raw_prob']:.1f}% vs Thr. {prediction['threshold']:.1f}%",
        delta_color="normal" if prediction["direction"] == "UP" else "inverse",
    )
    k4.metric("Micro-Trend", "Bullish" if result["indicators"]["up_trend"] > 50 else "Bearish")

    st.write("---")
    left, right = st.columns([2.5, 1.2])

    with left:
        st.subheader(f"72H Trend: {symbol}")
        chart = pd.DataFrame(result["history"], columns=["Price"])
        chart["SMA_20"] = chart["Price"].rolling(20).mean()
        st.line_chart(chart, color=["#58a6ff", "#f85149"], height=380)

        st.write("---")
        st.subheader("Strategic Insight")
        bb_level = result["indicators"]["bb_pos"]["label"]
        message = (
            f"Attention layers isolate {'bullish' if prediction['direction'] == 'UP' else 'bearish'} "
            f"structural signals for {horizon_label}."
        )
        if bb_level in {"High", "Low"}:
            message += f" Current condition is near Bollinger boundaries ({bb_level})."
        st.success(message)

    with right:
        st.subheader("Signal Conviction")
        st.progress(prediction["confidence"] / 100)
        st.write(f"**{prediction['direction']}** (Relative Confidence)")

        st.write("---")
        st.subheader("Alpha Factors")
        for name, value, label in [
            ("RSI", result["indicators"]["rsi"]["value"], result["indicators"]["rsi"]["label"]),
            ("Bollinger Z", result["indicators"]["bb_pos"]["value"], result["indicators"]["bb_pos"]["label"]),
            ("24H Return", result["indicators"]["momentum"], f"{result['indicators']['momentum']:.1f}%"),
            ("Up-Trend Density", result["indicators"]["up_trend"], ""),
        ]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="font-size:0.8rem;color:#8b949e">{name}</span><br>
                    <b>{value}</b> <span style="font-size:0.8rem;margin-left:8px">({label})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metrics_tab():
    st.write("---")
    st.subheader("Relative Edge Assessment")

    if not os.path.exists(METRICS_PATH):
        st.warning("No metrics found. Run 'python train.py'.")
        return

    report = load_json(METRICS_PATH)

    st.markdown(
        f"""
        <div class="report-box">
            <b>Setup:</b> {report['n_folds']}-fold Walk-Forward | <b>Gap:</b> {report['purge_gap']}h <br>
            <b>Model:</b> {report.get('architecture', 'LSTM + TemporalAttention + LayerNorm')} <br>
            <b>Inputs:</b> {report.get('n_timesteps', 48)}h window | {report.get('n_features', 20)} features <br>
            <b>Target:</b> Relative outperformance vs the previous same-horizon move
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []
    for horizon, metrics in report["summary"].items():
        rows.append(
            {
                "Horizon": horizon,
                "ROC-AUC": f"{metrics['auc']['mean']:.3f} (+/- {metrics['auc']['std']:.3f})",
                "F1 Score": f"{metrics['f1']['mean']:.3f} (+/- {metrics['f1']['std']:.3f})",
                "Accuracy": f"{metrics['accuracy']['mean'] * 100:.1f}%",
                "Precision": f"{metrics['precision']['mean']:.3f}",
                "Recall": f"{metrics['recall']['mean']:.3f}",
                "Dyn. Thresh": f"{metrics['threshold']['mean']:.3f}",
                "Baseline": f"{metrics['baseline']['mean'] * 100:.1f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("---")
    st.subheader("Diagnostics Per Fold")
    for idx, fold in enumerate(report["folds"], start=1):
        with st.expander(f"Fold #{idx} Details"):
            fold_rows = []
            for horizon, metrics in fold.items():
                fold_rows.append(
                    {
                        "Horizon": horizon,
                        "AUC": metrics["auc"],
                        "F1": metrics["f1"],
                        "Prec": metrics.get("precision", 0),
                        "Rec": metrics.get("recall", 0),
                        "Dyn. Thresh": metrics.get("threshold", 0.5),
                        "TN/FP/FN/TP": str(metrics.get("confusion_matrix", [[0, 0], [0, 0]])),
                    }
                )
            st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)


def render_backtest_tab():
    st.write("---")
    st.subheader("Strategy Simulation")
    st.caption("Logic: Go LONG if raw probability > dynamic threshold. Otherwise stay in cash.")

    if not os.path.exists(BACKTEST_PATH):
        st.warning("No backtest run. Execute 'python backtest.py'.")
        return

    backtest = load_json(BACKTEST_PATH)
    for horizon, data in backtest.items():
        st.markdown(f"### Strategy: {horizon}")
        realistic = data.get("scenarios", {}).get("Realistic", {})
        trade_stats = data.get("trade_stats", {})

        b1, b2, b3, b4 = st.columns(4)
        strategy_return = realistic.get("return", 0.0)
        b1.metric(
            "Net Accumulation",
            f"{strategy_return:+.2f}%",
            delta=f"vs B&H {data.get('buy_hold_return', 0.0):+.2f}%",
            delta_color="normal" if strategy_return > 0 else "inverse",
        )
        b2.metric("Sharpe", f"{realistic.get('sharpe', 0.0):.2f}")
        b3.metric("Hit Rate", f"{realistic.get('win_rate', 0.0):.1f}%")
        b4.metric("Max Trailing DD", f"{realistic.get('max_dd', 0.0):.2f}%")

        equity_curve = realistic.get("equity_curve", [])
        if equity_curve:
            equity_df = pd.DataFrame({"Strategy": equity_curve})
            drift = data.get("buy_hold_return", 0.0) / 100 / len(equity_curve)
            equity_df["Buy&Hold Avg"] = np.cumprod(1 + np.full(len(equity_curve), drift))
            st.line_chart(equity_df, color=["#00C805", "#505050"], height=320)

        scenario_rows = []
        for scenario_name, scenario_metrics in data.get("scenarios", {}).items():
            scenario_rows.append(
                {
                    "Scenario": scenario_name,
                    "Return %": scenario_metrics.get("return", 0.0),
                    "Sharpe": scenario_metrics.get("sharpe", 0.0),
                    "Win Rate %": scenario_metrics.get("win_rate", 0.0),
                    "Max DD %": scenario_metrics.get("max_dd", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)

        st.markdown(
            (
                "<div class='report-box'>"
                f"Trades Taken: {trade_stats.get('total', 0)} "
                f"({trade_stats.get('longs', 0)} LONG / {trade_stats.get('flats', 0)} FLAT)"
                f" | Avg holding period: {trade_stats.get('holding_days', 0):.1f} days"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.write("---")


def main():
    engine = get_engine()

    st.set_page_config(
        page_title="Crypto Pulse Analytics",
        page_icon="C",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .main { background-color: #0d1117; color: #c9d1d9; }
            .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
            .stButton>button { width: 100%; border-radius: 8px; background-color: #238636; color: white; border: none; font-weight: bold; }
            .stButton>button:hover { background-color: #2ea043; }
            h1, h2, h3 { color: #58a6ff; font-weight: 700; }
            .metric-card { padding: 15px; background: #161b22; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 10px; }
            .report-box { padding: 20px; background: #1c2128; border-radius: 12px; border-left: 4px solid #58a6ff; margin: 10px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_col, status_col = st.columns([4, 1])
    with title_col:
        st.title("Crypto Pulse Analytics")
        st.caption("LSTM Attention + LayerNorm | Relative Outperformance Classifier")
    with status_col:
        if engine.ready:
            st.info("Engine Online")
        else:
            st.warning(engine.last_error or "Train model first")

    live_tab, metrics_tab, backtest_tab = st.tabs(
        ["Live Tracker", "Edge Detection", "Portfolio Backtest"]
    )

    with live_tab:
        render_live_tab(engine)
    with metrics_tab:
        render_metrics_tab()
    with backtest_tab:
        render_backtest_tab()


if __name__ == "__main__":
    main()
