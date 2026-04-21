"""
backtest.py â€” Financial Backtest (V2)
---------------------------------------
Validates model predictions using the new thresholds.
Strategy:
  - Default: Cash (0% return)
  - If model prob > optimized threshold: Go LONG.
"""

import json

import numpy as np
import pandas as pd

from config import BACKTEST_PATH, COINS, FEATURES, FEE_SCENARIOS, HORIZONS, N_TIMESTEPS
from core import load_artifacts
from features import engineer_features, fetch_data


def run_backtest(horizon="3d"):
    print(f"\n{'='*50}")
    print(f"  BACKTESTING â€” {horizon} horizon")
    print(f"{'='*50}")

    # Section 1: load trained artifacts and pick the threshold for this horizon
    try:
        model, scaler, thresholds = load_artifacts()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return None

    thresh = thresholds.get(horizon, 0.5)
    print(f"  Using dynamic threshold for {horizon}: {thresh:.4f}")

    step = HORIZONS[horizon]
    horizon_idx = list(HORIZONS.keys()).index(horizon)

    # Section 2: replay the backtest window and collect trade outcomes
    all_trades = []
    for symbol in COINS:
        raw = fetch_data(symbol, total_candles=4000)
        if raw.empty:
            continue

        df = engineer_features(raw)
        bt_start = int(len(df) * 0.7)
        orig_df = df.iloc[bt_start:].reset_index(drop=True)
        bt_df = orig_df.copy()

        if len(bt_df) < N_TIMESTEPS + step:
            continue

        bt_df[FEATURES] = scaler.transform(bt_df[FEATURES])

        i = N_TIMESTEPS
        while i + step <= len(bt_df):
            window = bt_df[FEATURES].iloc[i - N_TIMESTEPS : i].values
            X = window.reshape(1, N_TIMESTEPS, len(FEATURES))

            prob = model.predict(X, verbose=0)[0][horizon_idx]
            signal = "LONG" if prob > thresh else "FLAT"

            entry_price = orig_df["Close"].iloc[i]
            exit_idx = min(i + step - 1, len(orig_df) - 1)
            exit_price = orig_df["Close"].iloc[exit_idx]
            actual_return = (exit_price - entry_price) / entry_price

            past_30d_idx = max(0, i - 24 * 30)
            past_month_ret = (
                (orig_df["Close"].iloc[i] - orig_df["Close"].iloc[past_30d_idx])
                / orig_df["Close"].iloc[past_30d_idx]
            )
            if past_month_ret > 0.05:
                regime = "Bull"
            elif past_month_ret < -0.05:
                regime = "Bear"
            else:
                regime = "Sideways"

            all_trades.append(
                {
                    "coin": symbol,
                    "signal": signal,
                    "regime": regime,
                    "actual_return": float(actual_return),
                }
            )
            i += step

    if not all_trades:
        print("No trades generated.")
        return None

    trades_df = pd.DataFrame(all_trades)

    # Section 3: measure returns under different fee/slippage assumptions
    trade_days = HORIZONS[horizon] / 24
    trades_per_year = 365 / trade_days
    scenario_results = {}

    print("\n  [FRICTION STRESS TEST]")
    print(f"  {'Scenario':<15} | {'Net Return':<10} | {'Sharpe':<8} | {'Max DD':<8}")
    print("-" * 55)

    for scenario_name, fee_penalty in FEE_SCENARIOS.items():
        strat_ret = np.where(trades_df["signal"] == "LONG", trades_df["actual_return"] - fee_penalty, 0.0)
        cum_strategy = float(np.prod(1 + strat_ret) - 1)
        sharpe = float(np.mean(strat_ret) / (np.std(strat_ret) + 1e-9) * np.sqrt(trades_per_year))

        equity = np.cumprod(1 + strat_ret)
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / peak))

        longs_only = strat_ret[trades_df["signal"] == "LONG"]
        win_rate = float((longs_only > 0).mean() * 100) if len(longs_only) > 0 else 0.0

        print(f"  {scenario_name:<15} | {cum_strategy*100:>+9.2f}% | {sharpe:>6.2f} | {max_dd*100:>+7.2f}%")
        scenario_results[scenario_name] = {
            "return": round(cum_strategy * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd * 100, 2),
            "win_rate": round(win_rate, 1),
            "equity_curve": [float(x) for x in equity.tolist()],
        }

    # Section 4: check how the LONG signals behave across market regimes
    print("\n  [REGIME VALIDATION (Realistic Fees)]")
    long_trades = trades_df[trades_df["signal"] == "LONG"].copy()
    long_trades["net_ret"] = long_trades["actual_return"] - FEE_SCENARIOS["Realistic"]

    for regime in ["Bull", "Bear", "Sideways"]:
        reg_trades = long_trades[long_trades["regime"] == regime]
        if len(reg_trades) == 0:
            continue
        reg_acc = (reg_trades["net_ret"] > 0).mean() * 100
        reg_ret = (np.prod(1 + reg_trades["net_ret"]) - 1) * 100
        print(
            f"  {regime:<10} | Trades: {len(reg_trades):<3} | "
            f"Win Rate: {reg_acc:>5.1f}% | Net Ret: {reg_ret:>+8.2f}%"
        )

    cum_buyhold = float(np.prod(1 + trades_df["actual_return"]) - 1)

    print("\n  [TRADE FREQUENCY]")
    print(f"  Total Trades Analyzed: {len(trades_df)} (Avg holding period: {trade_days:.1f} Days)")
    print(f"  Execution Rate: {len(long_trades)} LONGs, {len(trades_df)-len(long_trades)} FLATs")
    print(f"  Naive Buy & Hold: {cum_buyhold*100:+.2f}%")

    return {
        "horizon": horizon,
        "scenarios": scenario_results,
        "trade_stats": {
            "total": len(trades_df),
            "longs": len(long_trades),
            "flats": len(trades_df) - len(long_trades),
            "holding_days": trade_days,
        },
        "buy_hold_return": round(cum_buyhold * 100, 2),
    }


def main():
    all_results = {}
    for horizon in HORIZONS.keys():
        result = run_backtest(horizon)
        if result:
            all_results[horizon] = result

    if all_results:
        with open(BACKTEST_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[+] Backtest results -> {BACKTEST_PATH}")


if __name__ == "__main__":
    main()
