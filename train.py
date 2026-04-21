"""
train.py â€” Walk-Forward Training Pipeline (V2)
------------------------------------------------
V2 improvements:
  - Uses LSTM+Attention model
  - Threshold optimization (Youden's J) instead of fixed 0.5
  - Confusion matrix + precision/recall reporting
  - Saves optimized thresholds for deployment
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    COINS,
    EPOCHS,
    FEATURES,
    HORIZONS,
    METRICS_PATH,
    MODEL_PATH,
    N_FOLDS,
    N_TIMESTEPS,
    PATIENCE_LR,
    PATIENCE_STOP,
    PURGE_GAP,
    SCALER_PATH,
    TARGET_COLS,
)
from features import create_sequences, engineer_features, fetch_data
from model import build_model


SUMMARY_KEYS = ["accuracy", "f1", "mcc", "auc", "precision", "recall", "baseline", "threshold"]


def optimize_threshold(y_true, probs):
    """Find optimal threshold using Youden's J statistic (maximizes TPR - FPR)."""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    except Exception:
        return 0.5


def evaluate(model, X_test, y_test, thresholds=None):
    """Compute comprehensive metrics with optimized thresholds."""
    probs = model.predict(X_test, verbose=0)
    results = {}

    for idx, horizon in enumerate(HORIZONS.keys()):
        p = probs[:, idx]
        y = y_test[:, idx]
        thresh = thresholds.get(horizon, 0.5) if thresholds else 0.5
        pred = (p > thresh).astype(int)

        baseline = max(np.mean(y), 1 - np.mean(y))
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = 0.5

        cm = confusion_matrix(y, pred, labels=[0, 1])
        results[horizon] = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y, pred)), 4),
            "auc": round(float(auc), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "baseline": round(float(baseline), 4),
            "threshold": round(float(thresh), 4),
            "class_balance": round(float(np.mean(y)), 4),
            "n_samples": int(len(y)),
            "confusion_matrix": cm.tolist(),
        }
    return results


def print_pipeline_header():
    print("=" * 60)
    print("  WALK-FORWARD TRAINING PIPELINE (V2)")
    print("  LSTM + Attention | Threshold Optimization")
    print("=" * 60)


def load_coin_data():
    """Fetch market data, engineer features, and align all assets to a shared length."""
    print(f"\n[1/4] Loading OHLCV data for {len(COINS)} coins...")
    coin_dfs = []
    for symbol in COINS:
        raw = fetch_data(symbol, total_candles=4000)
        if raw.empty:
            print(f"  [SKIP] {symbol}: no data")
            continue

        df = engineer_features(raw)
        coin_dfs.append(df)
        print(f"  [OK] {symbol}: {len(df)} rows")

    if not coin_dfs:
        return []

    min_len = min(len(df) for df in coin_dfs)
    return [df.tail(min_len).reset_index(drop=True) for df in coin_dfs]


def build_callbacks():
    """Create the Keras callbacks used for each fold."""
    return [
        EarlyStopping(patience=PATIENCE_STOP, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_LR),
    ]


def prepare_fold_datasets(coin_dfs, scaler, train_end, test_start, test_end):
    """Scale per-fold data and build the train/test sequence tensors."""
    all_X_tr, all_y_tr, all_X_te, all_y_te = [], [], [], []
    for df in coin_dfs:
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        train_df[FEATURES] = scaler.transform(train_df[FEATURES])
        test_df[FEATURES] = scaler.transform(test_df[FEATURES])

        X_train_part, y_train_part = create_sequences(train_df, N_TIMESTEPS, FEATURES, TARGET_COLS)
        X_test_part, y_test_part = create_sequences(test_df, N_TIMESTEPS, FEATURES, TARGET_COLS)

        if len(X_train_part) > 0:
            all_X_tr.append(X_train_part)
            all_y_tr.append(y_train_part)
        if len(X_test_part) > 0:
            all_X_te.append(X_test_part)
            all_y_te.append(y_test_part)

    return np.vstack(all_X_tr), np.vstack(all_y_tr), np.vstack(all_X_te), np.vstack(all_y_te)


def split_train_validation(X_train, y_train):
    """Reserve the most recent slice of the training set for validation."""
    val_n = max(1, int(len(X_train) * 0.1))
    X_val, y_val = X_train[-val_n:], y_train[-val_n:]
    X_train, y_train = X_train[:-val_n], y_train[:-val_n]
    return X_train, y_train, X_val, y_val


def print_fold_results(fold_results):
    """Print fold metrics in the same compact format used before."""
    for horizon, metrics in fold_results.items():
        cm = metrics["confusion_matrix"]
        print(f"  [{horizon}] Acc={metrics['accuracy']*100:.1f}% F1={metrics['f1']:.3f} "
              f"MCC={metrics['mcc']:.3f} AUC={metrics['auc']:.3f} "
              f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f} "
              f"thresh={metrics['threshold']:.3f}")
        print(f"       CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}  "
              f"(baseline={metrics['baseline']*100:.1f}%)")


def run_walk_forward(coin_dfs):
    """Train across folds, tune thresholds, and keep the strongest checkpoint."""
    print(f"\n[2/4] {N_FOLDS}-fold walk-forward | purge={PURGE_GAP}h")

    n_rows = len(coin_dfs[0])
    test_size = n_rows // (N_FOLDS + 2)
    callbacks = build_callbacks()

    all_fold_results = []
    best_mcc = -1
    best_model = None
    best_scaler = None
    best_thresholds = {}
    best_X_test = None
    best_y_test = None

    for fold in range(N_FOLDS):
        # Section 1: choose the train/test ranges for this fold
        test_end = n_rows - (N_FOLDS - fold - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - PURGE_GAP

        if train_end < N_TIMESTEPS + 50:
            continue

        print(f"\n{'='*50}")
        print(f"  FOLD {fold+1}/{N_FOLDS}  |  train [0:{train_end}] test [{test_start}:{test_end}]")
        print(f"{'='*50}")

        scaler = StandardScaler()
        train_raw = [df.iloc[:train_end][FEATURES] for df in coin_dfs]
        scaler.fit(pd.concat(train_raw))

        # Section 2: build the scaled train/validation/test tensors
        X_train, y_train, X_test, y_test = prepare_fold_datasets(
            coin_dfs, scaler, train_end, test_start, test_end
        )
        X_train, y_train, X_val, y_val = split_train_validation(X_train, y_train)
        print(f"  Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

        # Section 3: train the fold model and calibrate its thresholds
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

        val_probs = model.predict(X_val, verbose=0)
        thresholds = {}
        for idx, horizon in enumerate(HORIZONS.keys()):
            thresholds[horizon] = optimize_threshold(y_val[:, idx], val_probs[:, idx])

        # Section 4: score the fold and keep the strongest checkpoint
        fold_results = evaluate(model, X_test, y_test, thresholds)
        all_fold_results.append(fold_results)
        print_fold_results(fold_results)

        avg_mcc = np.mean([metrics["mcc"] for metrics in fold_results.values()])
        if avg_mcc > best_mcc:
            best_mcc = avg_mcc
            best_model = model
            best_scaler = scaler
            best_thresholds = thresholds
            best_X_test = X_test.copy()
            best_y_test = y_test.copy()

    return {
        "folds": all_fold_results,
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_thresholds": best_thresholds,
        "best_X_test": best_X_test,
        "best_y_test": best_y_test,
    }


def build_summary(all_fold_results):
    """Aggregate mean/std metrics across the walk-forward folds."""
    print(f"\n{'='*60}")
    print("  WALK-FORWARD PERFORMANCE REPORT")
    print(f"{'='*60}")

    summary = {}
    for horizon in HORIZONS.keys():
        metrics_by_fold = [fold[horizon] for fold in all_fold_results if horizon in fold]
        if not metrics_by_fold:
            continue

        summary[horizon] = {}
        for key in SUMMARY_KEYS:
            values = [metrics[key] for metrics in metrics_by_fold]
            summary[horizon][key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }

        stats = summary[horizon]
        print(f"\n  [{horizon}]")
        print(f"    Accuracy  : {stats['accuracy']['mean']*100:.1f}% +/- {stats['accuracy']['std']*100:.1f}%  "
              f"(Baseline: {stats['baseline']['mean']*100:.1f}%)")
        print(f"    F1 Score  : {stats['f1']['mean']:.3f} +/- {stats['f1']['std']:.3f}")
        print(f"    MCC       : {stats['mcc']['mean']:.3f} +/- {stats['mcc']['std']:.3f}")
        print(f"    ROC-AUC   : {stats['auc']['mean']:.3f} +/- {stats['auc']['std']:.3f}")
        print(f"    Precision : {stats['precision']['mean']:.3f}  Recall: {stats['recall']['mean']:.3f}")
        print(f"    Threshold : {stats['threshold']['mean']:.3f}")

    return summary


def print_permutation_importance(best_model, best_X_test, best_y_test):
    """Show which features matter most on the best held-out fold."""
    if best_model is None or best_X_test is None or best_y_test is None:
        return

    print(f"\n{'='*60}")
    print("  FEATURE PERMUTATION IMPORTANCE (Ablation)")
    print(f"{'='*60}")
    print("  Shuffling features one by one to measure drop in ROC-AUC.")
    print("  (Note: Permutation may underestimate highly correlated features)")

    baseline_probs = best_model.predict(best_X_test, verbose=0)
    baseline_auc = roc_auc_score(best_y_test[:, 0], baseline_probs[:, 0])

    importances = []
    for feature_idx, feature_name in enumerate(FEATURES):
        np.random.seed(42)
        X_shuffled = best_X_test.copy()

        for seq_idx in range(X_shuffled.shape[0]):
            np.random.shuffle(X_shuffled[seq_idx, :, feature_idx])

        shuffled_probs = best_model.predict(X_shuffled, verbose=0)
        shuffled_auc = roc_auc_score(best_y_test[:, 0], shuffled_probs[:, 0])
        importances.append((feature_name, baseline_auc - shuffled_auc))

    importances.sort(key=lambda item: item[1], reverse=True)
    for rank, (feature_name, drop) in enumerate(importances, start=1):
        print(f"  {rank:>2}. {feature_name:<15} | AUC Drop: {drop:>+6.4f}")


def save_artifacts(best_model, best_scaler, best_thresholds, summary, all_fold_results):
    """Persist the best model plus the training report used by deployment."""
    print("\n[4/4] Saving trained artifacts...")
    os.makedirs("models", exist_ok=True)

    if best_model is not None:
        best_model.save(MODEL_PATH)
        joblib.dump(best_scaler, SCALER_PATH)
        print(f"[+] Model -> {MODEL_PATH}")
        print(f"[+] Scaler -> {SCALER_PATH}")

    report = {
        "n_folds": N_FOLDS,
        "purge_gap": PURGE_GAP,
        "coins": COINS,
        "n_features": len(FEATURES),
        "n_timesteps": N_TIMESTEPS,
        "architecture": "LSTM+Attention+LayerNorm",
        "thresholds": best_thresholds,
        "summary": summary,
        "folds": all_fold_results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Metrics -> {METRICS_PATH}")


def main():
    print_pipeline_header()

    coin_dfs = load_coin_data()
    if not coin_dfs:
        print("ERROR: No data loaded.")
        return

    walk_forward = run_walk_forward(coin_dfs)
    all_fold_results = walk_forward["folds"]
    best_model = walk_forward["best_model"]
    best_scaler = walk_forward["best_scaler"]
    best_thresholds = walk_forward["best_thresholds"]
    best_X_test = walk_forward["best_X_test"]
    best_y_test = walk_forward["best_y_test"]

    print("\n[3/4] Building validation report...")
    summary = build_summary(all_fold_results)
    print_permutation_importance(best_model, best_X_test, best_y_test)
    save_artifacts(best_model, best_scaler, best_thresholds, summary, all_fold_results)

    print(f"\n{'='*60}\n  DONE\n{'='*60}")


if __name__ == "__main__":
    main()
