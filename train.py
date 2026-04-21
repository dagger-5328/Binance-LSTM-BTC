"""
train.py — Walk-Forward Training Pipeline (V2)
------------------------------------------------
V2 improvements:
  - Uses LSTM+Attention model
  - Threshold optimization (Youden's J) instead of fixed 0.5
  - Confusion matrix + precision/recall reporting
  - Saves optimized thresholds for deployment
"""
import os
print("RUNNING FILE:", os.path.abspath(__file__))
import os, json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score, precision_score, recall_score,
                             confusion_matrix, roc_curve)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import *
from features import fetch_data, engineer_features, create_sequences
from model import build_model


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
    horizons = list(HORIZONS.keys())
    results = {}

    for idx, h in enumerate(horizons):
        p = probs[:, idx]
        y = y_test[:, idx]
        thresh = thresholds.get(h, 0.5) if thresholds else 0.5
        pred = (p > thresh).astype(int)

        baseline = max(np.mean(y), 1 - np.mean(y))
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = 0.5

        cm = confusion_matrix(y, pred, labels=[0, 1])

        results[h] = {
            'accuracy': round(float(accuracy_score(y, pred)), 4),
            'f1': round(float(f1_score(y, pred, zero_division=0)), 4),
            'mcc': round(float(matthews_corrcoef(y, pred)), 4),
            'auc': round(float(auc), 4),
            'precision': round(float(precision_score(y, pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y, pred, zero_division=0)), 4),
            'baseline': round(float(baseline), 4),
            'threshold': round(float(thresh), 4),
            'class_balance': round(float(np.mean(y)), 4),
            'n_samples': int(len(y)),
            'confusion_matrix': cm.tolist()
        }
    return results


def main():
    print("=" * 60)
    print("  WALK-FORWARD TRAINING PIPELINE (V2)")
    print("  LSTM + Attention | Threshold Optimization")
    print("=" * 60)

    # ---- Step 1: Load & Engineer ----
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
        print("ERROR: No data loaded.")
        return

    min_len = min(len(d) for d in coin_dfs)
    coin_dfs = [d.tail(min_len).reset_index(drop=True) for d in coin_dfs]

    # ---- Step 2: Walk-Forward ----
    print(f"\n[2/4] {N_FOLDS}-fold walk-forward | purge={PURGE_GAP}h")

    n = min_len
    test_size = n // (N_FOLDS + 2)
    all_fold_results = []
    best_mcc, best_model, best_scaler, best_thresholds = -1, None, None, {}

    callbacks = [
        EarlyStopping(patience=PATIENCE_STOP, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR)
    ]

    for fold in range(N_FOLDS):
        test_end = n - (N_FOLDS - fold - 1) * test_size
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

        all_X_tr, all_y_tr, all_X_te, all_y_te = [], [], [], []
        for df in coin_dfs:
            tr = df.iloc[:train_end].copy()
            te = df.iloc[test_start:test_end].copy()
            tr[FEATURES] = scaler.transform(tr[FEATURES])
            te[FEATURES] = scaler.transform(te[FEATURES])
            Xr, yr = create_sequences(tr, N_TIMESTEPS, FEATURES, TARGET_COLS)
            Xe, ye = create_sequences(te, N_TIMESTEPS, FEATURES, TARGET_COLS)
            if len(Xr) > 0: all_X_tr.append(Xr); all_y_tr.append(yr)
            if len(Xe) > 0: all_X_te.append(Xe); all_y_te.append(ye)

        X_train = np.vstack(all_X_tr); y_train = np.vstack(all_y_tr)
        X_test = np.vstack(all_X_te); y_test = np.vstack(all_y_te)

        val_n = max(1, int(len(X_train) * 0.1))
        X_val, y_val = X_train[-val_n:], y_train[-val_n:]
        X_train, y_train = X_train[:-val_n], y_train[:-val_n]

        print(f"  Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")
        
        # Save validation / test data context at fold scope
        fold_X_test = X_test.copy()
        fold_y_test = y_test.copy()

        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=callbacks, verbose=0)

        # Optimize thresholds on validation set
        val_probs = model.predict(X_val, verbose=0)
        thresholds = {}
        for idx, h in enumerate(HORIZONS.keys()):
            thresholds[h] = optimize_threshold(y_val[:, idx], val_probs[:, idx])

        # Evaluate with optimized thresholds
        fold_res = evaluate(model, X_test, y_test, thresholds)
        all_fold_results.append(fold_res)

        for h, m in fold_res.items():
            cm = m['confusion_matrix']
            print(f"  [{h}] Acc={m['accuracy']*100:.1f}% F1={m['f1']:.3f} "
                  f"MCC={m['mcc']:.3f} AUC={m['auc']:.3f} "
                  f"Prec={m['precision']:.3f} Rec={m['recall']:.3f} "
                  f"thresh={m['threshold']:.3f}")
            print(f"       CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}  "
                  f"(baseline={m['baseline']*100:.1f}%)")

        avg_mcc = np.mean([m['mcc'] for m in fold_res.values()])
        if avg_mcc > best_mcc:
            best_mcc = avg_mcc
            best_model, best_scaler, best_thresholds = model, scaler, thresholds
            best_X_test, best_y_test = fold_X_test, fold_y_test

    # ---- Step 3: Report ----
    print(f"\n{'='*60}")
    print("  WALK-FORWARD PERFORMANCE REPORT")
    print(f"{'='*60}")

    summary = {}
    for h in HORIZONS.keys():
        mets = [f[h] for f in all_fold_results if h in f]
        if not mets:
            continue
        summary[h] = {}
        for k in ['accuracy','f1','mcc','auc','precision','recall','baseline','threshold']:
            vals = [m[k] for m in mets]
            summary[h][k] = {'mean': round(float(np.mean(vals)), 4),
                             'std': round(float(np.std(vals)), 4)}
        s = summary[h]
        print(f"\n  [{h}]")
        print(f"    Accuracy  : {s['accuracy']['mean']*100:.1f}% +/- {s['accuracy']['std']*100:.1f}%  "
              f"(Baseline: {s['baseline']['mean']*100:.1f}%)")
        print(f"    F1 Score  : {s['f1']['mean']:.3f} +/- {s['f1']['std']:.3f}")
        print(f"    MCC       : {s['mcc']['mean']:.3f} +/- {s['mcc']['std']:.3f}")
        print(f"    ROC-AUC   : {s['auc']['mean']:.3f} +/- {s['auc']['std']:.3f}")
        print(f"    Precision : {s['precision']['mean']:.3f}  Recall: {s['recall']['mean']:.3f}")
        print(f"    Threshold : {s['threshold']['mean']:.3f}")

    # ---- Step 3.5: Permutation Importance (Ablation) ----
    if best_model is not None:
        print(f"\n{'='*60}")
        print("  FEATURE PERMUTATION IMPORTANCE (Ablation)")
        print(f"{'='*60}")
        print("  Shuffling features one by one to measure drop in ROC-AUC.")
        print("  (Note: Permutation may underestimate highly correlated features)")
        
        baseline_probs = best_model.predict(best_X_test, verbose=0)
        baseline_auc = roc_auc_score(best_y_test[:, 0], baseline_probs[:, 0]) # 3d horizon
        
        importances = []
        for i, feature_name in enumerate(FEATURES):
            np.random.seed(42)
            X_shuffled = best_X_test.copy()
            
            # Shuffle the chosen feature across the batch dimension, for all timesteps
            for seq_idx in range(X_shuffled.shape[0]):
                np.random.shuffle(X_shuffled[seq_idx, :, i])
                
            shuff_probs = best_model.predict(X_shuffled, verbose=0)
            shuff_auc = roc_auc_score(best_y_test[:, 0], shuff_probs[:, 0])
            drop = baseline_auc - shuff_auc
            importances.append((feature_name, drop))
            
        importances.sort(key=lambda x: x[1], reverse=True)
        for rank, (feat, drop) in enumerate(importances):
            print(f"  {rank+1:>2}. {feat:<15} | AUC Drop: {drop:>+6.4f}")

    # ---- Step 4: Save ----
    os.makedirs('models', exist_ok=True)
    if best_model:
        best_model.save("models/model.keras")
        joblib.dump(best_scaler, "models/scaler.pkl")
        print("\n[+] Model -> models/model.keras")
        print(f"[+] Scaler -> {SCALER_PATH}")

    report = {
        'n_folds': N_FOLDS, 'purge_gap': PURGE_GAP,
        'coins': COINS, 'n_features': len(FEATURES),
        'n_timesteps': N_TIMESTEPS,
        'architecture': 'LSTM+Attention+LayerNorm',
        'thresholds': best_thresholds,
        'summary': summary, 'folds': all_fold_results
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[+] Metrics -> {METRICS_PATH}")
    print(f"\n{'='*60}\n  DONE\n{'='*60}")

if __name__ == "__main__":
    main()
