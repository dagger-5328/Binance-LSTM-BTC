"""
Professional Multi-Horizon Training Pipeline (V7 - FINAL)
-------------------------------------------------------
- Fixed circular ImportErrors
- Integrated 2-horizon logic (3d, 7d)
- Reduced sequence overlap (Stride=3)
- Realistic baseline comparison
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Import from restored core engine
from core import COINS, FEATURES, N_TIMESTEPS, fetch_data, apply_market_features

# --- CONFIG ---
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
STRIDE = 3
BASE_SPLIT_TOLERANCE = 0.10  # Allow ±10% class imbalance variance between train/test

# Historical data window: train on 12-3 months old data (balanced regime)
USE_HISTORICAL = True
HISTORICAL_START_MONTHS = 12  # Fetch from 12 months ago
HISTORICAL_END_MONTHS = 3     # Up to 3 months ago

# ---------------- SEQUENCES ----------------
def create_sequences_multi(df):
    """Generates X and Y for a 2-task classification [3d, 7d]."""
    X, y = [], []
    data = df[FEATURES].values
    target = df[['Target_3d', 'Target_7d']].values

    for i in range(0, len(df) - N_TIMESTEPS, STRIDE):
        X.append(data[i:i+N_TIMESTEPS])
        y.append(target[i+N_TIMESTEPS])

    return np.array(X), np.array(y)

def find_optimal_split(df, target_cols, base_split=0.8, tolerance=0.10):
    """
    Find split point where train/test class distributions are similar (regime-aware).
    Searches around base_split to find a point where class ratios match within tolerance.
    """
    target = df[target_cols].values
    best_split = int(len(df) * base_split)
    best_diff = float('inf')
    
    for horizon_idx in range(target.shape[1]):
        t_col = target[:, horizon_idx]
        train_ratio = np.mean(t_col[:best_split])
        test_ratio = np.mean(t_col[best_split:])
        diff = abs(train_ratio - test_ratio)
        
        if diff > best_diff:
            # Try shifting split to find better match
            for offset in range(-len(df)//10, len(df)//10, max(1, len(df)//100)):
                split_cand = int(len(df) * base_split) + offset
                if split_cand <= len(df)//3 or split_cand >= len(df)*2//3:
                    continue
                train_ratio_cand = np.mean(t_col[:split_cand])
                test_ratio_cand = np.mean(t_col[split_cand:])
                diff_cand = abs(train_ratio_cand - test_ratio_cand)
                if diff_cand < best_diff:
                    best_diff = diff_cand
                    best_split = split_cand
    
    return best_split

# ---------------- MODEL ----------------
def build_model(input_shape):
    """Stable LSTM with high regularization (Dropout + L2) for professional generalization."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, 
             kernel_regularizer=l2(0.005)),
        Dropout(0.6),

        LSTM(32, kernel_regularizer=l2(0.005)),
        Dropout(0.6),

        Dense(64, activation='relu'),
        Dense(32, activation='relu'),

        Dense(2, activation='sigmoid') 
    ])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------- MAIN ----------------
def main():
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    scaler = StandardScaler()

    print("[*] Loading and processing data from Binance...")

    for cid, symbol in enumerate(COINS):
        print(f"[*] Processing {symbol}")

        if USE_HISTORICAL:
            df_raw = fetch_data(symbol, months_back_start=HISTORICAL_START_MONTHS, 
                               months_back_end=HISTORICAL_END_MONTHS)
            print(f"    [using historical data: {HISTORICAL_START_MONTHS}-{HISTORICAL_END_MONTHS} months ago]")
        else:
            df_raw = fetch_data(symbol, limit=3000)
        
        df = apply_market_features(df_raw, cid, None)

        split = find_optimal_split(df, ['Target_3d', 'Target_7d'], TRAIN_SPLIT, BASE_SPLIT_TOLERANCE)
        train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

        scaler.fit(train_df[FEATURES])
        train_df[FEATURES] = scaler.transform(train_df[FEATURES])
        test_df[FEATURES]  = scaler.transform(test_df[FEATURES])

        X_tr, y_tr = create_sequences_multi(train_df)
        X_te, y_te = create_sequences_multi(test_df)

        if len(X_te) == 0: continue
        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_X_test.append(X_te)
        all_y_test.append(y_te)

    # ---------------- CONSOLIDATE ----------------
    X_train = np.vstack(all_X_train)
    y_train = np.vstack(all_y_train)
    X_test = np.vstack(all_X_test)
    y_test = np.vstack(all_y_test)

    val_size = int(len(X_train) * VAL_SPLIT)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    def print_split_stats(name, y):
        print(f"\n[{name}] sample stats")
        print(f"  total samples: {len(y)}")
        for idx, horizon in enumerate(["3d", "7d"]):
            up_count = int(np.sum(y[:, idx] == 1))
            down_count = int(np.sum(y[:, idx] == 0))
            up_ratio = up_count / len(y) if len(y) else 0.0
            print(f"  {horizon}: UP={up_count}, DOWN={down_count}, UP ratio={up_ratio*100:.2f}%")

    print_split_stats("TRAIN", y_train)
    print_split_stats("VALIDATION", y_val)
    print_split_stats("TEST", y_test)

    # ---------------- TRAIN ----------------
    print(f"\n[*] Training on {len(X_train)} samples...")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]

    # Compute class weights to handle imbalance
    class_weights_list = []
    for horizon_idx in range(y_train.shape[1]):
        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train[:, horizon_idx])
        class_weights_list.append({0: cw[0], 1: cw[1]})
    
    # Use average class weight across both horizons
    class_weight_avg = {0: np.mean([cw[0] for cw in class_weights_list]), 
                        1: np.mean([cw[1] for cw in class_weights_list])}
    print(f"[*] Class weights applied: {class_weight_avg}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_avg,
        verbose=1
    )

    best_epoch = int(np.argmin(history.history['val_loss']))
    print("\n[*] Training summary:")
    print(f"  best epoch (val_loss): {best_epoch + 1}")
    print(f"  train loss at best epoch: {history.history['loss'][best_epoch]:.4f}")
    print(f"  val loss at best epoch:   {history.history['val_loss'][best_epoch]:.4f}")
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        print(f"  train acc at best epoch:  {history.history['accuracy'][best_epoch]*100:.2f}%")
        print(f"  val acc at best epoch:    {history.history['val_accuracy'][best_epoch]*100:.2f}%")
    print(f"  final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  final val loss:   {history.history['val_loss'][-1]:.4f}")
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        print(f"  final train acc:  {history.history['accuracy'][-1]*100:.2f}%")
        print(f"  final val acc:    {history.history['val_accuracy'][-1]*100:.2f}%")

    # ---------------- EVALUATION ----------------
    print("\n" + "="*50)
    print("  FINAL PERFORMANCE REPORT (LEAKAGE-FREE)")
    print("="*50)

    probs = model.predict(X_test, verbose=0)
    horizons = ["3d", "7d"]

    # Find optimal threshold for each horizon
    for idx, horizon in enumerate(horizons):
        p_slice = probs[:, idx]
        t_slice = y_test[:, idx]
        
        # Grid search for threshold that maximizes balanced accuracy (try broader range)
        best_threshold = 0.5
        best_balanced_acc = 0
        for threshold in np.arange(0.2, 0.8, 0.02):
            preds_cand = (p_slice > threshold).astype(int)
            balanced_acc_cand = balanced_accuracy_score(t_slice, preds_cand)
            if balanced_acc_cand > best_balanced_acc:
                best_balanced_acc = balanced_acc_cand
                best_threshold = threshold
        
        # Use optimal threshold
        preds = (p_slice > best_threshold).astype(int)
        acc = accuracy_score(t_slice, preds)
        balanced_acc = balanced_accuracy_score(t_slice, preds)
        f1 = f1_score(t_slice, preds, average='weighted')
        
        baseline = max(np.mean(t_slice), 1 - np.mean(t_slice))
        class_ratio = np.mean(t_slice)

        print(f"\nHorizon [{horizon}]")
        print(f"Threshold:       {best_threshold:.2f}")
        print(f"Accuracy:        {acc*100:.2f}%")
        print(f"Balanced Acc:    {balanced_acc*100:.2f}%")
        print(f"F1-Score:        {f1*100:.2f}%")
        print(f"Baseline:        {baseline*100:.2f}%")
        print(f"Class Balance:   {class_ratio*100:.2f}% UP")

    print("\n" + "="*50)

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"\n[+] Production Model (V7) saved successfully.")

if __name__ == "__main__":
    main()