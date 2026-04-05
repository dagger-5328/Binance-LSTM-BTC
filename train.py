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
from sklearn.metrics import accuracy_score

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

# ---------------- MODEL ----------------
def build_model(input_shape):
    """Stable LSTM with high regularization (Dropout + L2) for professional generalization."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, 
             kernel_regularizer=l2(0.001)),
        Dropout(0.4),

        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.4),

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

        df_raw = fetch_data(symbol, limit=3000)
        df = apply_market_features(df_raw, cid, None)

        split = int(len(df) * TRAIN_SPLIT)
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

    # ---------------- TRAIN ----------------
    print(f"\n[*] Training on {len(X_train)} samples...")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------- EVALUATION ----------------
    print("\n" + "="*50)
    print("  FINAL PERFORMANCE REPORT (LEAKAGE-FREE)")
    print("="*50)

    probs = model.predict(X_test, verbose=0)
    horizons = ["3d", "7d"]

    for idx, horizon in enumerate(horizons):
        p_slice = probs[:, idx]
        t_slice = y_test[:, idx]
        preds = (p_slice > 0.5).astype(int)
        acc = accuracy_score(t_slice, preds)
        
        baseline = max(np.mean(t_slice), 1 - np.mean(t_slice))
        class_ratio = np.mean(t_slice)

        print(f"\nHorizon [{horizon}]")
        print(f"Accuracy:        {acc*100:.2f}%")
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