#!/usr/bin/env python3
import os, time
from collections import deque
from datetime import datetime
import numpy as np
import xgboost as xgb
import pywt
import matplotlib.pyplot as plt

# ===== Configuration =====
lookback_window     = 30     # input sequence size
forecast_horizon    = 15     # predict next 15 values directly
train_buffer_limit  = 120    # rolling buffer for training
retrain_interval    = 60     # seconds between retrains
log_filename        = "ml_data.log"
predictions_file    = "ml_predictions.log"
wavelet             = "haar"
wavelet_level       = 4

# ===== Globals =====
training_buffer = deque(maxlen=train_buffer_limit)
model = None
last_train_time = 0

# ===== Utility Functions =====
def tail_f(fname):
    while not os.path.exists(fname):
        time.sleep(0.1)
    with open(fname) as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line

def wavelet_feats(seq):
    coeffs = pywt.wavedec(seq, wavelet, level=wavelet_level)
    return np.concatenate(coeffs)

def build_features(seq):
    diffs = np.diff(seq)
    rolling_mean = np.mean(seq[-5:])
    rolling_std = np.std(seq[-5:])
    max_val = np.max(seq[-5:])
    min_val = np.min(seq[-5:])
    trend_1 = seq[-1] - seq[-2]
    trend_3 = seq[-1] - seq[-4]
    accel_1 = trend_1 - (seq[-2] - seq[-3])

    return np.concatenate([
        wavelet_feats(seq),
        diffs,
        [rolling_mean, rolling_std, max_val, min_val, trend_1, trend_3, accel_1]
    ])

def create_training_data(seq, lookback, horizon):
    X, Y = [], []
    for i in range(len(seq) - lookback - horizon):
        window = seq[i:i + lookback]
        targets = seq[i + lookback : i + lookback + horizon]
        X.append(build_features(np.array(window)))
        Y.append(targets)
    return np.array(X), np.array(Y)

def maybe_retrain_model():
    global model, last_train_time
    now = time.time()
    if len(training_buffer) < (lookback_window + forecast_horizon):
        return
    if now - last_train_time < retrain_interval:
        return

    X_train, y_train = create_training_data(list(training_buffer), lookback_window, forecast_horizon)
    if len(X_train) < 20:
        return

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=0
    )
    model.fit(X_train, y_train)
    last_train_time = now
    print(f"📚 Re-trained XGBoost at {datetime.now().strftime('%H:%M:%S')} on {len(X_train)} samples")

def predict_direct(model, seq):
    feats = build_features(np.array(seq[-lookback_window:]))
    return model.predict(feats.reshape(1, -1))[0]

def append_predictions(preds):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = timestamp + "," + ",".join(f"{v:.2f}" for v in preds) + "\n"
    with open(predictions_file, "a") as f:
        f.write(line)

def read_initial_points():
    if not os.path.exists(log_filename):
        return []
    with open(log_filename) as f:
        lines = [line.strip() for line in f if line.strip() and "TEST" not in line]
        values = []
        for l in lines:
            parts = l.split(",")
            if len(parts) == 3:
                try:
                    values.append(float(parts[1]))
                except:
                    pass
        return values

# ===== Bootstrap from file if available =====
print("⏳ Waiting for 30 valid CPU values to begin...")
initial_vals = deque(read_initial_points(), maxlen=train_buffer_limit)
while len(initial_vals) < lookback_window:
    time.sleep(2)
    initial_vals = deque(read_initial_points(), maxlen=train_buffer_limit)

training_buffer.extend(initial_vals)
print(f"✅ Seeded with {len(training_buffer)} values. Starting prediction loop...")

# ===== Real-time Loop =====
for line in tail_f(log_filename):
    line = line.strip()
    if not line or "TEST" in line:
        continue

    parts = line.split(",")
    if len(parts) != 3:
        continue

    try:
        cpu_raw = float(parts[1])
    except:
        continue

    training_buffer.append(cpu_raw)

    # Train periodically
    maybe_retrain_model()

    # Predict if model exists
    if model:
        preds = predict_direct(model, list(training_buffer))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received: {cpu_raw:.2f} | Next 15 → " +
              ", ".join(f"{p:.2f}" for p in preds))
        append_predictions(preds)

        # Optional Plot
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(range(-len(training_buffer), 0), list(training_buffer), label="Recent CPU")
            plt.plot(range(1, forecast_horizon + 1), preds, linestyle="--", marker="o", label="Forecast")
            plt.xlabel("Steps (10s)")
            plt.ylabel("CPU Usage (%)")
            plt.title("Real-Time 2.5 min Direct Forecast (XGBoost)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("latest_forecast.png")
            plt.close()
        except Exception as e:
            print("⚠️  Plot error:", e)

