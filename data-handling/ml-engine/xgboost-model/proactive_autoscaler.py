import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import deque

# ========== CONFIGURATION ==========
SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 12
INIT_TRAIN_POINTS = 36
RETRAIN_FREQUENCY = 30
LOG_FILE = "ml_data.log"
HISTORY_FILE = "ml_history.log"

# ========== OUTPUT DIRECTORIES ==========
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# ========== OUTPUT FILES ==========
PREDICTION_FILE = os.path.join(LOG_DIR, "ml_predictions.log")
EVAL_LOG_FILE = os.path.join(LOG_DIR, "xgboost_eval.log")
T120_EVAL_LOG_FILE = os.path.join(LOG_DIR, "t120_evaluation.log")

# ========== GLOBAL STATE ==========
BUFFER = []
WINDOW = deque(maxlen=SEQUENCE_LENGTH)
XGBOOST_MODEL = None
LAST_TRAIN_SIZE = 0
HISTORICAL_PREDICTIONS = {}

# ========== MODEL FUNCTIONS ==========
def build_xgboost():
    return XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42
    )

def save_model():
    if XGBOOST_MODEL:
        model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        joblib.dump(XGBOOST_MODEL, model_path)
        if hasattr(XGBOOST_MODEL, 'get_booster'):
            booster = XGBOOST_MODEL.get_booster()
            booster.save_model(os.path.join(MODEL_DIR, "xgboost_model_raw.json"))
        print(f"📦 Model saved to {model_path}")

def load_model():
    global XGBOOST_MODEL
    model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    if os.path.exists(model_path):
        XGBOOST_MODEL = joblib.load(model_path)
        print("✅ XGBoost model loaded from disk")
        return True
    return False

def train_xgboost(incremental=False):
    global XGBOOST_MODEL, LAST_TRAIN_SIZE
    if len(BUFFER) < SEQUENCE_LENGTH + FORECAST_HORIZON:
        print("⚠️ Not enough data to train. Need at least", SEQUENCE_LENGTH + FORECAST_HORIZON)
        return False

    X, y = [], []
    series = np.array(BUFFER)
    for i in range(0, len(series) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1):
        X.append(series[i:i + SEQUENCE_LENGTH])
        y.append(series[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON])
    X, y = np.array(X), np.array(y)

    try:
        if incremental and isinstance(XGBOOST_MODEL, xgb.Booster):
            dtrain = xgb.DMatrix(X, y)
            update_params = {
                'process_type': 'update',
                'updater': 'refresh',
                'refresh_leaf': True,
                'objective': 'reg:squarederror'
            }
            raw_model_path = os.path.join(MODEL_DIR, "xgboost_model_raw.json")
            bst = xgb.Booster({'nthread': 4}, model_file=raw_model_path)
            XGBOOST_MODEL = xgb.train(update_params, dtrain, num_boost_round=10, xgb_model=bst)
            XGBOOST_MODEL.save_model(raw_model_path)
        else:
            XGBOOST_MODEL = build_xgboost()
            XGBOOST_MODEL.fit(X, y)
            booster = XGBOOST_MODEL.get_booster()
            booster.save_model(os.path.join(MODEL_DIR, "xgboost_model_raw.json"))
        LAST_TRAIN_SIZE = len(BUFFER)
        print(f"✅ {'Incremental' if incremental else 'Initial'} training done on {X.shape[0]} samples.")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def generate_forecast():
    if XGBOOST_MODEL is None or len(BUFFER) < SEQUENCE_LENGTH:
        print("⚠️ Cannot forecast: No model or insufficient data.")
        return None
    try:
        x_input = np.array(BUFFER[-SEQUENCE_LENGTH:]).reshape(1, -1)
        if isinstance(XGBOOST_MODEL, xgb.Booster):
            dmatrix = xgb.DMatrix(x_input)
            pred = XGBOOST_MODEL.predict(dmatrix)
        else:
            pred = XGBOOST_MODEL.predict(x_input)
        return pred[0][:FORECAST_HORIZON] if len(pred[0]) >= FORECAST_HORIZON else np.pad(pred[0], (0, FORECAST_HORIZON - len(pred[0])), 'edge')
    except Exception as e:
        print(f"❌ Forecast error: {e}")
        return np.zeros(FORECAST_HORIZON)

# ========== DATA HELPERS ==========
def parse_cpu(line):
    parts = line.strip().split(',')
    try:
        return float(parts[1])  # Assuming second column is CPU
    except:
        return None

def tail_log(skip_existing=False):
    while not os.path.exists(LOG_FILE):
        print(f"⏳ Waiting for {LOG_FILE}...")
        time.sleep(1)
    with open(LOG_FILE, 'r') as f:
        if skip_existing:
            f.seek(0, os.SEEK_END)
        else:
            for line in f:
                yield line.strip()
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            yield line.strip()

# ========== LOGGING AND VISUALIZATION ==========
def log_prediction(forecast, current_idx):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}," + ",".join([f"{x:.2f}" for x in forecast]) + "\n"
    HISTORICAL_PREDICTIONS[current_idx] = forecast[-1]
    with open(PREDICTION_FILE, 'a') as f:
        f.write(line)

def log_evaluation(forecast, actual):
    if len(forecast) != len(actual): 
        min_len = min(len(forecast), len(actual))
        forecast = forecast[:min_len]
        actual = actual[:min_len]
        if min_len == 0:
            return
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    horizons = [1, 3, 6, 12]
    horizon_errors = {}
    for h in horizons:
        if h <= len(forecast):
            horizon_errors[f"MSE_{h}"] = mean_squared_error(actual[:h], forecast[:h])
            horizon_errors[f"MAE_{h}"] = mean_absolute_error(actual[:h], forecast[:h])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "timestamp": ts,
        "MSE": mse,
        "MAE": mae,
        **horizon_errors
    }
    with open(EVAL_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_data) + "\n")
    return mse

def log_t120_evaluation(current_idx, actual_value):
    prediction_idx = current_idx - FORECAST_HORIZON
    if prediction_idx < 0 or prediction_idx not in HISTORICAL_PREDICTIONS:
        return
    predicted = HISTORICAL_PREDICTIONS[prediction_idx]
    error = abs(actual_value - predicted)
    relative_error = (error / actual_value) * 100 if actual_value != 0 else 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "timestamp": ts,
        "actual_idx": current_idx,
        "prediction_idx": prediction_idx,
        "actual_value": actual_value,
        "predicted_value": predicted,
        "error": error,
        "relative_error": relative_error
    }
    with open(T120_EVAL_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_data) + "\n")

# ========== MAIN LOOP ==========
def run():
    global BUFFER, LAST_TRAIN_SIZE, WINDOW
    print("🚀 Starting XGBoost forecaster...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    load_model()

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                cpu = parse_cpu(line.strip())
                if cpu is not None:
                    BUFFER.append(cpu)
        print(f"📂 Loaded {len(BUFFER)} data points from {LOG_FILE}")

    if len(BUFFER) >= INIT_TRAIN_POINTS:
        print("🔍 Training model on existing data...")
        train_xgboost(incremental=False)
        save_model()

    WINDOW = deque(BUFFER[-SEQUENCE_LENGTH:], maxlen=SEQUENCE_LENGTH)
    predictions = []

    print("📡 Now tailing new data...")
    for line in tail_log(skip_existing=True):
        cpu = parse_cpu(line)
        if cpu is None:
            continue

        BUFFER.append(cpu)
        WINDOW.append(cpu)
        current_idx = len(BUFFER) - 1

        if current_idx >= FORECAST_HORIZON:
            log_t120_evaluation(current_idx, cpu)

        if len(BUFFER) - LAST_TRAIN_SIZE >= RETRAIN_FREQUENCY:
            print(f"♻️ Retraining after {RETRAIN_FREQUENCY} new points...")
            train_xgboost(incremental=True)
            save_model()

        if len(WINDOW) >= SEQUENCE_LENGTH:
            forecast_result = generate_forecast()
            if forecast_result is not None:
                print(f"⏱️ {datetime.now().strftime('%H:%M:%S')} | CPU: {cpu:.2f} | Forecast: {forecast_result}")
                log_prediction(forecast_result, current_idx)
                predictions.append((current_idx, forecast_result))

        for idx, fc in list(predictions):
            if current_idx >= idx + FORECAST_HORIZON:
                actual = BUFFER[idx + 1:idx + FORECAST_HORIZON + 1]
                log_evaluation(fc, actual)
                predictions.remove((idx, fc))

# ========== ENTRY POINT ==========
def main():
    try:
        run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        save_model()
        print("✅ Exiting... model saved.")

if __name__ == "__main__":
    main()
