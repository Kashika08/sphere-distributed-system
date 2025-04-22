#!/usr/bin/env python3
"""
Real‑time CPU forecaster:
  • Holt‑Winters  → writes predictions to ml_predictions.log
  • Prophet       → prints predictions for comparison
  • SARIMAX       → prints predictions for comparison
Keeps console & log formats identical to previous versions.
"""

import os, time, warnings
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet                         # pip install prophet
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- CONFIGURATION -------------------- #
forecast_horizon    = 15        # 15 steps (2.5 min)
train_buffer_limit  = 60        # 60 pts  (10 min @10 s)
initial_wait_pts    = 20        # start after 20 pts (~3.3 min)
retrain_interval    = 60        # seconds
log_filename        = "ml_data.log"
predictions_file    = "ml_predictions.log"

# -------------------- GLOBALS -------------------------- #
buf          = deque(maxlen=train_buffer_limit)
hw_model     = None
sarima_model = None
prophet_model = None
last_train_time = 0

# -------------------- UTILITIES ------------------------ #
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

def append_predictions(preds):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = timestamp + "," + ",".join(f"{v:.2f}" for v in preds) + "\n"
    with open(predictions_file, "a") as f:
        f.write(line)

def read_initial_vals():
    vals = []
    if not os.path.exists(log_filename):
        return vals
    with open(log_filename) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
    return vals

# ---------- MODEL TRAIN / FORECAST HELPERS ------------- #
def fit_holtwinters(series):
    return ExponentialSmoothing(series, trend="add",
                                seasonal=None,
                                initialization_method="estimated"
                                ).fit(optimized=True)

def fit_sarima(series):
    return SARIMAX(series, order=(1,1,1), seasonal_order=(0,0,0,0),
                   enforce_stationarity=False, enforce_invertibility=False
                   ).fit(disp=False)

def fit_prophet(series):
    now = datetime.utcnow()
    start_ts = now - timedelta(seconds=10*(len(series)-1))
    ds = [start_ts + timedelta(seconds=10*i) for i in range(len(series))]
    df = pd.DataFrame({"ds": ds, "y": series})
    m = Prophet(interval_width=0.8, daily_seasonality=False,
                weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=forecast_horizon, freq="10S")
    forecast = m.predict(future).yhat.iloc[-forecast_horizon:].values
    return m, forecast

def retrain_models():
    global hw_model, sarima_model, prophet_model, last_train_time
    now = time.time()
    if len(buf) < initial_wait_pts or (now - last_train_time < retrain_interval):
        return

    series = pd.Series(list(buf))

    hw_model      = fit_holtwinters(series)
    sarima_model  = fit_sarima(series)
    prophet_model, _ = fit_prophet(series)           # store the fitted Prophet

    last_train_time = now
    print(f"📚 Models retrained @ {datetime.now().strftime('%H:%M:%S')} "
          f"on {len(series)} pts")

def forecast_all():
    # Holt‑Winters
    hw_preds = hw_model.forecast(forecast_horizon).values

    # SARIMA
    # sarima_preds = sarima_model.get_forecast(steps=forecast_horizon
    #                ).predicted_mean.values

    # Prophet
    now = datetime.utcnow()
    start_ts = now - timedelta(seconds=10*(len(buf)-1))
    df = pd.DataFrame({
        "ds": [start_ts + timedelta(seconds=10*i) for i in range(len(buf))],
        "y":  list(buf)
    })
    # future = prophet_model.make_future_dataframe(periods=forecast_horizon,
    #                                             freq="10S")
    # prophet_preds = prophet_model.predict(future).yhat.iloc[-forecast_horizon:].values

    return hw_preds

# -------------------- BOOTSTRAP ------------------------ #
print("⏳ Waiting for first", initial_wait_pts, "CPU points …")
buf.extend(read_initial_vals())
while len(buf) < initial_wait_pts:
    time.sleep(1)
    buf.extend(read_initial_vals()[-initial_wait_pts:])   # top‑up if new rows arrived
print(f"✅ Seeded with {len(buf)} points. Entering live loop …")

# -------------------- MAIN LOOP ------------------------ #
for line in tail_f(log_filename):
    parts = line.strip().split(",")
    if len(parts) != 3:
        continue
    try:
        cpu_raw = float(parts[1])
    except ValueError:
        continue

    buf.append(cpu_raw)
    retrain_models()

    if hw_model:
        hw_preds = forecast_all()

        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Received: {cpu_raw:.2f} | Next 15 HoltWinters → "
              + ", ".join(f"{p:.2f}" for p in hw_preds))
        # print(f"[{now_str}] Received: {cpu_raw:.2f} | Next 15 Prophet     → "
        #      + ", ".join(f"{p:.2f}" for p in prop_preds))
        # print(f"[{now_str}] Received: {cpu_raw:.2f} | Next 15 SARIMA      → "
        #      + ", ".join(f"{p:.2f}" for p in sarima_preds))

        # Log ONLY Holt‑Winters forecasts
        append_predictions(hw_preds)

        # Optional plot (Holt‑Winters)
        try:
            plt.figure(figsize=(10,4))
            plt.plot(range(-len(buf), 0), list(buf), label="CPU")
            plt.plot(range(1, forecast_horizon+1), hw_preds,
                     marker='o', linestyle='--', label="HW Forecast")
            plt.xlabel("Steps (10 s)"); plt.ylabel("CPU %")
            plt.title("Real‑Time 2.5 min Forecast – HW")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig("latest_forecast.png"); plt.close()
        except Exception as e:
            print("⚠️ Plot error:", e)

