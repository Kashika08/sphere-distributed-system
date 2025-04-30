import numpy as np
import time
import os
import sys
import pywt
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ----- Configuration Parameters -----
sequence_length = 30          # Window length (input)
forecast_horizon = 15         # Forecast 15 steps ahead
min_buffer_size = 6           # Retrain every 6 new points
log_filename = "ml_data.log"  # Input log
predictions_filename = "ml_predictions.log"  # Output log
wavelet = 'haar'              # Wavelet basis
level = 4                     # Decomposition level

# Global sliding window and buffer
window = deque(maxlen=sequence_length)
new_data_buffer = []

# ----- Wavelet Utilities -----
def decompose_signal(signal, wavelet, level):
    return pywt.wavedec(signal, wavelet, level=level)

def prepare_wavelet_sample(sequence, wavelet, level):
    coeffs = decompose_signal(sequence, wavelet, level)
    flat = np.concatenate([c for c in coeffs])
    return flat, coeffs

def reconstruct_signal(predicted_flat, coeffs_template, wavelet):
    coeffs = []
    start = 0
    for c in coeffs_template:
        length = len(c)
        coeffs.append(predicted_flat[start:start+length])
        start += length
    return pywt.waverec(coeffs, wavelet)

# ----- Initial Setup -----
# Create coeffs_template and expected output shape for forecast reconstruction
dummy_forecast = np.zeros(forecast_horizon)
_, coeffs_template = prepare_wavelet_sample(dummy_forecast, wavelet, level)
output_wavelet_length = sum(len(c) for c in coeffs_template)

# ----- Read Initial Window from Log -----
def get_initial_window(filename, seq_length):
    if not os.path.exists(filename):
        print(f"{filename} not found. Initializing window with zeros.")
        return np.zeros(seq_length)
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.zeros(seq_length)

    values = []
    for line in lines[-seq_length:]:
        if "TEST,TEST,TEST" in line:
            continue
        fields = line.strip().split(',')
        if len(fields) != 3:
            continue
        try:
            values.append(float(fields[1]))
        except:
            continue

    if len(values) < seq_length:
        values = [0.0] * (seq_length - len(values)) + values
    return np.array(values)

# ----- Tail the Log File -----
def tail_f(filename):
    while not os.path.exists(filename):
        print(f"Waiting for {filename} to be created...")
        time.sleep(1)
    with open(filename, 'r') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            yield line

# ----- Update Model -----
def update_model_with_new_data(new_data):
    global new_data_buffer, window, model
    for val in new_data:
        window.append(val)
        new_data_buffer.append(val)

    if len(new_data_buffer) >= min_buffer_size:
        recent = np.array(window)
        combined = np.concatenate([recent, new_data_buffer])
        if len(combined) < sequence_length + forecast_horizon:
            return

        X_new, y_new = [], []
        for i in range(len(combined) - sequence_length - forecast_horizon + 1):
            X_new.append(combined[i:i+sequence_length])
            y_new.append(combined[i+sequence_length:i+sequence_length+forecast_horizon])

        X_wavelet, y_wavelet = [], []
        for x, y in zip(X_new, y_new):
            flat_x, _ = prepare_wavelet_sample(x, wavelet, level)
            flat_y, _ = prepare_wavelet_sample(y, wavelet, level)
            if len(flat_y) != output_wavelet_length:
                continue
            X_wavelet.append(flat_x)
            y_wavelet.append(flat_y)

        X_wavelet = np.array(X_wavelet)
        y_wavelet = np.array(y_wavelet)

        if len(X_wavelet) == 0:
            print("No valid samples for training (shape mismatch).")
            return

        print(f"Training on {X_wavelet.shape[0]} samples with {len(new_data_buffer)} new data points.")
        model.fit(X_wavelet, y_wavelet, epochs=2, batch_size=32, verbose=1)
        new_data_buffer.clear()
    else:
        print(f"Accumulated {len(new_data_buffer)} new data points; waiting for {min_buffer_size}...")

# ----- Forecast -----
def multi_step_forecast(steps_ahead=forecast_horizon):
    current_window = np.array(window)
    input_wavelet, _ = prepare_wavelet_sample(current_window, wavelet, level)
    input_wavelet = input_wavelet.reshape(1, -1)
    predicted_wavelet = model.predict(input_wavelet, verbose=0)[0]

    if len(predicted_wavelet) != output_wavelet_length:
        print(f"[ERROR] Predicted wavelet length mismatch: {len(predicted_wavelet)} vs expected {output_wavelet_length}")
        return np.zeros(steps_ahead)

    predicted_signal = reconstruct_signal(predicted_wavelet, coeffs_template, wavelet)
    return predicted_signal[:steps_ahead]

# ----- Append Forecast to File -----
def append_predictions_to_file(forecast):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    forecast_str = ",".join(f"{x:.4f}" for x in forecast)
    try:
        with open(predictions_filename, "a") as f:
            f.write(f"{timestamp},{forecast_str}\n")
    except Exception as e:
        print(f"Error writing to prediction file: {e}")

# ----- Main Loop -----
def main():
    global model, window, new_data_buffer

    # Load model
    try:
        model = load_model("wavelet_autoscaler.h5", compile=False)
    except Exception as e:
        sys.exit(f"Failed to load model: {e}")

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    print(f"Model output shape: {model.output_shape}")
    print(f"Expected wavelet output length: {output_wavelet_length}")

    initial_window = get_initial_window(log_filename, sequence_length)
    window = deque(initial_window, maxlen=sequence_length)
    new_data_buffer = []
    print(f"Initial sliding window: {list(window)}")

    for line in tail_f(log_filename):
        line = line.strip()
        if not line or "TEST,TEST,TEST" in line:
            continue
        fields = line.split(',')
        if len(fields) != 3:
            continue
        try:
            cpu_val = float(fields[1])
        except:
            continue

        print(f"New CPU usage received: {cpu_val}%")
        update_model_with_new_data([cpu_val])

        forecast = multi_step_forecast()
        print(f"Forecast for next {forecast_horizon} steps: {forecast}")
        append_predictions_to_file(forecast)

        # Save plot
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(-sequence_length, 0), list(window), label="Recent CPU Usage")
        plt.plot(np.arange(1, forecast_horizon+1), forecast, marker='o', linestyle='--', label="Forecast")
        plt.xlabel("Time Steps (relative)")
        plt.ylabel("CPU Usage (%)")
        plt.title("Real-Time Forecast after Incremental Update")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("latest_forecast.png")
        plt.close()

if __name__ == '__main__':
    main()

