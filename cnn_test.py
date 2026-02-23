import serial
import numpy as np
import tensorflow as tf
from collections import deque
import time
from scipy import signal
import joblib

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
MODEL_PATH     = "cnn_gesture_model.h5"
LABEL_ENCODER  = "label_encoder.pkl"

SERIAL_PORT    = "COM3"
BAUD_RATE      = 115200

WINDOW_SIZE    = 75

SAMPLING_RATE  = 50.0          # MUST MATCH TRAINING
CUTOFF_FREQ    = 12.0
Wn = CUTOFF_FREQ / (SAMPLING_RATE / 2)
b, a = signal.butter(1, Wn, btype='low', analog=False, output='ba')

MOVEMENT_THRESHOLD    = 0.03
CONFIDENCE_THRESHOLD  = 0.85
# ────────────────────────────────────────────────────────────────

# Load model and label encoder
print("Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_ENCODER)
class_names = le.classes_
print("Model and label encoder loaded.")

# Pre-compute filter initial conditions (one per channel)
zi = np.zeros((4, len(a) - 1))  # 4 channels

# Serial setup
print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print("Serial port opened.")
except Exception as e:
    print(f"Failed to open serial: {e}")
    exit(1)

buffer = deque(maxlen=WINDOW_SIZE)
prev_pred = "None"

print("Starting real-time prediction. Press Ctrl+C to stop.\n")

while True:
    line = ser.readline().decode().strip()

    try:
        values = list(map(float, line.split(',')))
        if len(values) != 4:
            print(f"Warning: expected 4 values, got {len(values)}")
            continue

        # Apply filter to new sample (per channel)
        filtered_values = np.zeros(4)
        for ch in range(4):
            y, zi[ch] = signal.lfilter(b, a, [values[ch]], zi=zi[ch])
            filtered_values[ch] = y[0]

        buffer.append(filtered_values)

        if len(buffer) == WINDOW_SIZE:
            window = np.array(buffer, dtype=np.float32)

            start_time = time.perf_counter()

            pred_prob = model.predict(window[np.newaxis, ...], verbose=0)[0]
            pred_idx = np.argmax(pred_prob)
            confidence = np.max(pred_prob)
            pred_label = class_names[pred_idx]

            end_time = time.perf_counter()
            inference_ms = (end_time - start_time) * 1000

            if confidence > CONFIDENCE_THRESHOLD and prev_pred != pred_label:
                print(f"Gesture: {pred_label:10}   conf: {confidence:.3f} time: {inference_ms:.2f} ms")
                prev_pred = pred_label

    except Exception as e:
        print(f"Error: {e}")