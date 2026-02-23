import serial
import pandas as pd
import numpy as np
import joblib
from collections import deque
from scipy.stats import skew, kurtosis
import time  # ← added for timing

# Load trained model and scaler
model = joblib.load("svm_direct_model.pkl")
scaler = joblib.load("svm_direct_scaler.pkl")

# Serial connection
ser = serial.Serial('COM3', 115200)

# Sliding window buffer
WINDOW_SIZE = 75
buffer = deque(maxlen=WINDOW_SIZE)

# Movement threshold (you may tune this)
MOVEMENT_THRESHOLD = 0.03

def extract_features(df):
    feats = []
    for col in ['ay', 'az', 'gx', 'gz']:
        signal = df[col]
        feats.extend([
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),
            # np.median(signal),
            # np.median(np.abs(signal - np.median(signal))),
            # np.percentile(signal, 75) - np.percentile(signal, 25),
            np.sqrt(np.mean(signal ** 2)),
            skew(signal, nan_policy='omit'),
            # kurtosis(signal, nan_policy='omit'),
            np.sum(signal ** 2) / len(signal),
            np.sum(np.diff(np.sign(signal)) != 0) / len(signal),
        ])
    return np.array(feats)

prev_pred = "None"

while True:
    line = ser.readline().decode().strip()

    try:
        values = list(map(float, line.split(',')))
        buffer.append(values)

        if len(buffer) == WINDOW_SIZE:
            df = pd.DataFrame(buffer, columns=['ay', 'az', 'gx', 'gz'])

            # Start timing
            start_time = time.perf_counter()

            feat = extract_features(df)
            feat_scaled = scaler.transform([feat])

            pred = model.predict(feat_scaled)[0]
            prob = model.predict_proba(feat_scaled)[0]
            confidence = np.max(prob)

            # End timing
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000  # convert to milliseconds

            if confidence > 0.85 and prev_pred != pred:
                print(f"Gesture: {pred:10}   confidence: {confidence:.3f}   "
                      f"inference time: {inference_time_ms:.2f} ms")
                prev_pred = pred

    except Exception as e:
        print("Error:", e)