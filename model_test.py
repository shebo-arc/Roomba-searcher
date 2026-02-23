import serial
import pandas as pd
import numpy as np
import joblib
from collections import deque
from scipy.stats import skew, kurtosis, entropy

# Load trained model and scaler
model = joblib.load("svm_gesture_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

# Serial connection
ser = serial.Serial('COM3', 115200)

# Sliding window buffer
WINDOW_SIZE = 100  # adjust depending on sampling rate
buffer = deque(maxlen=WINDOW_SIZE)

# Movement threshold (you may tune this)
MOVEMENT_THRESHOLD = 0.03


def extract_features(df):
    feats = []
    for col in ['ay', 'az', 'gx', 'gz']:
        signal = df[col]
        feats.extend([
            np.min(signal),  # 1
            np.max(signal),  # 2
            np.max(signal) - np.min(signal),  # 3 peak-to-peak
            np.median(signal),  # 4
            np.median(np.abs(signal - np.median(signal))),  # 5 MAD
            np.percentile(signal, 75) - np.percentile(signal, 25),  # 6 IQR
            np.sqrt(np.mean(signal ** 2)),  # 7 RMS
            skew(signal, nan_policy='omit'),  # 8
            kurtosis(signal, nan_policy='omit'),  # 9
            np.sum(signal ** 2) / len(signal),  # 10 normalized energy
            np.sum(np.diff(np.sign(signal)) != 0) / len(signal),  # 11 zero-crossing rate
        ])
    return np.array(feats)

prev_pred="None"
while True:
    line = ser.readline().decode().strip()

    try:
        values = list(map(float, line.split(',')))
        buffer.append(values)

        if len(buffer) == WINDOW_SIZE:
            df = pd.DataFrame(buffer, columns=['ay', 'az', 'gx', 'gz'])

            '''
            if movement_energy < MOVEMENT_THRESHOLD:
                print("Gesture: idle")
            else:
                feat = extract_features(df)
                feat = scaler.transform([feat])
                pred = model.predict(feat)
                print("Gesture:", pred[0])
            '''
            feat = extract_features(df)
            feat = scaler.transform([feat])
            pred = model.predict(feat)[0]
            prob = model.predict_proba(feat)[0]
            confidence = np.max(prob)

            if confidence > 0.85 and prev_pred != pred:
                print(f"Gesture: {pred:10}   confidence: {confidence:.3f}")
                prev_pred = pred


    except Exception as e:
        print("Error:", e)