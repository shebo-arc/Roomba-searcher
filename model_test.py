import serial
import pandas as pd
import numpy as np
import joblib
from collections import deque

# Load trained model and scaler
model = joblib.load("svm_gesture_model.pkl")
scaler = joblib.load("scaler.pkl")

# Serial connection
ser = serial.Serial('COM3', 115200)

# Sliding window buffer
WINDOW_SIZE = 50  # adjust depending on sampling rate
buffer = deque(maxlen=WINDOW_SIZE)

# Movement threshold (you may tune this)
MOVEMENT_THRESHOLD = 0.03


def extract_features(df):
    features = []
    for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        features.append(df[col].mean())
        features.append(df[col].std())
        features.append(df[col].max())
        features.append(df[col].min())
    return np.array(features)


while True:
    line = ser.readline().decode().strip()

    try:
        values = list(map(float, line.split(',')))
        buffer.append(values)

        if len(buffer) == WINDOW_SIZE:
            df = pd.DataFrame(buffer, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])

            # 🔥 Movement energy calculation
            movement_energy = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].std().mean()

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
            pred = model.predict(feat)
            print("Gesture:", pred[0])


    except Exception as e:
        print("Error:", e)
