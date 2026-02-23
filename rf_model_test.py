import serial
import pandas as pd
import numpy as np
import joblib
from collections import deque
from scipy.stats import skew, kurtosis

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
MODEL_PATH = "rf_gesture_model.pkl"  # ← your new Random Forest model
SCALER_PATH = "rf_scaler.pkl"  # ← the final scaler trained on all data

SERIAL_PORT = "COM3"  # change if needed (e.g. '/dev/ttyUSB0' on Linux/Mac)
BAUD_RATE = 115200

WINDOW_SIZE = 50  # should match your training window size
STRIDE = 25  # for real-time: usually 1 (predict every new sample)

# Only the 4 best channels you selected
SENSOR_COLUMNS = ['ay', 'az', 'gx', 'gz']

# Threshold to detect "enough movement" before trying to classify
MOVEMENT_THRESHOLD = 0.03  # tune this (0.02–0.08 range is common)

# ────────────────────────────────────────────────────────────────
# Load model & scaler
# ────────────────────────────────────────────────────────────────
print("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model and scaler loaded successfully.")


# ────────────────────────────────────────────────────────────────
# Feature extraction function (must match exactly what you used in training)
# ────────────────────────────────────────────────────────────────
def extract_features(df):
    """
    Extract the same 11 features per channel as in training.
    Expects df with columns: ay, az, gx, gz
    Returns: numpy array of shape (44,)
    """
    features = []
    for col in SENSOR_COLUMNS:
        signal = df[col].to_numpy()

        if len(signal) < 3:
            features.extend([0.0] * 11)
            continue

        features.extend([
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
            np.sum(np.diff(np.sign(signal)) != 0) / len(signal),  # 11 ZCR
        ])

    return np.array(features)


# ────────────────────────────────────────────────────────────────
# Serial + sliding window setup
# ────────────────────────────────────────────────────────────────
print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Serial port opened.")
except Exception as e:
    print(f"Failed to open serial port: {e}")
    exit(1)

# Buffer for raw values (list of lists → will become DataFrame)
buffer = deque(maxlen=WINDOW_SIZE)

print("Starting real-time prediction. Press Ctrl+C to stop.\n")
prev_pred="None"

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        try:
            # Assuming incoming format: ay,az,gx,gz  (4 values only)
            values = list(map(float, line.split(',')))

            if len(values) != 4:
                print(f"Warning: expected 4 values, got {len(values)} → skipping")
                continue

            buffer.append(values)

            # Only predict when buffer is full
            if len(buffer) == WINDOW_SIZE:
                # Create DataFrame with correct column names
                df = pd.DataFrame(buffer, columns=SENSOR_COLUMNS)

                # Compute average movement energy (optional filter)
                movement_energy = df.std().mean()

                if movement_energy < MOVEMENT_THRESHOLD and prev_pred!="idle":
                    print(f"Low movement ({movement_energy:.4f}) → no gesture")
                    prev_pred="idle"
                    continue

                # Extract features (same as training)
                feat = extract_features(df)

                # Scale (very important!)
                feat_scaled = scaler.transform([feat])  # shape (1, 44)

                # Predict
                pred = model.predict(feat_scaled)[0]
                prob = model.predict_proba(feat_scaled)[0]
                confidence = np.max(prob)

                if confidence > 0.85 and prev_pred != pred:
                    print(f"Gesture: {pred:10}   confidence: {confidence:.3f}   energy: {movement_energy:.4f}")
                    prev_pred = pred

        except ValueError:
            print(f"Invalid data: {line}")
        except Exception as e:
            print(f"Processing error: {e}")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    ser.close()
    print("Serial port closed.")