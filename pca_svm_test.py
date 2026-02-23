import serial
import pandas as pd
import numpy as np
import joblib
from collections import deque

# ──── Load everything you trained with ────
model  = joblib.load("svm_gesture_model_smote.pkl")
scaler = joblib.load("scaler_smote.pkl")
pca    = joblib.load("pca_transformer_smote.pkl")         # ← important!

# Serial port – change if needed
PORT = "COM3"          # or "/dev/ttyUSB0", "/dev/ttyACM0", etc.
BAUD = 115200

# Must match training
WINDOW_SIZE = 75
STRIDE      = 25       # not used in real-time, but good to know
SENSORS     = ["ay", "az", "gx", "gz"]   # exactly as in training

# Movement detection threshold (tune after testing)
MOVEMENT_THRESHOLD = 5   # example – depends on your sensor scale & noise

# ──── Feature extraction – MUST match training exactly ────
def extract_features(window):
    """
    Updated feature extraction to capture:
    - Temporal relationships: autocorrelation (lags 1-3), mean abs diff, peak count & spacing, dominant freq & phase
    - Spatial relationships: correlations between channel pairs, vector norms (e.g., accel/gyro magnitude over time)
    - Keeps original stats for baseline
    """
    feats = []
    n_timesteps, n_channels = window.shape  # e.g., (100, 4)

    # ── Per-channel features (temporal focus) ──
    for col in range(n_channels):
        signal = window[:, col]

        if len(signal) < 3:
            feats.extend([0.0] * 23)  # adjust count based on new features
            continue

        # Original stats
        feats.extend([
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),  # peak-to-peak
            np.median(signal),
            np.median(np.abs(signal - np.median(signal))),  # MAD
            np.percentile(signal, 75) - np.percentile(signal, 25),  # IQR
            np.sqrt(np.mean(signal ** 2)),  # RMS
            skew(signal, nan_policy='omit'),
            kurtosis(signal, nan_policy='omit'),
            np.sum(signal ** 2) / len(signal),  # normalized energy
            np.sum(np.diff(np.sign(signal)) != 0) / len(signal),  # ZCR
        ])

        # NEW: Temporal-specific
        # Autocorrelation at lags 1,2,3 (captures sequential dependencies)
        autocorr = np.correlate(signal, signal, mode='full')[len(signal) - 1:]
        feats.append(autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0)  # lag 1
        feats.append(autocorr[2] / autocorr[0] if autocorr[0] != 0 else 0)  # lag 2
        feats.append(autocorr[3] / autocorr[0] if autocorr[0] != 0 else 0)  # lag 3

        # Mean absolute difference (temporal smoothness)
        feats.append(np.mean(np.abs(np.diff(signal))))

        # Number of peaks & average peak spacing (for wave-like patterns)
        peaks, _ = find_peaks(signal)
        feats.append(len(peaks))
        if len(peaks) > 1:
            feats.append(np.mean(np.diff(peaks)))
        else:
            feats.append(0.0)

        # FFT: dominant frequency + phase (temporal frequency & timing)
        fft_vals = fft(signal)
        fft_mag = np.abs(fft_vals[1:len(signal) // 2])
        if len(fft_mag) > 0:
            dom_idx = np.argmax(fft_mag) + 1
            dom_freq = dom_idx / len(signal)  # normalized
            dom_phase = np.angle(fft_vals[dom_idx])
            feats.append(dom_freq)
            feats.append(np.sin(dom_phase))  # sin/cos for phase
            feats.append(np.cos(dom_phase))
        else:
            feats.extend([0.0, 0.0, 0.0])

        # Signal entropy (temporal complexity)
        hist, _ = np.histogram(signal, bins=20, density=True)
        feats.append(-np.sum(hist * np.log(hist + 1e-9)))

    # ── Spatial relationships (across channels) ──
    # Pairwise Pearson correlations (captures inter-axis coordination)
    corr_matrix = np.corrcoef(window.T)
    triu_indices = np.triu_indices(n_channels, k=1)  # upper triangle, no diagonal
    for i, j in zip(*triu_indices):
        feats.append(corr_matrix[i, j])

    # Vector norms over time (spatial magnitude, e.g., accel vector)
    # Assume channels 0-1: ay/az (accel), 2-3: gx/gz (gyro)
    accel_norm = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2)  # ay/az norm
    gyro_norm = np.sqrt(window[:, 2] ** 2 + window[:, 3] ** 2)  # gx/gz norm

    # Stats on norms (temporal + spatial)
    for norm_sig in [accel_norm, gyro_norm]:
        feats.extend([
            np.mean(norm_sig),  # mean magnitude
            np.std(norm_sig),  # variation in magnitude
            np.max(norm_sig),  # peak magnitude
            skew(norm_sig, nan_policy='omit'),  # asymmetry in magnitude
        ])

    # Cross-channel deltas (e.g., ay - az, gx - gz)
    feats.append(np.mean(window[:, 0] - window[:, 1]))  # ay - az
    feats.append(np.mean(window[:, 2] - window[:, 3]))  # gx - gz

    return np.array(feats)

# ──── Initialize ────
ser = serial.Serial(PORT, BAUD, timeout=0.1)
buffer = deque(maxlen=WINDOW_SIZE)

print("Starting real-time gesture recognition...")
print(f"Window size = {WINDOW_SIZE}, expecting sensors: {SENSORS}")
print("Press Ctrl+C to stop\n")
prev_pred = "None"

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        try:
            # Expecting exactly 4 values: ay,az,gx,gz
            values = list(map(float, line.split(',')))
            if len(values) != len(SENSORS):
                print(f"→ Invalid sample (got {len(values)} values, expected {len(SENSORS)})")
                continue

            buffer.append(values)

            if len(buffer) == WINDOW_SIZE:
                # Convert to numpy array [WINDOW_SIZE, n_sensors]
                window = np.array(buffer)

                # Quick movement detection (optional but recommended)
                movement_energy = np.mean(np.std(window, axis=0))
                if movement_energy < MOVEMENT_THRESHOLD and prev_pred!="idle":
                    print("→ idle (low movement)")
                    prev_pred = "idle"
                else:
                    # Extract features → scale → PCA → predict
                    feat_raw = extract_features(window)           # shape (44,)
                    feat_scaled = scaler.transform([feat_raw])    # (1, 44)
                    feat_pca = pca.transform(feat_scaled)         # (1, n_components)

                    pred = model.predict(feat_pca)[0]
                    prob = model.predict_proba(feat_pca)[0]
                    confidence = np.max(prob)
                    if confidence>0.90 and prev_pred!=pred:
                        print(f"→ Gesture: {pred:8}   (energy: {movement_energy:.4f})")
                        prev_pred = pred

        except ValueError:
            print(f"→ Parse error: {line}")
        except Exception as e:
            print(f"→ Error: {e}")

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    ser.close()
    print("Serial port closed.")