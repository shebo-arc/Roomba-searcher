import numpy as np
import pandas as pd
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.signal import find_peaks

# ──── Optional: Uncomment only one balancing method at a time ────
BALANCING_METHOD = "none"  # Options: "none", "class_weight", "smote"
# BALANCING_METHOD = "class_weight"
# BALANCING_METHOD = "smote"

# ---------------- CONFIG ----------------
WINDOW_SIZE = 75
STRIDE = 50
FEATURES = ["ay", "az", "gx", "gz"]
GESTURES = ["forward", "left", "right", "stop"]

OUTPUT_FEATURES_CSV = "gesture_window_features.csv"
MODEL_FILE = f"svm_gesture_model_{BALANCING_METHOD}.pkl"
SCALER_FILE = f"scaler_{BALANCING_METHOD}.pkl"
PCA_FILE = f"pca_transformer_{BALANCING_METHOD}.pkl"
# ----------------------------------------

# Only import SMOTE if needed
if BALANCING_METHOD == "smote":
    from imblearn.over_sampling import SMOTE


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


# ────────────────────────────────────────────────
#     1. Collect windows & extract features
# ────────────────────────────────────────────────

X = []
y = []

files = glob.glob("data/*.csv")
print(f"Found {len(files)} CSV files in 'data/' folder...")

for file in files:
    df = pd.read_csv(file)
    df = df[df["label"].isin(GESTURES)]

    for gesture in GESTURES:
        gdf = df[df["label"] == gesture][FEATURES].values

        for i in range(0, len(gdf) - WINDOW_SIZE + 1, STRIDE):
            window = gdf[i:i + WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                feats = extract_features(window)
                X.append(feats)
                y.append(gesture)

X = np.array(X)
y = np.array(y)

print(f"Total windows extracted: {len(X):,}")
print(f"Features per window: {X.shape[1]}")

print("\nOriginal class distribution:")
print(pd.Series(y).value_counts().sort_index())

# ────────────────────────────────────────────────
#     2. Save original features (before any balancing)
# ────────────────────────────────────────────────
'''
stat_names = ["min", "max", "ptp", "median", "mad", "iqr", "rms", "skew", "kurt", "energy", "zcr",
              "autocorr_lag1", "autocorr_lag2", "autocorr_lag3", "mean_abs_diff", "num_peaks", "peak_spacing", "dom_freq", "phase_sin", "phase_cos", "entropy"]
feature_columns = [f"{sig}_{stat}" for sig in FEATURES for stat in stat_names]

df_features = pd.DataFrame(X, columns=feature_columns)
df_features["label"] = y
df_features.to_csv(OUTPUT_FEATURES_CSV, index=False)
print(f"Saved original features → {OUTPUT_FEATURES_CSV}\n")
'''
# ────────────────────────────────────────────────
#     3. Train/test split
# ────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ────────────────────────────────────────────────
#     4. Scaling
# ────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ────────────────────────────────────────────────
#     5. Optional balancing (SMOTE or nothing here)
# ────────────────────────────────────────────────

X_train_bal = X_train_scaled
y_train_bal = y_train

if BALANCING_METHOD == "smote":
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    # Save SMOTE version (with nice column names)
    stat_names = ["min", "max", "ptp", "median", "mad", "iqr", "rms", "skew", "kurt", "energy", "zcr"]
    feature_columns = [f"{sig}_{stat}" for sig in FEATURES for stat in stat_names]

    df_smote = pd.DataFrame(X_train_bal, columns=feature_columns)
    df_smote["label"] = y_train_bal

    smote_path = f"gesture_features_smote_{len(y_train_bal)}_samples.csv"
    df_smote.to_csv(smote_path, index=False)
    print(f"Saved SMOTE-balanced training set → {smote_path}")
    print(f"Shape: {X_train_bal.shape} | Classes:\n{pd.Series(y_train_bal).value_counts().sort_index()}\n")

elif BALANCING_METHOD == "none":
    print("No balancing applied.")

# ────────────────────────────────────────────────
#     6. PCA
# ────────────────────────────────────────────────

print("\n=== PCA Dimensionality Reduction ===")
print(f"Before PCA → {X_train_bal.shape[1]} features")

pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca = pca.transform(X_test_scaled)

n_components = X_train_pca.shape[1]
explained_var = pca.explained_variance_ratio_.sum() * 100

print(f"After PCA  → {n_components} components")
print(f"→ Kept {explained_var:.2f}% of variance")
print(f"→ Reduction: {X_train_bal.shape[1] - n_components} features "
      f"({(X_train_bal.shape[1] - n_components) / X_train_bal.shape[1] * 100:.1f}%)\n")

# ────────────────────────────────────────────────
#     7. Train SVM
# ────────────────────────────────────────────────

class_weight_param = 'balanced' if BALANCING_METHOD == "class_weight" else None

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    random_state=42,
    class_weight=class_weight_param,
    probability=True
)

print(f"Training SVM (balancing: {BALANCING_METHOD})...")
model.fit(X_train_pca, y_train_bal)
print("Training complete.\n")

# ────────────────────────────────────────────────
#     8. Evaluation
# ────────────────────────────────────────────────

y_pred = model.predict(X_train_pca)

print("Classification Report:")
print(classification_report(y_train_bal, y_pred, digits=3, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_train_bal, y_pred))

# ────────────────────────────────────────────────
#     9. Save models
# ────────────────────────────────────────────────

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(pca, PCA_FILE)

print(f"\nSaved files with suffix _{BALANCING_METHOD}:")
print(f"  • Model  → {MODEL_FILE}")
print(f"  • Scaler → {SCALER_FILE}")
print(f"  • PCA    → {PCA_FILE}")
print("Done!")