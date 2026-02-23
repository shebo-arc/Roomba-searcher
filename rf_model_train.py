import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

# ---------------- CONFIG ----------------
WINDOW_SIZE = 50
STRIDE = 25
FEATURES = ["ay", "az", "gx", "gz"]  # ← best channels according to your ANOVA/MI
GESTURES = ["idle", "forward", "left", "right", "stop"]

N_FOLDS = 5
AUGMENT_PER_SAMPLE = 0  # how many augmented versions per original window

# Augmentation parameters
NOISE_LEVEL = 0.025  # relative to std
SCALE_RANGE = (0.92, 1.08)
STRETCH_RANGE = (0.94, 1.06)  # mild time stretch


# ----------------------------------------

def extract_features(window):
    """Same rich feature set you already have"""
    feats = []
    for col in range(window.shape[1]):
        signal = window[:, col]
        if len(signal) < 3:
            feats.extend([0.0] * 11)
            continue
        feats.extend([
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),
            np.median(signal),
            np.median(np.abs(signal - np.median(signal))),
            np.percentile(signal, 75) - np.percentile(signal, 25),
            np.sqrt(np.mean(signal ** 2)),  # RMS
            skew(signal, nan_policy='omit'),
            kurtosis(signal, nan_policy='omit'),
            np.sum(signal ** 2) / len(signal),  # normalized energy
            np.sum(np.diff(np.sign(signal)) != 0) / len(signal),  # ZCR
        ])
    return np.array(feats)


def augment_window(window):
    """Simple but effective IMU/time-series augmentations"""
    aug = window.copy().astype(float)

    # 1. Random scaling
    scale = np.random.uniform(*SCALE_RANGE)
    aug *= scale

    # 2. Gaussian noise (per channel)
    stds = np.std(aug, axis=0, keepdims=True)
    noise = np.random.normal(0, NOISE_LEVEL * stds, aug.shape)
    aug += noise

    # 3. Mild time stretch (resample)
    stretch = np.random.uniform(*STRETCH_RANGE)
    old_time = np.linspace(0, 1, WINDOW_SIZE)
    new_time = np.linspace(0, 1, int(WINDOW_SIZE * stretch))
    if len(new_time) < 2:
        return window  # rare failure case

    for ch in range(aug.shape[1]):
        interp = interp1d(old_time, aug[:, ch], kind='linear', fill_value="extrapolate")
        aug_stretched = interp(new_time)
        # resize back to WINDOW_SIZE
        if len(aug_stretched) != WINDOW_SIZE:
            interp2 = interp1d(np.linspace(0, 1, len(aug_stretched)), aug_stretched,
                               kind='linear', fill_value="extrapolate")
            aug[:, ch] = interp2(old_time)
        else:
            aug[:, ch] = aug_stretched

    return aug


# ─── Collect data + augment ────────────────────────────────────────
X_raw = []  # original windows
y_raw = []

files = glob.glob("data/*.csv")

for file in files:
    df = pd.read_csv(file)
    df = df[df["label"].isin(GESTURES)]

    for gesture in GESTURES:
        gdf = df[df["label"] == gesture][FEATURES].values

        for i in range(0, len(gdf) - WINDOW_SIZE + 1, STRIDE):
            window = gdf[i:i + WINDOW_SIZE]
            if window.shape[0] != WINDOW_SIZE:
                continue
            X_raw.append(window)
            y_raw.append(gesture)

print(f"Original windows collected: {len(X_raw)}")

# Augment
X_aug = []
y_aug = []

for win, lab in zip(X_raw, y_raw):
    # original
    X_aug.append(extract_features(win))
    y_aug.append(lab)

    # augmented versions
    for _ in range(AUGMENT_PER_SAMPLE):
        aug_win = augment_window(win)
        X_aug.append(extract_features(aug_win))
        y_aug.append(lab)

X = np.array(X_aug)
y = np.array(y_aug)

print(f"Total samples after augmentation: {len(X)}  "
      f"({AUGMENT_PER_SAMPLE + 1} versions per original)")


# ─── Cross-validation + final model ────────────────────────────────
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

accuracies = []
macro_f1s = []
conf_matrices = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # You can also try: n_estimators=200–500, max_depth=12–20, class_weight='balanced'
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='macro')

    accuracies.append(acc)
    macro_f1s.append(f1)

    print(f"\nFold {fold}/{N_FOLDS}  —  Accuracy: {acc:.4f}   Macro-F1: {f1:.4f}")
    # print(classification_report(y_te, y_pred, zero_division=0))   # uncomment if you want per-fold detail

    # Optional: accumulate confusion matrix
    if fold == 1:
        cm_total = confusion_matrix(y_te, y_pred, labels=GESTURES)
    else:
        cm_total += confusion_matrix(y_te, y_pred, labels=GESTURES)

print("\n" + "═" * 60)
print(f"CV Results ({N_FOLDS}-fold Stratified)")
print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Macro F1:  {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
print("═" * 60)

print("\nAverage Confusion Matrix (summed across folds):")
print(cm_total)

# ─── Final model on all data for deployment ────────────────────────
print("\nTraining final model on ALL data...")

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
final_model.fit(X_scaled, y)

# Save
joblib.dump(final_model, "rf_gesture_model.pkl")
joblib.dump(scaler_final, "rf_scaler.pkl")

print("Final model & scaler saved.")
print("Done.")