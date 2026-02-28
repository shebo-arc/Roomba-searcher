import numpy as np
import pandas as pd
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
import time

# ---------------- CONFIG ----------------
WINDOW_SIZE = 75
STRIDE = 50
FEATURES = ["ay", "az", "gx", "gz"]
GESTURES = ["idle", "forward", "left", "right", "stop"]

TEST_SIZE = 0.2
AUGMENT_PER_SAMPLE = 1

NOISE_LEVEL = 0.025
SCALE_RANGE = (0.92, 1.08)
STRETCH_RANGE = (0.94, 1.06)
# ----------------------------------------

def extract_features(window):
    feats = []
    for col in range(window.shape[1]):
        signal = window[:, col]
        if len(signal) < 3:
            feats.extend([0.0] * 7)
            continue
        feats.extend([
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),
            np.sqrt(np.mean(signal ** 2)),
            skew(signal, nan_policy='omit'),
            np.sum(signal ** 2) / len(signal),
            np.sum(np.diff(np.sign(signal)) != 0) / len(signal),
        ])
    return np.array(feats)

def augment_window(window):
    aug = window.copy().astype(float)
    scale = np.random.uniform(*SCALE_RANGE)
    aug *= scale
    stds = np.std(aug, axis=0, keepdims=True)
    noise = np.random.normal(0, NOISE_LEVEL * stds, aug.shape)
    aug += noise
    stretch = np.random.uniform(*STRETCH_RANGE)
    old_time = np.linspace(0, 1, WINDOW_SIZE)
    new_time = np.linspace(0, 1, int(WINDOW_SIZE * stretch))
    if len(new_time) < 2:
        return window
    for ch in range(aug.shape[1]):
        interp = interp1d(old_time, aug[:, ch], kind='linear', fill_value="extrapolate")
        aug_stretched = interp(new_time)
        if len(aug_stretched) != WINDOW_SIZE:
            interp2 = interp1d(np.linspace(0, 1, len(aug_stretched)), aug_stretched,
                               kind='linear', fill_value="extrapolate")
            aug[:, ch] = interp2(old_time)
        else:
            aug[:, ch] = aug_stretched
    return aug

# ─── Collect data + augment ────────────────────────────────────────
X_raw = []
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

X_aug = []
y_aug = []

for win, lab in zip(X_raw, y_raw):
    X_aug.append(extract_features(win))
    y_aug.append(lab)
    for _ in range(AUGMENT_PER_SAMPLE):
        aug_win = augment_window(win)
        X_aug.append(extract_features(aug_win))
        y_aug.append(lab)

X = np.array(X_aug)
y = np.array(y_aug)

print(f"Total samples after augmentation: {len(X)}  ({AUGMENT_PER_SAMPLE + 1} versions per original)")

# ─── Train-test split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# ─── Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ─── Train SVM ──────────────────────────────────────────────────────
print("\nTraining SVM model...")
model = SVC(
    kernel="rbf",
    C=10,
    gamma=0.1,
    probability=True,
    random_state=42,
    class_weight='balanced'
)

start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"Training completed in {train_time:.2f} seconds")

# ─── Evaluation on test set ─────────────────────────────────────────
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

report = classification_report(y_test, y_pred, digits=4, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=GESTURES)

# ─── Save results to file ───────────────────────────────────────────
with open("svm_retrain_results.txt", "w", encoding="utf-8") as f:
    f.write("SVM TRAINING & TEST RESULTS\n")
    f.write("═══════════════════════════\n\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Window size: {WINDOW_SIZE} | Stride: {STRIDE}\n")
    f.write(f"Augmentation: {AUGMENT_PER_SAMPLE} per sample\n")
    f.write(f"Total samples: {len(X)}\n")
    f.write(f"Train / Test split: {len(X_train)} / {len(X_test)}\n\n")

    f.write(f"Training time: {train_time:.2f} seconds\n\n")

    f.write("Test Set Performance:\n")
    f.write("────────────────────\n")
    f.write(f"Accuracy:     {acc:.4f}\n")
    f.write(f"Macro F1:     {macro_f1:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

print("\nResults saved to: svm_retrain_results.txt")

# ─── Save final model ───────────────────────────────────────────────
joblib.dump(model, "svm_retrain_model.pkl")
joblib.dump(scaler, "svm_retrain_scaler.pkl")

print("\nFinal model & scaler saved.")
print("Done.")