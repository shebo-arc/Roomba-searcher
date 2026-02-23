import numpy as np
import pandas as pd
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from scipy.stats import skew, kurtosis, entropy

# ---------------- CONFIG ----------------
WINDOW_SIZE = 75
STRIDE = 50
FEATURES = ["ay", "az", "gx", "gz"]
GESTURES = ["idle","forward", "left", "right", "stop"]
# ----------------------------------------

def extract_features(window):
    feats = []
    for col in range(window.shape[1]):
        signal = window[:, col]
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
    return feats

X = []
y = []

# Load all CSV files
files = glob.glob("data/*.csv")  # assumes only gesture CSVs in folder

for file in files:
    df = pd.read_csv(file)
    df = df[df["label"].isin(GESTURES)]

    for gesture in GESTURES:
        gdf = df[df["label"] == gesture][FEATURES].values

        for i in range(0, len(gdf) - WINDOW_SIZE, STRIDE):
            window = gdf[i:i + WINDOW_SIZE]
            feats = extract_features(window)
            X.append(feats)
            y.append(gesture)

X = np.array(X)
y = np.array(y)
print("Total samples:", len(X))

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- SVM ----------------
model = SVC(kernel="rbf", C=10, gamma="scale",probability=True)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save model + scaler
joblib.dump(model, "svm_gesture_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")

print("\nModel saved.")