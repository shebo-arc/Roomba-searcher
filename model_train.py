import numpy as np
import pandas as pd
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------- CONFIG ----------------
WINDOW_SIZE = 32
STRIDE = 16
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]
GESTURES = ["idle","forward", "left", "right", "stop"]  # exclude idle
# ----------------------------------------

def extract_features(window):
    feats = []
    for col in range(window.shape[1]):
        signal = window[:, col]
        feats.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal)
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
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save model + scaler
joblib.dump(model, "svm_gesture_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved.")
