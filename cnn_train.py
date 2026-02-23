import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import time
from scipy import signal

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 75
STRIDE         = 50
FEATURES       = ["ay", "az", "gx", "gz"]
GESTURES       = ["idle", "forward", "left", "right", "stop"]

TEST_SIZE      = 0.2
BATCH_SIZE     = 32
EPOCHS         = 200
PATIENCE       = 12

CONV_FILTERS   = 8
KERNEL_SIZE    = 3
DENSE_1        = 64
DENSE_2        = 32
DROPOUT_RATE   = 0.3

# Butterworth filter parameters
SAMPLING_RATE  = 50.0          # Hz — CHANGE THIS to your actual IMU sampling rate!
CUTOFF_FREQ    = 12.0           # Hz — recommended 10–15 Hz for gestures
# ────────────────────────────────────────────────────────────────

# Design 1st-order Butterworth low-pass filter (done once)
Wn = CUTOFF_FREQ / (SAMPLING_RATE / 2)
b, a = signal.butter(1, Wn, btype='low', analog=False, output='ba')

def apply_filter(signal_array):
    """Apply first-order Butterworth low-pass filter to a 1D signal"""
    filtered, _ = signal.lfilter(b, a, signal_array, zi=signal.lfilter_zi(b, a))
    return filtered

def load_and_prepare_data():
    X_raw = []
    y_raw = []

    files = glob.glob("data/*.csv")

    for file in files:
        df = pd.read_csv(file)
        df = df[df["label"].isin(GESTURES)]

        for gesture in GESTURES:
            gdf = df[df["label"] == gesture][FEATURES].values

            # Apply filter to each channel BEFORE windowing
            filtered_gdf = np.apply_along_axis(apply_filter, axis=0, arr=gdf)

            for i in range(0, len(filtered_gdf) - WINDOW_SIZE + 1, STRIDE):
                window = filtered_gdf[i:i + WINDOW_SIZE]
                if window.shape[0] != WINDOW_SIZE:
                    continue
                X_raw.append(window)
                y_raw.append(gesture)

    print(f"Collected {len(X_raw)} filtered windows")

    X = np.array(X_raw, dtype=np.float32)
    y_str = np.array(y_raw)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    joblib.dump(le, "label_encoder.pkl")

    print(f"Classes: {le.classes_}")
    return X, y, le.classes_

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(CONV_FILTERS, KERNEL_SIZE, padding='same', input_shape=input_shape),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Conv1D(CONV_FILTERS, KERNEL_SIZE, padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Conv1D(CONV_FILTERS, KERNEL_SIZE, padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Dropout(DROPOUT_RATE),

        layers.Flatten(),
        layers.Dense(DENSE_1, activation='relu'),
        layers.Dense(DENSE_2, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# ─── Main ───────────────────────────────────────────────────────────
def main():
    X, y, class_names = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42
    )

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    model = build_cnn_model(input_shape=(WINDOW_SIZE, len(FEATURES)), num_classes=len(class_names))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    print("\nStarting training...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    train_time = time.time() - start_time
    print(f"\nTraining finished in {train_time:.1f} seconds")

    # ─── Evaluation ─────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ─── Save results ───────────────────────────────────────────────
    with open("cnn_training_results.txt", "w", encoding="utf-8") as f:
        f.write("CNN TRAINING & TEST RESULTS (with Butterworth filtering)\n")
        f.write("═══════════════════════════════════════════════════════════\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sampling rate: {SAMPLING_RATE} Hz   Cutoff: {CUTOFF_FREQ} Hz\n")
        f.write(f"Window size: {WINDOW_SIZE}   Stride: {STRIDE}\n")
        f.write(f"Train / Test split: {len(X_train)} / {len(X_test)}\n")
        f.write(f"Training time: {train_time:.1f} seconds\n\n")

        f.write("Test Set Performance:\n")
        f.write(f"Accuracy:     {acc:.4f}\n")
        f.write(f"Macro F1:     {macro_f1:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print("\nResults saved to: cnn_training_results.txt")

    model.save("cnn_gesture_model.h5")
    print("Model saved: cnn_gesture_model.h5")
    print("Done.")

if __name__ == "__main__":
    main()