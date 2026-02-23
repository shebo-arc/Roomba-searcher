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
from scipy.interpolate import interp1d

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

# Augmentation
AUGMENT_PER_SAMPLE = 1
NOISE_LEVEL = 0.025
SCALE_RANGE = (0.92, 1.08)
STRETCH_RANGE = (0.94, 1.06)

# Butterworth filter
SAMPLING_RATE  = 50.0
CUTOFF_FREQ    = 12.0
# ────────────────────────────────────────────────────────────────

# Design filter once
Wn = CUTOFF_FREQ / (SAMPLING_RATE / 2)
b, a = signal.butter(1, Wn, btype='low', analog=False, output='ba')


def apply_filter(signal_array):
    filtered, _ = signal.lfilter(b, a, signal_array, zi=signal.lfilter_zi(b, a))
    return filtered


def augment_window(window):
    aug = window.copy().astype(float)

    # Random scaling
    scale = np.random.uniform(*SCALE_RANGE)
    aug *= scale

    # Noise injection
    stds = np.std(aug, axis=0, keepdims=True)
    noise = np.random.normal(0, NOISE_LEVEL * stds, aug.shape)
    aug += noise

    # Time stretching
    stretch = np.random.uniform(*STRETCH_RANGE)

    old_time = np.linspace(0, 1, WINDOW_SIZE)
    new_time = np.linspace(0, 1, int(WINDOW_SIZE * stretch))

    if len(new_time) < 2:
        return window

    for ch in range(aug.shape[1]):
        interp = interp1d(old_time, aug[:, ch], kind='linear', fill_value="extrapolate")
        aug_stretched = interp(new_time)

        if len(aug_stretched) != WINDOW_SIZE:
            interp2 = interp1d(
                np.linspace(0, 1, len(aug_stretched)),
                aug_stretched,
                kind='linear',
                fill_value="extrapolate"
            )
            aug[:, ch] = interp2(old_time)
        else:
            aug[:, ch] = aug_stretched

    return aug


def load_and_prepare_data():
    X_raw = []
    y_raw = []

    files = glob.glob("data/*.csv")

    for file in files:
        df = pd.read_csv(file)
        df = df[df["label"].isin(GESTURES)]

        for gesture in GESTURES:
            gdf = df[df["label"] == gesture][FEATURES].values

            # Apply filter before windowing
            filtered_gdf = np.apply_along_axis(apply_filter, axis=0, arr=gdf)

            for i in range(0, len(filtered_gdf) - WINDOW_SIZE + 1, STRIDE):
                window = filtered_gdf[i:i + WINDOW_SIZE]

                if window.shape[0] != WINDOW_SIZE:
                    continue

                # Original
                X_raw.append(window)
                y_raw.append(gesture)

                # Augmented
                for _ in range(AUGMENT_PER_SAMPLE):
                    aug_window = augment_window(window)
                    X_raw.append(aug_window)
                    y_raw.append(gesture)

    print(f"Total windows (after augmentation): {len(X_raw)}")

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


def main():
    X, y, class_names = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42
    )

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    model = build_cnn_model(
        input_shape=(WINDOW_SIZE, len(FEATURES)),
        num_classes=len(class_names)
    )

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

    # Evaluation
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    with open("cnn_training_results.txt", "w", encoding="utf-8") as f:
        f.write("CNN TRAINING & TEST RESULTS (with Augmentation + Filtering)\n")
        f.write("═══════════════════════════════════════════════════════════\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Train/Test: {len(X_train)} / {len(X_test)}\n")
        f.write(f"Training time: {train_time:.1f} sec\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    model.save("cnn_gesture_model.h5")

    print("\nModel saved: cnn_gesture_model.h5")
    print("Results saved: cnn_training_results.txt")
    print("Done.")


if __name__ == "__main__":
    main()