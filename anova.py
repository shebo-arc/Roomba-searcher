import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]
DATA_FOLDER = "data"  # change if needed
MIN_ROWS_PER_GESTURE = 30  # skip very short segments

# ─────────────────────────────────────────────────────────
means_per_gesture = []  # list of mean vectors (one per gesture)
labels = []  # corresponding gesture name

files = glob.glob(f"{DATA_FOLDER}/*.csv")

for filepath in files:
    print(f"\nProcessing: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  → Error reading file: {e}")
        continue

    if not all(col in df.columns for col in FEATURES):
        print(f"  → Missing some sensor columns — skipping")
        continue

    # Optional: keep only rows with known labels
    if 'label' not in df.columns:
        print(f"  → No 'label' column found — cannot segment gestures")
        continue

    df = df[df['label'].notna()]

    # ─── Detect gesture changes ──────────────────────────────
    # Create group id: new group when label changes
    df['gesture_group'] = (df['label'] != df['label'].shift()).cumsum()

    # Group by the detected segments
    for group_id, group_df in df.groupby('gesture_group'):
        gesture_label = group_df['label'].iloc[0]  # all rows in group have same label

        data = group_df[FEATURES].values  # shape: (n_rows_in_this_gesture, 6)

        if len(data) < MIN_ROWS_PER_GESTURE:
            print(f"  → Skipping short segment ({len(data)} rows) - label: {gesture_label}")
            continue

        # Compute mean per channel for this gesture segment
        mean_row = np.nanmean(data, axis=0)  # shape (6,)

        means_per_gesture.append(mean_row)
        labels.append(gesture_label)

        print(f"  → Added gesture: {gesture_label:12}  ({len(data):3} rows)")

# ─── Convert to arrays ───────────────────────────────────────
if len(means_per_gesture) == 0:
    print("\nNo valid gestures found in any file.")
    exit()

X_means = np.array(means_per_gesture)  # shape: (total_gestures, 6)
y_str = np.array(labels)

# Encode string labels → integers
le = LabelEncoder()
y = le.fit_transform(y_str)

print(f"\nTotal gestures collected: {len(y)}")
print(f"Classes found: {list(le.classes_)}")
print(f"Samples per class:\n{pd.Series(y_str).value_counts()}")

# ─── ANOVA per channel ───────────────────────────────────────
print("\n" + "=" * 60)
print("ANOVA F-scores per channel (higher = better class separation)")
print("=" * 60)

f_scores = []
p_values = []

for i, ch in enumerate(FEATURES):
    groups = [X_means[y == lbl, i] for lbl in np.unique(y)]

    # Skip if any class has too few samples
    if any(len(g) < 2 for g in groups):
        print(f"  {ch:>3}:  SKIPPED  (some class has < 2 samples)")
        f_scores.append(np.nan)
        p_values.append(np.nan)
        continue

    f, p = f_oneway(*groups)
    f_scores.append(f)
    p_values.append(p)

    print(f"  {ch:>3}:  F = {f:10.3f}    p = {p:.4e}")

# Ranked
print("\nChannels ranked by ANOVA F-score (descending):")
sorted_idx = np.argsort(f_scores)[::-1]
for idx in sorted_idx:
    if np.isnan(f_scores[idx]):
        continue
    print(f"  {FEATURES[idx]:>3}:  {f_scores[idx]:10.3f}   (p={p_values[idx]:.2e})")

# ─── Mutual Information ──────────────────────────────────────
print("\n" + "=" * 60)
print("Mutual Information per channel")
print("=" * 60)

mi_scores = mutual_info_classif(X_means, y, random_state=42)

for ch, score in zip(FEATURES, mi_scores):
    print(f"  {ch:>3}:  {score:.4f}")

print("\nChannels ranked by MI (descending):")
sorted_mi = np.argsort(mi_scores)[::-1]
for idx in sorted_mi:
    print(f"  {FEATURES[idx]:>3}:  {mi_scores[idx]:.4f}")