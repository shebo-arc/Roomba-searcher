import pandas as pd
import glob
from collections import defaultdict

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_FOLDER = "data"  # ← change if your folder is named differently
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]  # just for validation

# ─────────────────────────────────────────────────────────
lengths_by_label = defaultdict(list)  # key: label → value: list of int lengths

files = glob.glob(f"{DATA_FOLDER}/*.csv")

for filepath in sorted(files):  # sorted → nicer output order
    print(f"\n→ Processing: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"   Error reading file: {e}")
        continue

    if 'label' not in df.columns:
        print("   → No 'label' column found — skipping file")
        continue

    # Optional: drop rows without label
    df = df[df['label'].notna()].copy()

    if df.empty:
        print("   → File is empty after filtering — skipping")
        continue

    # Create group id: new group every time label changes
    df['group'] = (df['label'] != df['label'].shift()).cumsum()

    # Group and get length + label for each segment
    for group_id, group_df in df.groupby('group'):
        label = group_df['label'].iloc[0]
        length = len(group_df)

        # Optional quick sanity check
        if set(group_df['label'].unique()) != {label}:
            print(f"   Warning: inconsistent label in group {group_id}")
            continue

        lengths_by_label[label].append(length)

        print(f"   Gesture '{label}' → {length} rows")

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GESTURE LENGTH STATISTICS (rows per instance)")
print("=" * 70)

all_labels = sorted(lengths_by_label.keys())

for label in all_labels:
    lengths = lengths_by_label[label]
    if not lengths:
        continue

    count = len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = sum(lengths) / count
    total_rows = sum(lengths)

    print(f"\n{label:12}  ({count} instances)")
    print(f"  lengths: {lengths}")
    print(f"  min    = {min_len:4d} rows")
    print(f"  max    = {max_len:4d} rows")
    print(f"  mean   = {mean_len:5.1f} rows")
    print(f"  total  = {total_rows:5d} rows")

print("\nDone.")