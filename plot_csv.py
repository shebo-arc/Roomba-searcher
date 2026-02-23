import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/kri19.csv"

df = pd.read_csv(CSV_FILE)
df["label"] = df["label"].astype(str)

SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz"]

gestures = sorted(df["label"].unique())

n = len(gestures)
cols = 2
rows = (n + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

for i, gesture in enumerate(gestures):
    ax = axes[i]
    subset = df[df["label"] == gesture]

    if subset.empty:
        ax.set_title(f"{gesture} (no data)")
        continue

    subset[SIGNALS].reset_index(drop=True).plot(ax=ax, alpha=0.8)

    ax.set_title(gesture)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sensor value")
    ax.legend(loc="upper right", fontsize=8)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
