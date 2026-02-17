import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("gesture_data1.csv")

# Drop first column (time_ms)
df = df.iloc[:, 1:]


def smooth(x, window=5):
    return x.rolling(window, center=True).mean()

for col in ["gx", "gy", "gz"]:
    df[col] = smooth(df[col])

# Drop empty rows
df = df.dropna(how="all")

print(df.head())

# Plot everything
df.plot(figsize=(12, 6))
plt.xlabel("Sample")
plt.ylabel("Value")
plt.title("Sensor Data (without time column)")
plt.grid(True)
plt.tight_layout()
plt.show()
