import serial
import keyboard
import csv
import time

PORT = "COM3"      # change this
BAUD = 115200
OUTPUT_FILE = "data/gesture_data16.csv"

GESTURES = ["idle", "forward", "left", "right", "stop"]
#GESTURES = ["idle", "left"]
gesture_index = 0

recording = False

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Controls:")
print(" r : start/stop recording")
print(" n : next gesture")
print(" q : quit")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    # header
    writer.writerow([
        "time_ms",
        "ax","ay","az",
        "gx","gy","gz",
        "acc_mag","gyro_mag",
        "label"
    ])

    while True:
        if keyboard.is_pressed("r"):
            recording = not recording
            print("Recording:", recording)
            time.sleep(0.4)

        if keyboard.is_pressed("n"):
            gesture_index = (gesture_index + 1) % len(GESTURES)
            print("Gesture:", GESTURES[gesture_index])
            time.sleep(0.4)

        if keyboard.is_pressed("q"):
            print("Quitting...")
            break

        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()

            if not line or line.startswith("time_ms"):
                continue

            if recording:
                row = line.split(",")
                row.append(GESTURES[gesture_index])
                writer.writerow(row)
