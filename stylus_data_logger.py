import serial
import csv
import time

# ============================================================
# SETTINGS
# ============================================================

PORT = "COM11"
BAUD_RATE = 115200

STILL_TIME = 20
WRITING_TIME = 30
RAPID_TIME = 20

OUTPUT_FILE = "stylus_motion_clean.csv"


# ============================================================
# CONNECT TO ESP32
# ============================================================

print("=" * 60)
print("       STYLUS SENSOR - CLEAN DATA LOGGER")
print("=" * 60)
print()

print(f"Connecting to {PORT}...")

try:
    ser = serial.Serial(
        PORT,
        BAUD_RATE,
        timeout=1
    )
except Exception as e:
    print("ERROR: Could not open serial port.")
    print(e)
    input("Press Enter to exit...")
    raise SystemExit

print("Connected.")
print()
print("ESP32 is resetting and calibrating...")
print("Wait until calibration is complete.")
print()


# ============================================================
# WAIT FOR ESP32 TO START PRODUCING SENSOR DATA
# ============================================================

sensor_data_found = False

while not sensor_data_found:

    line = ser.readline().decode(
        "utf-8",
        errors="ignore"
    ).strip()

    if not line:
        continue

    values = line.split(",")

    # Sensor rows have exactly 9 values
    if len(values) == 9:

        try:
            [float(x) for x in values]

            sensor_data_found = True

        except ValueError:
            pass


print("Sensor stream detected.")
print()


# ============================================================
# IMPORTANT:
# THROW AWAY ANY OLD DATA
# ============================================================

ser.reset_input_buffer()

print("Serial buffer cleared.")
print()


# ============================================================
# CREATE CSV
# ============================================================

csv_file = open(
    OUTPUT_FILE,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Time_ms",
    "Ax",
    "Ay",
    "Az",
    "Gx",
    "Gy",
    "Gz",
    "AccelMag",
    "GyroMag",
    "Test"
])


# ============================================================
# COLLECTION FUNCTION
# ============================================================

def collect_data(duration, test_name):

    print()
    print("=" * 60)
    print(f"           {test_name}")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # CRITICAL:
    # Remove everything currently waiting in serial buffer
    # --------------------------------------------------------

    ser.reset_input_buffer()

    print(f"Recording for {duration} seconds...")
    print()

    start_time = time.perf_counter()

    sample_count = 0

    while True:

        elapsed = time.perf_counter() - start_time

        if elapsed >= duration:
            break

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        values = line.split(",")

        if len(values) != 9:
            continue

        try:

            numbers = [
                float(value)
                for value in values
            ]

        except ValueError:
            continue

        # ----------------------------------------------------
        # SAVE DATA
        # ----------------------------------------------------

        writer.writerow([
            numbers[0],
            numbers[1],
            numbers[2],
            numbers[3],
            numbers[4],
            numbers[5],
            numbers[6],
            numbers[7],
            numbers[8],
            test_name
        ])

        sample_count += 1

        remaining = max(
            0,
            duration - elapsed
        )

        print(
            f"\rRemaining: {remaining:5.1f}s"
            f" | Samples: {sample_count}",
            end="",
            flush=True
        )

    print()
    print()
    print(f"Finished {test_name}.")
    print(f"Samples collected: {sample_count}")

    # Flush immediately
    csv_file.flush()


# ============================================================
# TEST 1 — STILL
# ============================================================

print()
print("TEST 1 — STILL")
print()
print("Place the MPU6050 completely still on your desk.")
print("Do not touch it during recording.")
print()

input("Press ENTER to start the 20-second recording...")

collect_data(
    STILL_TIME,
    "STILL"
)


# ============================================================
# GAP BETWEEN TESTS
# ============================================================

print()
print("Test 1 complete.")
print("Prepare for normal writing.")
time.sleep(2)


# ============================================================
# TEST 2 — NORMAL WRITING
# ============================================================

print()
print("TEST 2 — NORMAL WRITING")
print()
print("Hold the sensor like a stylus.")
print("Write normally for 30 seconds.")
print("Do not deliberately shake it.")
print()

input("Press ENTER to start the 30-second recording...")

collect_data(
    WRITING_TIME,
    "NORMAL_WRITING"
)


# ============================================================
# GAP BETWEEN TESTS
# ============================================================

print()
print("Test 2 complete.")
print("Prepare for rapid controlled movement.")
time.sleep(2)


# ============================================================
# TEST 3 — RAPID CONTROLLED MOVEMENT
# ============================================================

print()
print("TEST 3 — RAPID CONTROLLED MOVEMENT")
print()
print("Hold the sensor like a stylus.")
print()
print("Make SMALL, rapid back-and-forth movements.")
print("Do NOT violently shake it.")
print()

input("Press ENTER to start the 20-second recording...")

collect_data(
    RAPID_TIME,
    "RAPID_MOVEMENT"
)


# ============================================================
# FINISH
# ============================================================

csv_file.close()
ser.close()

print()
print()
print("=" * 60)
print("           DATA COLLECTION COMPLETE")
print("=" * 60)
print()
print(f"Saved file:")
print(f"    {OUTPUT_FILE}")
print()
print("The file is ready for signal analysis.")
print()

input("Press Enter to exit...")