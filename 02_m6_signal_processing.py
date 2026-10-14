import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# M6 — STYLUS SIGNAL PROCESSING
# ============================================================

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

INPUT_FILE = "stylus_motion_clean.csv"

OUTPUT_FOLDER = "stylus_m6"

FS = 100.0                 # ESP32 sampling frequency
LOW_CUTOFF = 3.0           # Hz
HIGH_CUTOFF = 12.0         # Hz

WINDOW_SECONDS = 2.0
OVERLAP_SECONDS = 1.0

WINDOW_SIZE = int(WINDOW_SECONDS * FS)
STEP_SIZE = int((WINDOW_SECONDS - OVERLAP_SECONDS) * FS)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

output_dir = Path(OUTPUT_FOLDER)
output_dir.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("       M6 — STYLUS SIGNAL PROCESSING")
print("=" * 65)
print()

print(f"Loading: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print()
    print("ERROR: CSV file not found.")
    print()
    print(f"Put '{INPUT_FILE}' in the same folder as this script.")
    input("Press Enter to exit...")
    raise SystemExit

print("CSV loaded successfully.")
print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Ax",
    "Ay",
    "Az",
    "Gx",
    "Gy",
    "Gz"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("ERROR: Missing columns:")
    print(missing_columns)
    print()
    print("Available columns:")
    print(df.columns.tolist())

    input("Press Enter to exit...")
    raise SystemExit


# ============================================================
# DETERMINE LABEL COLUMN
# ============================================================

if "Test" in df.columns:

    df["Label"] = df["Test"]

elif "Label" in df.columns:

    pass

else:

    print("ERROR: No Test/Label column found.")

    input("Press Enter to exit...")
    raise SystemExit


# ============================================================
# CONVERT SENSOR DATA TO NUMPY
# ============================================================

gyro = df[
    ["Gx", "Gy", "Gz"]
].astype(float).to_numpy()

accel = df[
    ["Ax", "Ay", "Az"]
].astype(float).to_numpy()


# ============================================================
# DESIGN BAND-PASS FILTER
# ============================================================

print("Designing 3–12 Hz Butterworth band-pass filter...")

sos = butter(
    4,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=FS,
    output="sos"
)

print("Filter ready.")
print()


# ============================================================
# APPLY FILTER
# ============================================================

print("Filtering gyroscope data...")

gyro_filtered = np.column_stack([
    sosfiltfilt(
        sos,
        gyro[:, axis]
    )
    for axis in range(3)
])


print("Filtering accelerometer data...")

accel_filtered = np.column_stack([
    sosfiltfilt(
        sos,
        accel[:, axis]
    )
    for axis in range(3)
])

print("Filtering complete.")
print()


# ============================================================
# MAGNITUDE
# ============================================================

gyro_magnitude = np.linalg.norm(
    gyro_filtered,
    axis=1
)

accel_magnitude = np.linalg.norm(
    accel_filtered,
    axis=1
)


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def spectral_features(signal):

    frequencies, power = welch(
        signal,
        fs=FS,
        nperseg=min(
            256,
            len(signal)
        )
    )

    band_mask = (
        (frequencies >= LOW_CUTOFF)
        &
        (frequencies <= HIGH_CUTOFF)
    )

    band_frequencies = frequencies[band_mask]
    band_power = power[band_mask]

    if len(band_power) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    total_band_power = np.trapezoid(
        band_power,
        band_frequencies
    )

    dominant_frequency = (
        band_frequencies[
            np.argmax(band_power)
        ]
    )

    probability = (
        band_power /
        (np.sum(band_power) + 1e-12)
    )

    spectral_entropy = -np.sum(
        probability *
        np.log2(
            probability + 1e-12
        )
    )

    return (
        total_band_power,
        dominant_frequency,
        spectral_entropy
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    gyro_window,
    accel_window
):

    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    gyro_mag = np.linalg.norm(
        gyro_window,
        axis=1
    )

    accel_mag = np.linalg.norm(
        accel_window,
        axis=1
    )


    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    gyro_rms = np.sqrt(
        np.mean(
            gyro_mag ** 2
        )
    )

    accel_rms = np.sqrt(
        np.mean(
            accel_mag ** 2
        )
    )


    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    gyro_std = np.std(
        gyro_mag
    )

    accel_std = np.std(
        accel_mag
    )


    # --------------------------------------------------------
    # Frequency-domain features
    # --------------------------------------------------------

    (
        gyro_band_power,
        gyro_dominant_frequency,
        gyro_entropy
    ) = spectral_features(
        gyro_mag
    )


    (
        accel_band_power,
        accel_dominant_frequency,
        accel_entropy
    ) = spectral_features(
        accel_mag
    )


    # --------------------------------------------------------
    # Jerk
    # --------------------------------------------------------

    dt = 1.0 / FS

    gyro_velocity_change = np.diff(
        gyro_mag
    )

    gyro_jerk = (
        gyro_velocity_change /
        dt
    )

    gyro_jerk_rms = np.sqrt(
        np.mean(
            gyro_jerk ** 2
        )
    )


    # --------------------------------------------------------
    # Zero crossings
    # --------------------------------------------------------

    centered_gyro = (
        gyro_mag -
        np.mean(gyro_mag)
    )

    zero_crossings = np.sum(
        np.diff(
            np.signbit(
                centered_gyro
            )
        ) != 0
    )


    # --------------------------------------------------------
    # Axis energy
    # --------------------------------------------------------

    gx_energy = np.mean(
        gyro_window[:, 0] ** 2
    )

    gy_energy = np.mean(
        gyro_window[:, 1] ** 2
    )

    gz_energy = np.mean(
        gyro_window[:, 2] ** 2
    )


    ax_energy = np.mean(
        accel_window[:, 0] ** 2
    )

    ay_energy = np.mean(
        accel_window[:, 1] ** 2
    )

    az_energy = np.mean(
        accel_window[:, 2] ** 2
    )


    # --------------------------------------------------------
    # Return feature dictionary
    # --------------------------------------------------------

    return {

        "gyro_rms":
            gyro_rms,

        "gyro_std":
            gyro_std,

        "accel_rms":
            accel_rms,

        "accel_std":
            accel_std,

        "gyro_band_power_3_12Hz":
            gyro_band_power,

        "gyro_dominant_freq_Hz":
            gyro_dominant_frequency,

        "gyro_spectral_entropy":
            gyro_entropy,

        "accel_band_power_3_12Hz":
            accel_band_power,

        "accel_dominant_freq_Hz":
            accel_dominant_frequency,

        "accel_spectral_entropy":
            accel_entropy,

        "gyro_jerk_rms":
            gyro_jerk_rms,

        "gyro_zero_crossings":
            zero_crossings,

        "gx_energy":
            gx_energy,

        "gy_energy":
            gy_energy,

        "gz_energy":
            gz_energy,

        "ax_energy":
            ax_energy,

        "ay_energy":
            ay_energy,

        "az_energy":
            az_energy
    }


# ============================================================
# SLIDING WINDOW FEATURE EXTRACTION
# ============================================================

print("=" * 65)
print("Extracting sliding-window features...")
print()

feature_rows = []

total_samples = len(df)

start = 0

while (
    start + WINDOW_SIZE
    <= total_samples
):

    end = (
        start +
        WINDOW_SIZE
    )


    # --------------------------------------------------------
    # Get labels inside window
    # --------------------------------------------------------

    labels = df[
        "Label"
    ].iloc[start:end].unique()


    # Don't create windows crossing
    # different activities

    if len(labels) != 1:

        start += STEP_SIZE
        continue


    label = labels[0]


    # --------------------------------------------------------
    # Sensor windows
    # --------------------------------------------------------

    gyro_window = (
        gyro_filtered[
            start:end
        ]
    )

    accel_window = (
        accel_filtered[
            start:end
        ]
    )


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_features(
        gyro_window,
        accel_window
    )


    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    features["start_time_s"] = (
        start / FS
    )

    features["end_time_s"] = (
        end / FS
    )

    features["Label"] = label


    feature_rows.append(
        features
    )


    start += STEP_SIZE


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    feature_rows
)


if features_df.empty:

    print("ERROR: No feature windows generated.")

    input("Press Enter to exit...")
    raise SystemExit


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

feature_file = (
    output_dir /
    "stylus_m6_features.csv"
)

features_df.to_csv(
    feature_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_columns = [

    "gyro_rms",

    "gyro_std",

    "accel_rms",

    "accel_std",

    "gyro_band_power_3_12Hz",

    "gyro_dominant_freq_Hz",

    "gyro_spectral_entropy",

    "accel_band_power_3_12Hz",

    "accel_dominant_freq_Hz",

    "accel_spectral_entropy",

    "gyro_jerk_rms",

    "gx_energy",

    "gy_energy",

    "gz_energy"
]


summary_df = (
    features_df
    .groupby("Label")[summary_columns]
    .agg(["mean", "std"])
    .round(4)
)


summary_file = (
    output_dir /
    "stylus_m6_summary.csv"
)

summary_df.to_csv(
    summary_file
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 65)
print("M6 COMPLETE")
print("=" * 65)
print()

print(
    f"Total samples: {len(df)}"
)

print(
    f"Sampling rate: {FS} Hz"
)

print(
    f"Filter: {LOW_CUTOFF}–{HIGH_CUTOFF} Hz"
)

print(
    f"Window: {WINDOW_SECONDS} seconds"
)

print(
    f"Overlap: {OVERLAP_SECONDS} seconds"
)

print()

print("Generated windows:")

print(
    features_df[
        "Label"
    ].value_counts()
)

print()

print("Mean feature summary:")

print(summary_df)

print()

print("Files created:")

print(
    feature_file
)

print(
    summary_file
)

print()

# ============================================================
# PLOT 1 — FILTERED GYRO MAGNITUDE
# ============================================================

time_axis = (
    np.arange(
        len(gyro_magnitude)
    )
    / FS
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_axis,
    gyro_magnitude
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Filtered Gyro Magnitude (deg/s)"
)

plt.title(
    "ESP32 Stylus — 3–12 Hz Filtered Gyroscope"
)

plt.tight_layout()


filtered_plot = (
    output_dir /
    "filtered_gyro_3_12Hz.png"
)

plt.savefig(
    filtered_plot,
    dpi=160
)

plt.close()


# ============================================================
# PLOT 2 — GYRO RMS DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for label in [
    "STILL",
    "NORMAL_WRITING",
    "RAPID_MOVEMENT"
]:

    values = features_df.loc[
        features_df["Label"] == label,
        "gyro_rms"
    ]

    plt.hist(
        values,
        bins=15,
        alpha=0.45,
        label=label
    )


plt.xlabel(
    "3–12 Hz Gyro RMS (deg/s)"
)

plt.ylabel(
    "Number of Windows"
)

plt.title(
    "Gyro RMS Distribution"
)

plt.legend()

plt.tight_layout()


distribution_plot = (
    output_dir /
    "gyro_rms_distribution.png"
)

plt.savefig(
    distribution_plot,
    dpi=160
)

plt.close()


# ============================================================
# FINISH
# ============================================================

print(
    f"Saved plot: {filtered_plot}"
)

print(
    f"Saved plot: {distribution_plot}"
)

print()
print("=" * 65)
print("ALL M6 PROCESSING FINISHED")
print("=" * 65)

input(
    "\nPress Enter to exit..."
)