import os
import numpy as np
import pandas as pd
import librosa

# =========================================================
# General project settings
# =========================================================

RAW_DIR = "data/raw"
OUTPUT_CSV = "data/processed/windows_index.csv"

# Working sample rate.
# Recordings were made at 44.1kHz, but 22.05kHz is sufficient for water sounds
# (most energy is in the 200-5000 Hz range) and reduces computational load.
SAMPLE_RATE = 22050

# Window size: 0.5 seconds.
# Short enough to capture rapid changes in sound,
# long enough to compute stable MFCC and spectrum.
WINDOW_SEC = 0.5

# Hop: 0.25 seconds => 50% overlap.
# Gives smoother predictions over time and increases the number of training samples.
HOP_SEC = 0.25


# =========================================================
# Load audio
# =========================================================
def load_audio(path, sr=SAMPLE_RATE):
    # librosa: loads WAV, converts to mono, resamples to the chosen sr
    y, sr = librosa.load(path, sr=sr, mono=True)

    # Normalize amplitude to [-1, 1] range —
    # important so that recording volume (microphone distance) does not affect the model
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y, sr


# =========================================================
# Create windows for a single file
# =========================================================
def create_windows_for_file(audio_path, cup_id, take_id):
    y, sr = load_audio(audio_path)

    window_size = int(WINDOW_SEC * sr)
    hop_size = int(HOP_SEC * sr)
    duration_sec = len(y) / sr

    rows = []
    window_id = 0

    for start in range(0, len(y) - window_size, hop_size):

        end = start + window_size

        # Window center in time — used to compute the Ground Truth
        center_sample = (start + end) / 2
        center_time = center_sample / sr

        # Ground Truth: assumption of linear fill over time.
        # i.e.: halfway through the recording duration ~ 50% fill.
        # A good engineering assumption when the faucet flow is relatively constant (as required).
        fill_percent = min(100.0, 100.0 * center_time / duration_sec)

        rows.append({
            "audio_path": audio_path,
            "cup_id": cup_id,        # important for GroupKFold
            "take_id": take_id,
            "window_id": window_id,
            "start_sample": start,
            "end_sample": end,
            "start_time": start / sr,
            "end_time": end / sr,
            "fill_percent": fill_percent
        })

        window_id += 1

    return rows


# =========================================================
# Build the full dataset
# =========================================================
def build_dataset():

    all_rows = []
    n_files = 0
    n_errors = 0

    for cup_name in sorted(os.listdir(RAW_DIR)):

        cup_path = os.path.join(RAW_DIR, cup_name)

        if not os.path.isdir(cup_path):
            continue

        cup_id = cup_name

        for filename in sorted(os.listdir(cup_path)):

            if not filename.lower().endswith(".wav"):
                continue

            audio_path = os.path.join(cup_path, filename)
            take_id = os.path.splitext(filename)[0]

            # try/except: a corrupted WAV file will not crash the entire run
            try:
                rows = create_windows_for_file(audio_path, cup_id, take_id)
            except Exception as e:
                print(f"WARNING: skipping {audio_path} — {e}")
                n_errors += 1
                continue

            if len(rows) == 0:
                print(f"WARNING: no windows created for {audio_path} (too short?)")
                continue

            all_rows.extend(rows)
            n_files += 1

            print(f"Processed: {audio_path}  ({len(rows)} windows)")

    df = pd.DataFrame(all_rows)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print("======================================")
    print(f"Saved dataset index to: {OUTPUT_CSV}")
    print(f"Files processed:  {n_files}")
    print(f"Files skipped:    {n_errors}")
    print(f"Total windows:    {len(df)}")
    print(f"Unique cups:      {df['cup_id'].nunique()}")
    print(f"Unique recordings:{df['take_id'].nunique()}")
    print("======================================")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    build_dataset()
