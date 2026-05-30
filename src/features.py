import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

WINDOWS_CSV = "data/processed/windows_index.csv"
OUTPUT_CSV = "data/processed/features.csv"

SAMPLE_RATE = 22050

# Increased from 13 to 20 MFCC coefficients:
# more coefficients = finer description of the spectral shape,
# which allows detecting subtler changes in the water sound.
N_MFCC = 20


# =========================================================
# Band-Pass Filter
# Keeps the frequency range 200-5000 Hz.
# Water produces main sounds in this range;
# below 200 Hz — usually faucet/motor noise;
# above 5000 Hz — usually white noise and background.
# =========================================================
def bandpass_filter(y, sr, low=200, high=5000, order=4):
    nyquist = sr / 2
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype="band")
    y_filtered = filtfilt(b, a, y)
    return y_filtered


# =========================================================
# 1/f slope calculation
# Checks how energy decreases with frequency on a log-log scale.
# As the cup fills, the resonance frequency rises — the slope changes.
# =========================================================
def spectral_slope_1overf(y, sr):
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)

    mask = freqs > 0
    freqs = freqs[mask]
    spectrum = spectrum[mask]

    log_f = np.log(freqs)
    log_p = np.log(spectrum + 1e-12)

    slope, _ = np.polyfit(log_f, log_p, 1)
    return slope


# =========================================================
# Spectral Entropy
#
# Measures how "spread" vs "focused" the spectrum is:
# high entropy = noise (energy spread across many frequencies)
# low entropy  = pure tone (energy focused on specific frequencies)
#
# As the cup fills, the sound shifts from "falling noise" to "resonance tone" —
# entropy is expected to decrease toward the end of filling.
# =========================================================
def spectral_entropy(y, sr):
    spectrum = np.abs(np.fft.rfft(y)) ** 2 + 1e-12
    prob = spectrum / spectrum.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    return float(entropy)


# =========================================================
# Sub-band Energy Ratios
#
# Splits the spectrum into 4 bands and computes
# what percentage of total energy is in each band.
#
# As the cup fills, the resonance frequency of the liquid rises —
# energy shifts from lower to higher bands.
# This is one of the most sensitive indicators of volume change.
# =========================================================
def subbands_energy_ratio(y, sr):
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)
    spectrum = np.abs(np.fft.rfft(y)) ** 2

    total = spectrum.sum() + 1e-12

    # The four bands are chosen to cover the acoustic range
    # relevant to water: cup resonance, falling wave, and harmonics
    bands = [
        (200, 500),    # basic wave of water falling
        (500, 1500),   # cup resonance (main range)
        (1500, 3000),  # first harmonics
        (3000, 5000),  # high harmonics
    ]

    result = {}
    for i, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        result[f"subband_energy_ratio_{i + 1}"] = float(spectrum[mask].sum() / total)

    return result


# =========================================================
# Feature extraction from a single audio window
# =========================================================
def extract_features_from_window(y_window, sr):
    features = {}

    # ---------- Energy features ----------
    features["rms"] = float(np.mean(librosa.feature.rms(y=y_window)))

    # ---------- Zero Crossing Rate ----------
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y_window)))

    # ---------- Basic spectral features ----------
    features["spectral_centroid"] = float(np.mean(
        librosa.feature.spectral_centroid(y=y_window, sr=sr)
    ))

    features["spectral_bandwidth"] = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y_window, sr=sr)
    ))

    features["spectral_rolloff"] = float(np.mean(
        librosa.feature.spectral_rolloff(y=y_window, sr=sr)
    ))

    features["spectral_flatness"] = float(np.mean(
        librosa.feature.spectral_flatness(y=y_window)
    ))

    # ---------- 1/f slope ----------
    features["slope_1overf"] = spectral_slope_1overf(y_window, sr)

    # ---------- Spectral entropy ----------
    features["spectral_entropy"] = spectral_entropy(y_window, sr)

    # ---------- Sub-band energy ratios ----------
    subbands = subbands_energy_ratio(y_window, sr)
    features.update(subbands)

    # ---------- Spectral Contrast ----------
    # Measures the difference between peaks and valleys in each band — 7 values (6 bands + overall).
    # Useful for distinguishing between different types of water sounds (falling, resonance, bubbles).
    contrast = librosa.feature.spectral_contrast(y=y_window, sr=sr, n_bands=6)
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_{i + 1}"] = float(np.mean(contrast[i]))

    # ---------- MFCC + Delta + Delta-Delta ----------
    #
    # MFCC:         represents the spectral shape at a given time point
    # Delta:        rate of MFCC change — "speed" of spectral change
    # Delta-Delta:  acceleration of change — "acceleration" of spectral change
    #
    # Delta and Delta-Delta are critical here:
    # cup filling is a dynamic process — the spectrum changes continuously.
    # Representing only the current shape (MFCC) misses the
    # rate and character of change (delta), which is the real signal.
    mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}"] = float(np.mean(mfcc[i]))
        features[f"mfcc_delta_{i + 1}"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta2_{i + 1}"] = float(np.mean(mfcc_delta2[i]))

    return features


# =========================================================
# Full audio loading
# =========================================================
def load_audio(path, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y, sr


# =========================================================
# Build features.csv
# =========================================================
def build_features():
    windows_df = pd.read_csv(WINDOWS_CSV)
    total = len(windows_df)

    all_rows = []
    current_audio_path = None
    current_audio = None
    current_sr = None

    for idx, row in windows_df.iterrows():

        audio_path = row["audio_path"]

        # Reload only when switching to a new audio file — saves runtime
        if audio_path != current_audio_path:
            current_audio_path = audio_path
            current_audio, current_sr = load_audio(audio_path)
            current_audio = bandpass_filter(current_audio, current_sr)
            print(f"Loaded and filtered: {audio_path}")

        start = int(row["start_sample"])
        end = int(row["end_sample"])
        y_window = current_audio[start:end]

        if len(y_window) == 0:
            continue

        feats = extract_features_from_window(y_window, current_sr)

        feats["audio_path"] = row["audio_path"]
        feats["cup_id"] = row["cup_id"]
        feats["take_id"] = row["take_id"]
        feats["window_id"] = row["window_id"]
        feats["start_time"] = row["start_time"]
        feats["end_time"] = row["end_time"]
        feats["fill_percent"] = row["fill_percent"]

        all_rows.append(feats)

        # Print progress every 100 windows
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{total} windows")

    features_df = pd.DataFrame(all_rows)
    features_df.to_csv(OUTPUT_CSV, index=False)

    n_feature_cols = len(features_df.columns) - 7  # excluding 7 metadata columns

    print("======================================")
    print(f"Saved features to: {OUTPUT_CSV}")
    print(f"Total feature rows: {len(features_df)}")
    print(f"Feature count: {n_feature_cols}")
    print("======================================")


if __name__ == "__main__":
    build_features()
