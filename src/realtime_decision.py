import os
import numpy as np
import pandas as pd

# =========================================================
# Model predictions file
# =========================================================
PREDICTIONS_CSV = "results/predictions_random_forest.csv"

# Output file after smoothing + monotonicity + stop decision
OUTPUT_CSV = "results/decision_random_forest.csv"

# Fill percentage at which we want to stop the faucet
TARGET_STOP_PERCENT = 90

# EMA (Exponential Moving Average) coefficient:
# 0.3 = 30% current value + 70% history.
# EMA responds faster to real changes compared to a simple moving average,
# while maintaining enough smoothing against noise spikes.
EMA_ALPHA = 0.3

# How many consecutive windows above 90% are required before stopping —
# prevents early stopping due to a one-time noise spike
CONSECUTIVE_WINDOWS = 3


# =========================================================
# Exponential Moving Average
#
# Improvement over simple moving average:
# newer windows receive higher weight than older ones.
# Allows detecting an upward trend faster near the end of filling.
#
# Formula: EMA[t] = alpha * x[t] + (1 - alpha) * EMA[t-1]
# =========================================================
def exponential_moving_average(values, alpha=EMA_ALPHA):
    result = np.zeros_like(values, dtype=float)
    result[0] = float(values[0])
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


# =========================================================
# Enforce monotonicity
#
# Physically, while filling a cup, the fill percentage should not decrease.
# np.maximum.accumulate returns at each point the maximum seen so far.
# =========================================================
def enforce_monotonicity(values):
    return np.maximum.accumulate(values)


# =========================================================
# Find stop time
#
# We do not stop on a single window above the threshold — it may be momentary noise.
# We require CONSECUTIVE_WINDOWS consecutive windows above TARGET_STOP_PERCENT.
# =========================================================
def find_stop_time(df):
    above_threshold = df["prediction_monotonic"] >= TARGET_STOP_PERCENT
    count = 0

    for i, is_above in enumerate(above_threshold):

        if is_above:
            count += 1
        else:
            count = 0

        if count >= CONSECUTIVE_WINDOWS:
            stop_time = df.iloc[i]["end_time"]
            true_fill = df.iloc[i]["fill_percent"]
            return stop_time, true_fill

    return None, None


# =========================================================
# Apply decision logic to all recordings
# =========================================================
def apply_decision_logic():

    if not os.path.exists(PREDICTIONS_CSV):
        print(f"ERROR: predictions file not found:\n  {PREDICTIONS_CSV}")
        print("Run train.py first.")
        return

    df = pd.read_csv(PREDICTIONS_CSV)

    all_results = []
    stop_events = []

    # Process each recording separately —
    # do not smooth across different recordings (that would be incorrect data mixing)
    grouped = df.groupby(["cup_id", "take_id"])

    for (cup_id, take_id), group in grouped:

        group = group.sort_values("start_time").copy()
        raw_pred = group["prediction"].values

        # Step 1: EMA — smoothing with higher weight on recent predictions
        group["prediction_smooth"] = exponential_moving_average(raw_pred)

        # Step 2: enforce monotonicity — physically, fill level does not decrease
        group["prediction_monotonic"] = enforce_monotonicity(
            group["prediction_smooth"].values
        )

        # Step 3: find the stop point
        stop_time, true_fill_at_stop = find_stop_time(group)

        group["stop_triggered"] = False
        if stop_time is not None:
            group.loc[group["end_time"] >= stop_time, "stop_triggered"] = True

        all_results.append(group)
        stop_events.append({
            "cup_id": cup_id,
            "take_id": take_id,
            "stop_time_sec": stop_time,
            "true_fill_at_stop": true_fill_at_stop,
            "target_stop_percent": TARGET_STOP_PERCENT
        })

    result_df = pd.concat(all_results, ignore_index=True)
    stop_df = pd.DataFrame(stop_events)

    os.makedirs("results", exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    stop_df.to_csv("results/stop_events_random_forest.csv", index=False)

    print("Saved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  results/stop_events_random_forest.csv")

    print()
    print("Stop events:")
    print(stop_df.to_string(index=False))

    # =========================================================
    # Summary statistics on stop events
    #
    # Shows how many recordings reached the stop threshold, and what
    # the actual fill percentage was at the moment the system would have stopped.
    # Small error = system stops close to 90% => good performance.
    # =========================================================
    detected = stop_df.dropna(subset=["true_fill_at_stop"])
    n_detected = len(detected)
    n_total = len(stop_df)

    print()
    print("======================================")
    print(f"Stop events detected: {n_detected}/{n_total}")

    if n_detected > 0:
        mean_fill = detected["true_fill_at_stop"].mean()
        std_fill = detected["true_fill_at_stop"].std()
        mean_time = detected["stop_time_sec"].mean()
        error = (detected["true_fill_at_stop"] - TARGET_STOP_PERCENT).abs()
        mae_stop = error.mean()

        print(f"Mean fill at stop:    {mean_fill:.1f}% ± {std_fill:.1f}%")
        print(f"Mean stop time:       {mean_time:.2f} sec")
        print(f"Mean |error| from {TARGET_STOP_PERCENT}%: {mae_stop:.1f}%")

    print("======================================")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    apply_decision_logic()
