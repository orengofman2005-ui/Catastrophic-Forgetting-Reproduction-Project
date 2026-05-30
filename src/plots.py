import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Input files
# =========================================================
DECISION_CSV = "results/decision_random_forest.csv"
METRICS_CSV = "results/metrics.csv"
FIGURES_DIR = "results/figures"


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================================================
# Select an example recording
# =========================================================
def get_example_recording(df):
    first_row = df.iloc[0]
    cup_id = first_row["cup_id"]
    take_id = first_row["take_id"]

    example = df[
        (df["cup_id"] == cup_id) & (df["take_id"] == take_id)
    ].copy().sort_values("start_time")

    return example, cup_id, take_id


# =========================================================
# Plot 1: Ground Truth vs Prediction
# =========================================================
def plot_prediction_vs_ground_truth():
    df = pd.read_csv(DECISION_CSV)
    example, cup_id, take_id = get_example_recording(df)

    plt.figure(figsize=(10, 5))

    plt.plot(example["end_time"], example["fill_percent"], label="Ground Truth")
    plt.plot(
        example["end_time"], example["prediction_monotonic"],
        label="Prediction (smoothed + monotonic)"
    )
    plt.axhline(y=90, linestyle="--", label="Stop threshold = 90%")

    plt.xlabel("Time [sec]")
    plt.ylabel("Fill [%]")
    plt.title(f"Prediction vs Ground Truth | {cup_id}, {take_id}")
    plt.legend()
    plt.grid(True)

    out = os.path.join(FIGURES_DIR, "prediction_vs_ground_truth.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =========================================================
# Plot 2: Smoothing stages comparison
# =========================================================
def plot_smoothing_demo():
    df = pd.read_csv(DECISION_CSV)
    example, cup_id, take_id = get_example_recording(df)

    plt.figure(figsize=(10, 5))

    plt.plot(example["end_time"], example["prediction"], label="Raw prediction", alpha=0.5)
    plt.plot(example["end_time"], example["prediction_smooth"], label="EMA smoothed")
    plt.plot(example["end_time"], example["prediction_monotonic"], label="Monotonic")
    plt.axhline(y=90, linestyle="--", label="Stop threshold = 90%")

    plt.xlabel("Time [sec]")
    plt.ylabel("Predicted fill [%]")
    plt.title(f"Decision Logic Stages | {cup_id}, {take_id}")
    plt.legend()
    plt.grid(True)

    out = os.path.join(FIGURES_DIR, "prediction_smoothing_demo.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =========================================================
# Plot 3: Model comparison by RMSE and R²
# =========================================================
def plot_metrics_comparison():
    metrics = pd.read_csv(METRICS_CSV)
    summary = metrics.groupby("model")[["RMSE", "R2"]].mean().reset_index()
    std_df = metrics.groupby("model")[["RMSE", "R2"]].std().reset_index()

    # --- RMSE (with error bars = std across folds) ---
    plt.figure(figsize=(9, 5))
    bars = plt.bar(summary["model"], summary["RMSE"])
    plt.errorbar(
        summary["model"], summary["RMSE"],
        yerr=std_df["RMSE"], fmt="none", color="black", capsize=4
    )
    plt.xlabel("Model")
    plt.ylabel("Mean RMSE [%]")
    plt.title("Model Comparison — RMSE (lower is better)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "metrics_rmse_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # --- R² (with error bars) ---
    plt.figure(figsize=(9, 5))
    plt.bar(summary["model"], summary["R2"])
    plt.errorbar(
        summary["model"], summary["R2"],
        yerr=std_df["R2"], fmt="none", color="black", capsize=4
    )
    plt.xlabel("Model")
    plt.ylabel("Mean R²")
    plt.title("Model Comparison — R² (higher is better)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "metrics_r2_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =========================================================
# Plot 4: Feature importances
#
# Shows the top_n most important features for a tree-based model.
# Helps explain "what the system learns" and validates that relevant features
# (like MFCC and spectral features) actually contribute to the prediction.
# =========================================================
def plot_feature_importances(model_name="random_forest", top_n=20):
    imp_path = f"results/feature_importances_{model_name}.csv"

    if not os.path.exists(imp_path):
        print(f"Skipping feature importances — file not found: {imp_path}")
        print("Run train.py first.")
        return

    imp_df = pd.read_csv(imp_path).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, f"feature_importances_{model_name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =========================================================
# Plot 5: Per-cup performance
#
# Shows which cups the model is less accurate on —
# allows identifying "hard" cups (different material/size) and understanding why.
# This plot is important for the report: it shows the generalization limits of the model.
# =========================================================
def plot_per_cup_performance():
    metrics = pd.read_csv(METRICS_CSV)

    # Expand the test_cups column (may contain multiple cups separated by comma)
    rows = []
    for _, row in metrics.iterrows():
        for cup in str(row["test_cups"]).split(","):
            cup = cup.strip()
            if cup:
                rows.append({"cup": cup, "model": row["model"], "RMSE": row["RMSE"]})

    cup_df = pd.DataFrame(rows)
    pivot = cup_df.pivot_table(
        index="cup", columns="model", values="RMSE", aggfunc="mean"
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_xlabel("Cup ID")
    ax.set_ylabel("RMSE [%]")
    ax.set_title("Per-Cup RMSE by Model")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "per_cup_rmse.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =========================================================
# Run all plots
# =========================================================
def main():
    ensure_figures_dir()

    plot_prediction_vs_ground_truth()
    plot_smoothing_demo()
    plot_metrics_comparison()
    plot_feature_importances()
    plot_per_cup_performance()

    print()
    print("All figures saved successfully.")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()
