import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS_DIR = "results_repro"

# Color and label style for each condition
STYLE_MAP = {
    "Sigmoid_SGD":     {"color": "blue",    "label": "SGD Sigmoid"},
    "Sigmoid_Dropout": {"color": "green",   "label": "Dropout Sigmoid"},
    "ReLU_SGD":        {"color": "red",     "label": "SGD ReLU"},
    "ReLU_Dropout":    {"color": "cyan",    "label": "Dropout ReLU"},
    "Maxout_SGD":      {"color": "magenta", "label": "SGD Maxout"},
    "Maxout_Dropout":  {"color": "gold",    "label": "Dropout Maxout"},
    "LWTA_SGD":        {"color": "black",   "label": "SGD LWTA"},
    "LWTA_Dropout":    {"color": "gray",    "label": "Dropout LWTA"},
}


def get_lower_convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Compute the lower convex hull of a point set in linear space.
    This allows the curve to rise back up on the right side, matching the paper.
    """
    # Sort points left to right by x coordinate
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]

    lower = []
    for p in sorted_points:
        while len(lower) >= 2:
            p1 = lower[-2]
            p2 = lower[-1]
            p3 = p
            # Cross product to ensure we keep only left-turning points (lower convex hull)
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)

    return np.array(lower)


def get_frontier_points(trial_summaries: Dict[str, List[dict]]) -> Dict[str, np.ndarray]:
    """Extract the lower convex hull frontier from all trial trajectories per condition."""
    frontier_points = {}

    for cond_name, trials in trial_summaries.items():
        if not trials:
            continue

        all_pts = []
        for trial in trials:
            all_pts.extend(trial["points"])

        pts = np.asarray(all_pts, dtype=float)
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 0] > 0) & (pts[:, 1] > 0)
        pts = pts[valid]

        if len(pts) > 0:
            frontier = get_lower_convex_hull(pts)
            frontier_points[cond_name] = frontier

    return frontier_points


# Axis limits per scenario to prevent overcrowding (especially Scenario 2 / Amazon)
AXIS_LIMITS = {
    1: dict(xlim=None, ylim=None),
    3: dict(xlim=(0.08, 0.7), ylim=(0.1, 0.6)),
    5: dict(xlim=None, ylim=None),
}


def plot_frontier_from_all_trials(
    trial_summaries: Dict[str, List[dict]],
    title: str,
    save_path: str,
    scenario_num: int,
):
    """Plot the Possibilities Frontier for all 8 conditions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    frontier_points = get_frontier_points(trial_summaries)

    for cond_name, pts in frontier_points.items():
        style = STYLE_MAP.get(cond_name, {"color": "gray", "label": cond_name})
        ax.plot(
            pts[:, 0], pts[:, 1],
            color=style["color"], linewidth=1.5,
            marker="o", markersize=4, alpha=0.9,
            label=style["label"],
        )

    ax.set_xlabel("Old Task Classification Error", fontsize=13)
    ax.set_ylabel("New Task Classification Error", fontsize=13)
    ax.set_title(title, fontsize=18, pad=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="both", labelsize=11)

    limits = AXIS_LIMITS.get(scenario_num, dict(xlim=None, ylim=None))
    if limits["xlim"]:
        ax.set_xlim(limits["xlim"])
        ax.set_ylim(limits["ylim"])
    else:
        ax.margins(0.1)

    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left",
              frameon=True, fontsize=10, borderpad=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_winning_model_sizes(
    winning_models: Dict[str, int],
    title: str,
    save_path: str,
):
    """
    Grouped bar chart of winning model parameter counts — SGD vs Dropout per activation.
    Matches the style of Figures 2, 4, 6 in Goodfellow et al. (2015):
    4 activation groups on X-axis, two bars each (blue=SGD, red=Dropout).
    """
    ACTIVATIONS = ["Sigmoid", "ReLU", "LWTA", "Maxout"]
    sgd_counts     = [winning_models.get(f"{act}_SGD",     0) for act in ACTIVATIONS]
    dropout_counts = [winning_models.get(f"{act}_Dropout", 0) for act in ACTIVATIONS]

    x      = np.arange(len(ACTIVATIONS))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_sgd = ax.bar(x - width / 2, sgd_counts,     width, label="SGD",     color="#3a6dc2", alpha=0.9)
    bars_do  = ax.bar(x + width / 2, dropout_counts, width, label="Dropout", color="#d94f3d", alpha=0.9)

    ax.set_ylabel("Model size (# parameters)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIVATIONS, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"{int(val):,}")
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    scenario_titles = {
        1: "Scenario 1: Reformatting Task (MNIST)",
        3: "Scenario 2: Similar Task (Amazon Reviews)",
        5: "Scenario 3: Dissimilar Task (MNIST vs Amazon Reviews)",
    }

    for i in [1, 3, 5]:
        ckpt_path = os.path.join(RESULTS_DIR, f"scenario_{i}_repro.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path} (not found)")
            continue

        data = torch.load(ckpt_path, weights_only=False)

        # Frontier figure
        plot_frontier_from_all_trials(
            data["trial_summaries"],
            scenario_titles[i],
            os.path.join(RESULTS_DIR, f"fig_s{i}_frontier.png"),
            scenario_num=i,
        )

        # Model sizes figure
        label_num = {1: 1, 3: 2, 5: 3}[i]
        plot_winning_model_sizes(
            data["winning_models"],
            f"Scenario {label_num}: Parameter Count of Winning Models",
            os.path.join(RESULTS_DIR, f"fig_s{i}_params.png"),
        )

        print(f"Plotted scenario {i}")
