import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS_DIR = "results_fixed"

# A lookup table that decides what colors and labels to use when drawing lines on our graphs.
STYLE_MAP = {
    "Sigmoid_SGD": {"color": "blue", "label": "SGD Sigmoid"},
    "Sigmoid_Dropout": {"color": "green", "label": "Dropout Sigmoid"},
    "ReLU_SGD": {"color": "red", "label": "SGD ReLU"},
    "ReLU_Dropout": {"color": "cyan", "label": "Dropout ReLU"},
    "Maxout_SGD": {"color": "magenta", "label": "SGD Maxout"},
    "Maxout_Dropout": {"color": "gold", "label": "Dropout Maxout"},
    "LWTA_SGD": {"color": "black", "label": "SGD LWTA"},
    "LWTA_Dropout": {"color": "gray", "label": "Dropout LWTA"},
}

# This function finds the "bottom edge" of a group of points on our graph.
# Imagine stretching a rubber band under the dots to see which ones stick out at the bottom.
def get_lower_convex_hull(points: np.ndarray) -> np.ndarray:
    """
    מחשב את הקצה התחתון של הקמור (Lower Convex Hull) במרחב הליניארי.
    זה מאפשר לעקומה לעלות חזרה למעלה בצד ימין, בדיוק כמו במאמר.
    """
    # First, we sort the dots from left to right so we can easily read them in order.
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]

    lower = []
    for p in sorted_points:
        while len(lower) >= 2:
            p1 = lower[-2]
            p2 = lower[-1]
            p3 = p
            
            # חישוב מכפלה וקטורית כדי לוודא שאנחנו יוצרים קמור תחתון (פניות שמאלה בלבד)
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)

    return np.array(lower)


# This function goes through all the results from our tests
# and extracts the most successful points so we can draw them.
def get_frontier_points(trial_summaries: Dict[str, List[dict]]) -> Dict[str, np.ndarray]:
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
            # שימוש בקמור התחתון במקום בפארטו המחמיר
            frontier = get_lower_convex_hull(pts)
            frontier_points[cond_name] = frontier

    return frontier_points

# This function creates a visual graph (a scatter plot with connecting lines).
# It shows us how many mistakes the model made on the old task versus the new task.
def plot_frontier_from_all_trials(
    trial_summaries: Dict[str, List[dict]],
    title: str,
    save_path: str,
    scenario_num: int,
):
    fig, ax = plt.subplots(figsize=(10, 8))

    frontier_points = get_frontier_points(trial_summaries)

    for cond_name, pts in frontier_points.items():
        style = STYLE_MAP.get(cond_name, {"color": "gray", "label": cond_name})

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=style["color"],
            linewidth=1.5,
            marker="o",
            markersize=4,
            alpha=0.9,
            label=style["label"],
        )

    ax.set_xlabel("Old Task Classification Error", fontsize=13)
    ax.set_ylabel("New Task Classification Error", fontsize=13)
    ax.set_title(title, fontsize=18, pad=16)

    # מעבר לסקאלה לוגריתמית כמו במאמר
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=11)
    ax.margins(0.1)

    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=10,
        borderpad=0.8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# This function draws a bar chart (like pillars).
# It shows us how big (how many 'brain cells' or parameters) the best models were.
def plot_winning_model_sizes(
    winning_models: Dict[str, int],
    title: str,
    save_path: str,
):
    names = list(winning_models.keys())
    counts = [winning_models[k] for k in names]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(names, counts, color="teal", alpha=0.8)

    ax.set_ylabel("Total Parameter Count", fontsize=12)
    ax.set_title(title, fontsize=16, pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    scenario_titles = {
        1: "Scenario 1: Reformatting Task (MNIST)",
        2: "Scenario 2: Similar Task (Amazon Reviews)",
        3: "Scenario 3: Dissimilar Task (MNIST vs Amazon Reviews)",
    }

    for i in [1, 2, 3]:
        ckpt_path = os.path.join(RESULTS_DIR, f"scenario_{i}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path} (not found)")
            continue

        data = torch.load(ckpt_path, weights_only=False)

        # קריאה לפונקציה המעודכנת שמציירת את הגרפים הלוגריתמיים
        plot_frontier_from_all_trials(
            data["trial_summaries"],
            scenario_titles[i],
            os.path.join(RESULTS_DIR, f"fig_s{i}_frontier.png"),
            scenario_num=i,
        )

        plot_winning_model_sizes(
            data["winning_models"],
            f"Scenario {i}: Parameter Count of Winning Models",
            os.path.join(RESULTS_DIR, f"fig_s{i}_params.png"),
        )

        print(f"Plotted scenario {i}")