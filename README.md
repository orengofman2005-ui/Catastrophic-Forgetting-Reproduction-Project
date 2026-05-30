# Catastrophic Forgetting — PyTorch Reproduction

PyTorch reproduction of:
> **"An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"**
> Goodfellow, Mirza, Xiao, Courville, Bengio (arXiv:1312.6211, 2015)

---

## Quick links

| | |
|---|---|
| [תובנות ומסקנות](takeaways.md) | ניתוח אישי של התוצאות, מה הפתיע, מה למדתי |
| [מתודולוגיה](docs/methodology.md) | מבנה הניסוי, hyperparameter search, סטיות מהמאמר |
| [תוצאות מפורטות](docs/results.md) | כל 6 הגרפים עם הסבר לכל תרחיש |

---

## What this reproduces

The paper trains two-layer MLPs on pairs of tasks sequentially and measures the trade-off between performance on the old task vs. the new task. We reproduce all three experimental scenarios:

| Scenario | Old Task | New Task | Paper Figure |
|---|---|---|---|
| 1 — Input Reformatting | MNIST | Permuted MNIST | Fig 1–2 |
| 2 — Similar Tasks | Amazon Kitchen reviews | Amazon DVD reviews | Fig 3–4 |
| 3 — Dissimilar Tasks | MNIST (digits 2 & 9) | Amazon DVD reviews | Fig 5–6 |

Each scenario trains **8 conditions** (4 activations × SGD / Dropout):
Sigmoid, ReLU, Maxout, LWTA — each with and without Dropout.

---

## Reproduced figures

| Fig 1 — Input Reformatting Frontier | Fig 2 — Model Sizes |
|---|---|
| ![](paper_figures/Fig1_frontier_input_reformatting.png) | ![](paper_figures/Fig2_model_sizes_input_reformatting.png) |

| Fig 3 — Similar Tasks Frontier | Fig 4 — Model Sizes |
|---|---|
| ![](paper_figures/Fig3_frontier_similar_tasks.png) | ![](paper_figures/Fig4_model_sizes_similar_tasks.png) |

| Fig 5 — Dissimilar Tasks Frontier | Fig 6 — Model Sizes |
|---|---|
| ![](paper_figures/Fig5_frontier_dissimilar_tasks.png) | ![](paper_figures/Fig6_model_sizes_dissimilar_tasks.png) |

---

## Repo structure

```
final_experiment_repro.py   # main experiment — all 3 scenarios
plot_results.py             # generates frontier + model-size plots
prepare_amazon_npz.py       # preprocesses raw Amazon review files → .npz
bonus_improvements.py       # additional ablations
requirements.txt            # pinned dependencies
takeaways.md                # findings and personal reflections
docs/
  methodology.md            # experimental design and deviations from paper
  results.md                # detailed results with all 6 figures
paper_figures/              # final plots named to match paper figures (Fig1–Fig6)
results_repro/              # checkpoints + plots from the reproduction run
```

> **Note:** `data/` is excluded from git. Run `prepare_amazon_npz.py` to generate Amazon files.
> MNIST downloads automatically on first run.

---

## How to run

```bash
# 1 — prepare Amazon data (only needed once)
python prepare_amazon_npz.py

# 2 — run all experiments (auto-checkpoint, auto-resume if interrupted)
python final_experiment_repro.py

# 3 — generate plots
python plot_results.py
```

---

## Key findings

- **Dropout wins** across all 3 scenarios — best trade-off between old-task retention and new-task adaptation.
- **Maxout + Dropout** is the only method that appears on the frontier in all three task pairs.
- Activation function ranking is **task-dependent** — always cross-validate.
- See [full results](docs/results.md) and [takeaways](takeaways.md) for details.
