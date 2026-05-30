# Catastrophic Forgetting — PyTorch Reproduction

PyTorch reproduction of:
> **"An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"**
> Goodfellow, Mirza, Xiao, Courville, Bengio (arXiv:1312.6211, 2015)

---

## Research Question

> *Can the central findings of Goodfellow et al. (2015) — in particular, the consistent superiority of Dropout over SGD in mitigating catastrophic forgetting — be reproduced in a modern PyTorch environment under consumer-hardware constraints (8 trials per condition instead of 25)?*

See [Introduction & Research Question](docs/introduction.md) for full background and hypotheses.

---

## Quick links

| Document | Content |
|---|---|
| [Introduction & Research Question](docs/introduction.md) | Background, hypotheses, key definitions |
| [Methodology](docs/methodology.md) | Experimental design, HP search, deviations from paper |
| [Results](docs/results.md) | All 6 figures with per-scenario analysis |
| [Conclusion](docs/conclusion.md) | Quantitative comparison table, final answers, limitations |
| [Takeaways](takeaways.md) | Per-scenario findings and reflections |

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

The vertical dotted line in each frontier plot marks the median old-task error at the start of new-task training (pre-forgetting baseline).

---

## Repo structure

```
final_experiment_repro.py   # main experiment — all 3 scenarios
plot_results.py             # generates frontier + model-size plots
prepare_amazon_npz.py       # preprocesses raw Amazon review files → .npz
bonus_improvements.py       # additional ablations
requirements.txt            # pinned dependencies (Python 3.11)
takeaways.md                # per-scenario findings and reflections
docs/
  introduction.md           # research question, background, hypotheses
  methodology.md            # experimental design and deviations from paper
  results.md                # detailed results with all 6 figures
  conclusion.md             # quantitative comparison, final answers, limitations
paper_figures/              # final plots named to match paper figures (Fig1–Fig6)
results_repro/              # checkpoints + plots from the reproduction run
```

---

## How to run

**Requirements:** Python 3.11, see `requirements.txt`. Estimated runtime: ~6–12 hours on a consumer GPU (RTX 3060 or equivalent); ~24–48 hours on CPU only.

```bash
# 1 — install dependencies
pip install -r requirements.txt

# 2 — prepare Amazon data (only needed once)
#     Download raw Amazon reviews from:
#     https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
#     Place in data/amazon/{books,dvd,electronics,kitchen}/ then run:
python prepare_amazon_npz.py

# 3 — run all experiments (auto-checkpoint, auto-resume if interrupted)
python final_experiment_repro.py

# 4 — generate plots
python plot_results.py
```

MNIST downloads automatically on first run. Amazon data requires manual download (see step 2).

---

## Key findings

- **Dropout consistently achieves superior Frontier performance** across all 3 scenarios — lower joint error than SGD in 6 out of 8 conditions in Scenario 1.
- **Maxout + Dropout** is the only method that appears on the frontier in all three task pairs, consistent with the original paper.
- **Activation function ranking is task-dependent** — no universal winner; cross-validation is essential.
- See [Conclusion](docs/conclusion.md) for quantitative comparison table and full discussion.
