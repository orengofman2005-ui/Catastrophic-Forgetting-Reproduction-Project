# Catastrophic Forgetting — PyTorch Reproduction

Reproduction of:
> **"An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"**
> Goodfellow, Mirza, Xiao, Courville, Bengio — arXiv:1312.6211, 2015

---

## Research Question

> *Can the central conclusions of Goodfellow et al. (2015) — the superiority of Dropout over SGD in preventing catastrophic forgetting, and the scenario-dependent ranking of activation functions — be reproduced in a modern PyTorch environment under consumer hardware constraints (8 trials instead of 25)?*

---

## Quick Navigation

| Document | Contents |
|---|---|
| [Introduction & Research Question](docs/introduction.md) | Background, hypotheses, key definitions, related work |
| [Methodology](docs/methodology.md) | Experiment design, hyperparameter search, deviations from the paper |
| [Results](docs/results.md) | All 6 figures with per-scenario analysis |
| [Conclusion](docs/conclusion.md) | Quantitative comparison table, answer to research question, limitations |
| [Takeaways](takeaways.md) | Per-scenario analysis and reflection |

---

## What Is Reproduced

The paper trains two-layer MLP networks on pairs of tasks sequentially and measures the trade-off between performance on the old task versus the new task. Three original scenarios are reproduced:

| Scenario | Old Task | New Task | Paper Figures |
|---|---|---|---|
| 1 — Input Reformatting | MNIST | Permuted MNIST | Fig 1–2 |
| 2 — Similar Tasks | Amazon Kitchen reviews | Amazon DVD reviews | Fig 3–4 |
| 3 — Dissimilar Tasks | MNIST (digits 2 and 9) | Amazon DVD reviews | Fig 5–6 |

Each scenario includes **8 conditions** (4 activation functions × SGD / Dropout):
Sigmoid, ReLU, Maxout, LWTA — each with and without Dropout.

---

## Reproduced Figures

The gray dashed vertical line in each Frontier plot marks the median old-task error at the start of new-task training — the pre-forgetting reference point.

| Fig 1 — Frontier, Input Reformatting | Fig 2 — Model Sizes |
|---|---|
| ![](paper_figures/Fig1_frontier_input_reformatting.png) | ![](paper_figures/Fig2_model_sizes_input_reformatting.png) |

| Fig 3 — Frontier, Similar Tasks | Fig 4 — Model Sizes |
|---|---|
| ![](paper_figures/Fig3_frontier_similar_tasks.png) | ![](paper_figures/Fig4_model_sizes_similar_tasks.png) |

| Fig 5 — Frontier, Dissimilar Tasks | Fig 6 — Model Sizes |
|---|---|
| ![](paper_figures/Fig5_frontier_dissimilar_tasks.png) | ![](paper_figures/Fig6_model_sizes_dissimilar_tasks.png) |

---

## Repository Structure

```
final_experiment_repro.py   # main experiment — all 3 scenarios
plot_results.py             # generates Frontier and model-size figures
prepare_amazon_npz.py       # preprocesses Amazon data -> .npz
requirements.txt            # dependencies (Python 3.11)
takeaways.md                # per-scenario analysis and reflection
docs/
  introduction.md           # research question, background, hypotheses, related work
  methodology.md            # experimental design and deviations from the paper
  results.md                # detailed results with all 6 figures
  conclusion.md             # quantitative table, conclusions, limitations
paper_figures/              # final figures named to match the paper (Fig1–Fig6)
results_repro/              # checkpoints and figures from the run
```

---

## Running the Code

**Requirements:** Python 3.11, CUDA 11.8+ (optional). Estimated runtime: 6–12 hours on a consumer GPU (RTX 3060 or better), 24–48 hours on CPU only.

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Prepare Amazon data (once only)
#     Download from: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
#     Place in: data/amazon/{books,dvd,electronics,kitchen}/
python prepare_amazon_npz.py

# 3 — Run experiments (automatic checkpoint saving, resumes from last stop)
python final_experiment_repro.py

# 4 — Generate figures
python plot_results.py
```

MNIST data is downloaded automatically on the first run.

---

## Key Findings

- **Dropout shows superior Frontier curves** over SGD in all 3 scenarios — lower best_joint in 6 out of 8 conditions in Scenario 1.
- **Maxout + Dropout** is the only condition appearing on the Frontier in all three scenarios — consistent with the original paper's claim.
- **Activation function ranking is scenario-dependent** — there is no universal winner; cross-validation is essential.
- For full quantitative details see [Conclusion](docs/conclusion.md).
