# Catastrophic Forgetting — PyTorch Reproduction

PyTorch reproduction of:
> **"An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"**
> Goodfellow, Mirza, Xiao, Courville, Bengio (arXiv:1312.6211, 2015)

---

## What this reproduces

The paper trains two-layer MLPs on pairs of tasks sequentially and measures the trade-off between performance on the old task vs. the new task. We reproduce all three experimental scenarios:

| Scenario | Old Task | New Task |
|---|---|---|
| 1 — Input Reformatting | MNIST | Permuted MNIST |
| 2 — Similar Tasks | Amazon Kitchen reviews | Amazon DVD reviews |
| 3 — Dissimilar Tasks | MNIST (digits 2 & 9) | Amazon DVD reviews |

Each scenario trains **8 conditions** (4 activations × SGD / Dropout):
Sigmoid, ReLU, Maxout, LWTA — each with and without Dropout.

---

## Repo structure

```
final_experiment_repro.py   # main experiment — all 3 scenarios
plot_results.py             # generates frontier + model-size plots
prepare_amazon_npz.py       # preprocesses raw Amazon review files → .npz
bonus_improvements.py       # additional ablations
requirements.txt            # dependencies
takeaways.md                # findings and deviations from the paper

paper_figures/              # final plots named to match paper figures
  Fig1_frontier_input_reformatting.png
  Fig2_model_sizes_input_reformatting.png
  Fig3_frontier_similar_tasks.png
  Fig4_model_sizes_similar_tasks.png
  Fig5_frontier_dissimilar_tasks.png
  Fig6_model_sizes_dissimilar_tasks.png

results_repro/              # checkpoints + plots from the reproduction run
  scenario_1_repro.pt
  scenario_3_repro.pt
  scenario_5_repro.pt
  fig_s1_frontier.png  /  fig_s1_params.png
  fig_s3_frontier.png  /  fig_s3_params.png
  fig_s5_frontier.png  /  fig_s5_params.png
```

> **Note:** `data/` (MNIST ~45 MB, Amazon NPZ ~38 MB each) is excluded from git via `.gitignore`.
> Run `prepare_amazon_npz.py` to generate the Amazon files from raw reviews.
> MNIST is downloaded automatically by PyTorch on first run.

---

## How to run

```bash
# 1 — prepare Amazon data (only needed once)
python prepare_amazon_npz.py

# 2 — run all experiments (auto-saves checkpoint after each condition)
python final_experiment_repro.py

# 3 — generate plots from saved checkpoints
python plot_results.py
```

Plots are saved to `results_repro/`. If the run is interrupted it resumes automatically from the last saved checkpoint.

---

## Deviations from the paper

| Parameter | Paper | This repo | Reason |
|---|---|---|---|
| Trials per condition | 25 | 8 | Consumer hardware |
| Early-stopping patience | 100 epochs | 15 epochs | Consumer hardware |
| Framework | Theano / Pylearn2 | PyTorch | Theano is deprecated |

All architectural choices (2 hidden layers, softmax output, max-norm constraint, random hyperparameter search) match the paper exactly.

---

## Results

Plots in `paper_figures/` correspond directly to Figures 1–6 in the paper.
Key finding reproduced: **Dropout consistently dominates** — it achieves the best trade-off between old-task retention and new-task adaptation across all three scenarios.
