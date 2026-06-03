# Workflow — Stages and Verification

## Overview

The project is divided into 6 sequential stages. Each stage has a defined output and a defined verification method.

---

## Stage 1 — Data Preparation

**Goal:** Prepare all datasets so both MNIST and Amazon Reviews are in a format the experiment script can consume.

### Steps

1. **MNIST** — downloaded automatically by PyTorch (`torchvision.datasets.MNIST`) on first run. No manual action needed.
2. **Amazon Reviews** — downloaded manually from the [Sentiment Dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) and placed under `data/amazon/`. Then preprocessed into `.npz` files:

```bash
python prepare_amazon_npz.py
```

This script reads the raw `.review` files, extracts bag-of-words features (top-5000 vocabulary), applies TF-IDF weighting, and saves:
- `data/amazon/<domain>_train.npz`
- `data/amazon/<domain>_test.npz`

For Scenario 3, `TruncatedSVD` is applied to reduce Amazon features from 5000 to 784 dimensions (matching MNIST input size). The SVD is fit on the training set only to prevent data leakage.

### Verification

| Check | Method |
|---|---|
| MNIST loads correctly | Run `python -c "import torchvision; torchvision.datasets.MNIST('./data', download=True)"` — should complete without error |
| `.npz` files created | Check that `data/amazon/<domain>_train.npz` and `_test.npz` exist for all 4 domains (books, dvd, electronics, kitchen) |
| No data leakage in SVD | SVD is fit only on `X_train` — confirmed in `prepare_amazon_npz.py` |
| Shapes correct | Load any `.npz` and print `data['X'].shape` — should be `(N, 5000)` for Scenarios 2, `(N, 784)` for Scenario 3 |

---

## Stage 2 — Model Definitions

**Goal:** Define the 4 activation function variants (Sigmoid, ReLU, Maxout, LWTA) as PyTorch modules.

**Code location:** `final_experiment_repro.py` — classes `MaxoutLayer`, `LWTALayer`, `TwoLayerNet`

### Key design decisions

- **Maxout:** each output unit is the max over `k=2` linear inputs. Pre-activation layer size = `hidden_dim × k`.
- **LWTA:** within each group of `k=2` units, the larger unit passes its gradient; the smaller receives zero gradient.
- **Max-norm constraint:** applied after each optimizer step via `apply_max_norm()` — clips per-column weight norm to a sampled ceiling.
- **Dropout:** `dropout_input=0.2` on the input layer, `dropout_hidden=0.5` on both hidden layers (fixed, not part of HP search).

### Verification

| Check | Method |
|---|---|
| Forward pass runs | Instantiate each model variant and pass a random batch — confirm output shape = `(batch, n_classes)` |
| Max-norm applied | Print weight norms before and after `apply_max_norm()` — norms should not exceed the ceiling |
| LWTA zeroes losers | Manually inspect gradients after backward — losing unit in each pair should have zero grad |
| Dropout active during train | Confirm `model.train()` activates dropout; `model.eval()` deactivates it |

---

## Stage 3 — Hyperparameter Search

**Goal:** For each of the 8 conditions, run 8 random HP trials and select the winning model (lowest best_joint).

**Code location:** `final_experiment_repro.py` — `sample_hparams()`, `run_trial()`, `run_scenario()`

### HP search space

| Parameter | Range |
|---|---|
| Learning rate | `10^U[-2.5, -1.0]` |
| Hidden layer size | `U[250, 2000]` (ReLU/Sigmoid) or `U[250, 1000]` (Maxout/LWTA) |
| Max-norm per layer | `U[1.0, 5.0]` independently per layer |
| Weight initialization range | `10^U[-2.3, -1.0]` |
| Momentum schedule | `0.5 → U[0.5, 0.99]` linearly over training |

### Verification

| Check | Method |
|---|---|
| 8 trials per condition | After run, `len(d['trial_summaries'][cond]) == 8` for all conditions |
| Checkpoint saves correctly | Interrupt the run mid-condition — on restart, it resumes from the last saved condition |
| HP sampling is reproducible | Fix `seed=42` and re-run — identical trial order and results |
| No trial silently fails | Confirm no `best_joint = inf` or `nan` in any trial summary |

---

## Stage 4 — Sequential Training and Frontier Construction

**Goal:** For each trial, train on Task A until convergence, then train on Task B while recording `(old_error, new_error)` at each epoch. Construct the Pareto frontier from all trial point clouds.

**Code location:** `final_experiment_repro.py` — `train_task_a()`, `train_task_b()`, `pareto_lower_left()`

### Training protocol

1. Train on Task A with early stopping (patience=15 epochs on validation loss)
2. Freeze nothing — all weights remain trainable during Task B training
3. At each Task B epoch: evaluate Task A error and Task B error, record the pair as a frontier point
4. After all 8 trials: compute the Pareto lower-left frontier from the union of all point clouds

### Verification

| Check | Method |
|---|---|
| Task A converges before Task B starts | Print Task A validation error at the end of Stage 1 — should be well below 0.5 |
| Forgetting is measurable | Task A error should increase after Task B training begins for SGD conditions |
| Frontier is monotone | All frontier points should satisfy: no point `p` where another point `q` has `q[0] ≤ p[0]` and `q[1] ≤ p[1]` |
| Results saved in checkpoint | Load `.pt` file and confirm `trial_summaries` and `results` keys exist with correct structure |

---

## Stage 5 — Figure Generation

**Goal:** Produce 3 figure types per scenario: Frontier curve, model-size bar chart, and error-bar chart.

```bash
python plot_results.py
```

**Output:** `results_repro/fig_s{1,3,5}_{frontier,params,errorbars}.png`

### Figure types

| Figure | X axis | Y axis | Content |
|---|---|---|---|
| Frontier | Old task error | New task error | Pareto lower-left curve per condition |
| Model sizes | Condition | Parameter count | Winning model size per condition |
| Error bars | Condition | best_joint mean ± std | Variance across 8 trials |

### Verification

| Check | Method |
|---|---|
| All 9 figures generated | Confirm `fig_s1_frontier.png`, `fig_s1_params.png`, `fig_s1_errorbars.png` (×3 scenarios) exist in `results_repro/` |
| Frontier is non-empty | Open each frontier figure — all 8 conditions should appear as labeled curves |
| Frontier curves do not cross unexpectedly | Visually compare to `paper_figures/Fig{1,3,5}_*.png` — qualitative ordering should match |
| Error bars reflect variance | Conditions with high std (e.g. Sigmoid) should show visibly wider bars |

---

## Stage 6 — Ablation Study

**Goal:** Isolate the contribution of Dropout rate and Weight Decay to forgetting resistance, on top of the Scenario 1 baseline.

```bash
python ablation_study.py
```

**Output:** `results_repro/ablation_dropout.png`, `results_repro/ablation_wd.png`, `results_repro/ablation_results.pt`

### Ablation design

Each ablation varies one component at a time, holding everything else fixed (ReLU activation, Scenario 1 data, 8 trials):

| Ablation | Variable | Values tested |
|---|---|---|
| Dropout rate | `dropout_hidden` | 0.0, 0.2, 0.5 |
| Weight decay | L2 regularization λ | 0, 1e-4, 1e-3 |

### Verification

| Check | Method |
|---|---|
| Results saved | `ablation_results.pt` exists and contains keys `dropout`, `weight_decay` |
| Monotonic dropout trend | `forgetting_mean` should decrease as dropout rate increases (0.0 → 0.2 → 0.5) |
| Weight decay effect negligible | Differences across `wd` values should be smaller than 1 std |
| Figures generated | `ablation_dropout.png` and `ablation_wd.png` exist in `results_repro/` |
| No diverged trials | `best_joint_std` should not exceed `best_joint_mean` — a std larger than the mean signals a diverged trial |
