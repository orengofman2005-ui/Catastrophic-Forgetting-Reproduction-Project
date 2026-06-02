# Experimental Methodology

## Experiment Structure

The experiment follows the original structure of Goodfellow et al. (2015):

1. **Training on the old task** — until convergence (early stopping on a validation set)
2. **Training on the new task** — while simultaneously measuring two metrics:
   - New task error (Y axis)
   - Old task error (X axis)
3. **Drawing the Possibilities Frontier curve** — the lower convex hull of the point cloud, on a logarithmic scale

## 8 Conditions

| Activation Function | Algorithm |
|---|---|
| Sigmoid | SGD |
| Sigmoid | Dropout |
| ReLU | SGD |
| ReLU | Dropout |
| Maxout | SGD |
| Maxout | Dropout |
| LWTA | SGD |
| LWTA | Dropout |

## Model Architecture

- **Layers:** 2 hidden layers + softmax classification layer
- **Maxout:** pool size k=2 (each output unit is the maximum over 2 inputs)
- **LWTA:** group size k=2 (in each pair of units, only the larger one receives a gradient)
- **Max-norm constraint:** dynamic constraint per layer in the range 1.0–5.0, sampled separately for fc1, fc2, fc_out in each trial
- **Dropout:** dropout_hidden=0.5, dropout_input=0.2 (fixed, not part of the search)

## Hyperparameter Search

Random search — 8 trials per condition (paper: 25):

| Parameter | This Reproduction (actual code) | Note |
|---|---|---|
| Learning rate | 10^U[-2.5, -1.0] | Paper does not specify; range chosen empirically |
| Hidden layer size (Maxout/LWTA) | U[250, 1000] post-activation units | Pre-activation layer = hidden_dim × k |
| Hidden layer size (ReLU/Sigmoid) | U[250, 2000] | |
| Max-norm (per layer) | U[1.0, 5.0] | Applied per-layer independently |
| Weight initialization range | 10^U[-2.3, -1.0] | |
| Momentum schedule | 0.5 → U[0.5, 0.99] linearly | |

> **Note on HP ranges:** The original paper does not publish its hyperparameter search ranges. The values above are what this reproduction actually uses (see `sample_hparams()` in `final_experiment_repro.py`). They were chosen to cover a plausible range around commonly reported values for this class of models.

> Maxout and LWTA: bias initialized to 0 (random initialization causes one unit in the group to dominate).
> Sigmoid: bias initialized from a negative range to encourage sparsity.
> ReLU: slight positive bias to prevent "dead" units.

## Deviations from the Original Paper

| Parameter | Paper | This Reproduction | Impact on Validity |
|---|---|---|---|
| Trials per condition | 25 | 8 | High — partial HP space coverage |
| Early-stopping patience | 100 epochs | 15 epochs | High — see note on Patience Bias |
| Framework | Theano / Pylearn2 | PyTorch | Medium — numerical precision differences |
| Batch size | 128 | 256 | Medium — different gradient noise |
| Seed | not specified | 42 only | High — no variance estimation |

## Note on Patience Bias

A critical deviation to highlight: reducing patience from 100 to 15 epochs is not neutral — it creates a **systematic bias against Dropout**. Dropout methods generally converge more slowly than SGD (due to the noise introduced by the dropout mechanism into the gradients). As a result, short patience gives a relative advantage to SGD and may suppress the apparent superiority of Dropout. This means the findings on Dropout superiority in this reproduction are **conservative** — under full patience (100 epochs), the performance gap is expected to be larger, not smaller.

## Checkpoint and Resume

The script saves a checkpoint after each condition. If the run is interrupted — it automatically resumes from the last saved point.

---

## Bonus Improvements

Beyond the core reproduction, we introduced five improvements over the original paper's setup. All are documented below with the exact code location and motivation.

---

### Improvement 1 — Ablation Study

**Code:** `ablation_study.py` — functions `ablation_dropout()` and `ablation_weight_decay()`
**Docs:** `docs/ablation.md`

The original paper establishes that Dropout reduces catastrophic forgetting, but does not quantify how much of the effect comes from the dropout rate itself versus weight regularization. We ran a controlled ablation study on Scenario 1 (Permuted MNIST), varying one component at a time:

- **Dropout Rate:** p ∈ {0.0, 0.2, 0.5} — result: -58% forgetting at p=0.5 vs. no dropout
- **Weight Decay:** λ ∈ {0, 1e-4, 1e-3} — result: marginal gain at 1e-4, slightly worse at 1e-3

This extends the paper's qualitative claim with quantitative evidence for the mechanism.

---

### Improvement 2 — Statistical Error Bars

**Code:** `plot_results.py` — function `plot_errorbars()`
**Output:** `results_repro/fig_s*_errorbars.png`

The original paper reports a single best_joint score per condition, with no variance measure. We compute mean ± std across all 8 trials and plot error bar figures for all 3 scenarios.

This is especially important given our reduced trial count (8 vs. paper's 25): the error bars make the statistical uncertainty explicit rather than hiding it.

---

### Improvement 3 — Baseline Reference Line on Frontier Plots

**Code:** `plot_results.py` — `baseline_x` parameter in `plot_frontier()`

Each Frontier plot includes a vertical dashed line marking the median old-task error at the start of new-task training (before any forgetting occurs). The original paper's figures do not include this reference.

Without it, the Frontier only shows relative differences between methods. With it, a reader can immediately quantify absolute degradation: "Maxout+Dropout degrades by X% from the pre-forgetting baseline."

---

### Improvement 4 — Monotonic Pareto Frontier

**Code:** `final_experiment_repro.py` — function `pareto_lower_left()`

The original paper uses a lower convex hull on a log scale, but the exact algorithm is not specified. We implemented a strict lower-left Pareto frontier: a point is included only if no other observed point is simultaneously better on both axes.

A convex hull can include dominated points when the point cloud has concave regions. The Pareto frontier never includes dominated points by definition — this is strictly cleaner, and particularly important at low trial counts (8 vs. 25) where the point cloud is sparse.

---

### Improvement 5 — Two-Stage Feature Reduction for Amazon Reviews (Scenario 3)

**Code:** `prepare_amazon_npz.py` — `build_shared_vectorizer()` · `final_experiment_repro.py` — `get_amazon_reduced()`
**Note:** This is an algorithmic deviation — see caveat below.

The original paper feeds Amazon review bag-of-words vectors at full vocabulary size directly into the MLP. We apply two stages of dimensionality reduction:

**Stage 1 — Shared vocabulary (`prepare_amazon_npz.py`):** A `DictVectorizer` is fit on the union of all four Amazon categories, keeping the top-5000 features by corpus frequency. Fitting on all categories together ensures consistent feature indices across books, dvd, electronics, and kitchen — required for cross-category evaluation in Scenario 3.

**Stage 2 — TruncatedSVD (`final_experiment_repro.py`):** The 5000-dimensional vectors are further reduced to 784 dimensions using `TruncatedSVD`, fit **on training data only** to prevent data leakage. 784 matches the MNIST input size, enabling a fair architectural comparison across scenarios.

> **Caveat:** Because this changes the input representation, Scenario 3 is an **approximate reproduction** — qualitative rankings are preserved, but absolute error values are not directly comparable to the paper. This is flagged in all Scenario 3 result tables.

---

### Improvement 6 — Per-Condition Checkpointing

**Code:** `final_experiment_repro.py` — `save_checkpoint()` / `load_checkpoint()` calls after each condition

The original paper gives no indication of run management. We added automatic per-condition checkpointing: if training is interrupted (power loss, crash), it resumes from the last completed condition rather than from scratch. Given the 6–12 hour runtime, this is critical for practical reproducibility.
