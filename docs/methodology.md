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

### Improvement 2 — Monotonic Pareto Frontier

**Code:** `final_experiment_repro.py` — function `pareto_lower_left()`

The original paper uses a lower convex hull on a log scale, but the exact algorithm is not specified. We implemented a strict lower-left Pareto frontier: a point is included only if no other observed point is simultaneously better on both axes.

A convex hull can include dominated points when the point cloud has concave regions. The Pareto frontier never includes dominated points by definition — this is strictly cleaner, and particularly important at low trial counts (8 vs. 25) where the point cloud is sparse.

---

### Improvement 3 — Shared Vocabulary Feature Selection for Amazon Reviews

**Code:** `prepare_amazon_npz.py` — `build_shared_vectorizer()` · `final_experiment_repro.py` — `get_amazon_from_npz()`, `get_padded_binary_mnist_loaders()`

The original paper feeds Amazon review bag-of-words vectors at full vocabulary size directly into the MLP. We match this approach as closely as possible:

**Stage 1 — Shared vocabulary (`prepare_amazon_npz.py`):** A `DictVectorizer` is fit on the union of all four Amazon categories, keeping the top-5000 features by corpus frequency. Fitting on all categories together ensures consistent feature indices across books, dvd, electronics, and kitchen — required for cross-category evaluation in Scenario 3.

**Stage 2 — Zero-padding MNIST (`final_experiment_repro.py`, Scenario 3 only):** In Scenario 3, MNIST images (784 pixels) are zero-padded to 5000 dimensions to match Amazon's feature size — the same approach the original paper used. The first 784 values are pixel intensities; positions 784–4999 are always zero. Both tasks share a 5000-dimensional input layer. Scenario 2 uses the 5000-dimensional Amazon vectors directly, since both tasks (Kitchen and DVD) already share the same dimension.

---

### Improvement 6 — Per-Condition Checkpointing

**Code:** `final_experiment_repro.py` — `save_checkpoint()` / `load_checkpoint()` calls after each condition

The original paper gives no indication of run management. We added automatic per-condition checkpointing: if training is interrupted (power loss, crash), it resumes from the last completed condition rather than from scratch. Given the 6–12 hour runtime, this is critical for practical reproducibility.
