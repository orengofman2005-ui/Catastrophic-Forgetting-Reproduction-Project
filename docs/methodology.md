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

## Bonus: Attempted Improvements

Beyond the core reproduction (which is approximate — see Deviations table above), we attempted several improvements over the original paper's setup. These are documented here as the bonus component of the project.

### 1. Expanded Hyperparameter Search Range

Since the original paper does not publish its HP search ranges, our choices (`10^U[-2.5, -1.0]` for LR, `U[250, 1000]` post-activation units for Maxout/LWTA) were determined empirically. An earlier version of this code used narrower ranges; widening the lower LR bound produced slightly more stable Frontier curves in Scenario 1, particularly for Maxout_Dropout.

### 2. Monotonic Frontier Instead of Raw Lower Convex Hull

The original paper appears to use a lower convex hull on a log scale, but the exact method is not specified. We implemented a strict lower-left Pareto frontier in log space (`pareto_lower_left` in `final_experiment_repro.py`), which more cleanly separates dominant solutions from dominated ones. This produces cleaner curves than a naive convex hull, especially when point density is low (8 trials vs. 25).

### 3. Baseline Reference Line

We added a vertical dashed baseline to each Frontier plot marking the median old-task error at the start of new-task training. This is not present in the original paper's figures but makes it immediately visible how much each method degrades relative to the pre-forgetting reference — improving interpretability.

### 4. SVD-Based Feature Reduction for Amazon (Scenario 3) — Approximate Reproduction

In Scenario 3, the original paper feeds Amazon reviews directly into an MLP at full vocabulary dimensionality (~5000+ features). We applied TruncatedSVD to reduce the feature space to 784 dimensions (matching MNIST input size), fitting the SVD on the training set only to avoid data leakage.

> **This is an algorithmic deviation, not merely a technical improvement.** Scenario 3 should therefore be treated as an **approximate reproduction**: the qualitative rankings are preserved, but absolute error values are not directly comparable to the paper. All Scenario 3 findings are reported as qualitative agreement only.

### 5. Per-Condition Checkpointing

The original paper gives no indication of how runs were managed. We added automatic per-condition checkpointing so that if training is interrupted (e.g., power loss, kernel crash), it resumes from the last completed condition rather than from scratch. This is particularly valuable given the 6–12 hour runtime.

**Note on bonus outcomes:** Improvements 3–5 are clearly beneficial (cleaner visualization, no leakage, robustness). Improvements 1–2 produced modest qualitative gains consistent with the paper's conclusions. No improvement reversed or contradicted any finding from the original paper.

---

## Ablation Study

Beyond reproduction, we ran controlled ablation experiments to isolate the mechanisms behind Dropout's forgetting resistance. Two ablations were performed on Scenario 1 (Permuted MNIST):

1. **Dropout Rate** — comparing p=0.0, 0.2, and 0.5 (paper value)
2. **Weight Decay** — comparing none, 1e-4, and 1e-3

See [ablation.md](ablation.md) for full results and interpretation.
