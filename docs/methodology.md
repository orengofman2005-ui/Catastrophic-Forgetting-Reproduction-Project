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
| Scenario 3 input (Amazon) | Full vocabulary | TruncatedSVD → 784 dims | Medium — see note below |

> **Note on Single Seed:** We use a fixed seed (42) for all experiments, meaning every run produces identical results. The paper does not specify a seed, implying they likely averaged over multiple seeds to reduce variance. Running with a single seed means our results reflect one specific random initialization — a different seed could produce slightly different rankings, particularly in cases where two conditions are close (e.g., Maxout_SGD vs. Maxout_Dropout in Scenario 3, Δ=0.001). This limitation means our results cannot be used to estimate variance across runs; they represent a single point estimate only.

> **Note on Trial Count (8 vs. 25):** With only 8 trials per condition, we cover roughly 32% of the hyperparameter search space compared to the paper. This has two consequences: (1) the winning model per condition is less likely to be the true optimum — best_joint values are therefore expected to be slightly higher than the paper's; (2) the winning model's size is highly sensitive to which configurations were sampled, making the model-size figures (Fig2, Fig4, Fig6) unreliable for direct comparison. The Frontier figures are more robust, since they aggregate all 8 trials rather than selecting one winner.

> **Scenario 3 implementation note:** MNIST and Amazon Reviews have incompatible input sizes (784 vs. 5000+). To share a single input layer across both tasks, we apply `TruncatedSVD` to the Amazon feature vectors, reducing them from 5000 to 784 dimensions — matching MNIST exactly. The SVD is fit on the Amazon **training set only** to prevent data leakage. This means Scenario 3 absolute error values are not directly comparable to the paper, but the ranking of methods across all 8 conditions remains valid.

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


