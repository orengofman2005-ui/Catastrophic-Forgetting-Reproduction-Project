# Experimental Methodology and Workflow

## Overview

The project is divided into 6 sequential stages. Each stage has a defined goal, implementation notes, and verification steps.

---

## Stage 1 — Data Preparation

**Goal:** Prepare all datasets so both MNIST and Amazon Reviews are in a format the experiment script can consume.

1. **MNIST** — downloaded automatically by PyTorch on first run. No manual action needed.
2. **Amazon Reviews** — downloaded manually from the [Sentiment Dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) and placed under `data/amazon/`. Then preprocessed into `.npz` files:

```bash
python prepare_amazon_npz.py
```

This script reads the raw `.review` files, extracts bag-of-words features (top-5000 vocabulary), and saves `data/amazon/<domain>.npz` for all 4 domains (books, dvd, electronics, kitchen).

For Scenario 3, `TruncatedSVD` is applied inside `get_amazon_reduced()` to compress Amazon features from 5000 to 784 dimensions — matching MNIST's input size. The SVD is fit on the training set only to prevent data leakage.

### Verification

| Check | Method |
|---|---|
| `.npz` files created | `data/amazon/<domain>.npz` exists for all 4 domains |
| No data leakage in SVD | SVD is fit only on `X_train` — confirmed in `get_amazon_reduced()` |
| Shapes correct | `(N, 5000)` for Scenario 2; `(N, 784)` after SVD for Scenario 3 |

```python
# Verify .npz shapes after running prepare_amazon_npz.py
import numpy as np
for domain in ["books", "dvd", "electronics", "kitchen"]:
    d = np.load(f"data/amazon/{domain}.npz")
    print(f"{domain}: train={d['X_train'].shape}, test={d['X_test'].shape}")
# Expected: (N, 5000) for all domains

# Verify SVD output for Scenario 3
from final_experiment_repro import get_amazon_reduced
_, _, _, dim, n_cls = get_amazon_reduced("data/amazon/dvd.npz", target_dim=784)
print(f"input_dim={dim}, n_classes={n_cls}")  # Expected: 784, 2
```

---

## Stage 2 — Model Architecture

**Goal:** Define the 4 activation function variants (Sigmoid, ReLU, Maxout, LWTA) as PyTorch modules.

**Code location:** `final_experiment_repro.py` — classes `Maxout`, `LWTA`, `MLP`

### 8 Conditions

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

### Key design decisions

- **Maxout:** each output unit is the max over `k=2` linear inputs. Pre-activation layer size = `hidden_dim × k`.
- **LWTA:** within each group of `k=2` units, the larger unit passes its gradient; the smaller receives zero gradient.
- **Max-norm constraint:** applied after each optimizer step — clips per-row weight norm to a per-layer ceiling sampled from `U[1.0, 5.0]`.
- **Dropout:** `dropout_input=0.2` on the input layer, `dropout_hidden=0.5` on both hidden layers (fixed, not part of HP search).

### What is Dropout and why does it matter here?

Dropout (Hinton et al., 2012) is a training technique that randomly silences a random subset of neurons at each training step. Each neuron is independently switched off with probability `p` — so with `p=0.5`, roughly half the network is inactive on any given update. At test time, all neurons are active but their output weights are scaled down by `p` to compensate.

**Why this helps against catastrophic forgetting:**

Without dropout, a network trained with SGD tends to concentrate knowledge in a small number of highly-specialized neurons. When Task B training begins, those neurons get overwritten aggressively, erasing Task A. Dropout prevents this by forcing the network to distribute knowledge across many neurons — no single neuron can become the sole carrier of any piece of information, because it might be silenced at any moment. When Task B comes along and some weights shift, the information about Task A is redundantly stored across enough neurons that much of it survives.

A secondary effect is that dropout enables training of larger networks without overfitting. Larger networks have more spare capacity to accommodate a second task without fully overwriting the first.

**SGD vs Dropout — the core distinction:**

| | SGD | Dropout |
|---|---|---|
| Knowledge storage | Concentrated in few neurons | Distributed across many neurons |
| Optimal model size | Small (to avoid overfitting) | Large (dropout handles regularization) |
| Forgetting | High | Significantly lower |

```python
# Maxout layer — each output unit is the max over k=2 inputs
class Maxout(nn.Module):
    def forward(self, x):
        b, d = x.shape
        return x.view(b, d // self.pool_size, self.pool_size).max(dim=2).values

# LWTA layer — within each group of k=2, only the winner passes gradients
class LWTA(nn.Module):
    def forward(self, x):
        b, d = x.shape
        x_g  = x.view(b, d // self.group_size, self.group_size)
        mask = (x_g >= x_g.max(dim=2, keepdim=True).values).float()
        return (x_g * mask).view(b, d)

# Max-norm constraint — applied once per epoch after optimizer.step()
def apply_max_norm_constraint(model, hp):
    for layer, max_norm in zip([model.fc1, model.fc2, model.fc3],
                                [hp.col_norm_h0, hp.col_norm_h1, hp.col_norm_out]):
        W = layer.weight
        col_norms = W.norm(2, dim=1, keepdim=True)
        W.mul_(col_norms.clamp(max=max_norm) / col_norms.clamp(min=1e-8))
```

### Verification

| Check | Method |
|---|---|
| Forward pass runs | All 8 variants produce output shape `(batch, n_classes)` |
| Max-norm applied | Weight norms do not exceed ceiling after `apply_max_norm_constraint()` |
| LWTA zeroes losers | Losing unit in each pair has zero gradient after backward |
| Dropout active during train | `model.train()` activates dropout; `model.eval()` deactivates it |

```python
# Verify all 4 model variants produce correct output shapes
import torch, random
from final_experiment_repro import build_model, sample_hparams

x   = torch.randn(32, 784)
rng = random.Random(42)

for activation in ["Sigmoid", "ReLU", "Maxout", "LWTA"]:
    for use_dropout in [False, True]:
        hp  = sample_hparams(activation, rng)
        out = build_model(784, 10, activation, use_dropout, hp)(x)
        assert out.shape == (32, 10)
        print(f"{activation} + {'Dropout' if use_dropout else 'SGD'}: OK")
```

---

## Stage 3 — Hyperparameter Search

**Goal:** For each of the 8 conditions, run 8 random HP trials and select the winning model (lowest best_joint).

**Code location:** `final_experiment_repro.py` — `sample_hparams()`, `run_hyperparameter_search()`

### HP search space

| Parameter | Range | Note |
|---|---|---|
| Learning rate | `10^U[-2.5, -1.0]` | Paper does not specify; range chosen empirically |
| Hidden layer size | `U[250, 2000]` (ReLU/Sigmoid) or `U[250, 1000]` (Maxout/LWTA) | Pre-act layer = hidden_dim × k for Maxout/LWTA |
| Max-norm per layer | `U[1.0, 5.0]` independently per layer | Applied via `apply_max_norm_constraint()` |
| Weight initialization range | `10^U[-2.3, -1.0]` | |
| Momentum schedule | `0.5 → U[0.5, 0.99]` linearly over training | |

```python
# sample_hparams() — one random draw per trial
# hidden_dim is capped lower for Maxout/LWTA because pre-activation width = hidden_dim × k
if activation in {"Maxout", "LWTA"}:
    hidden_dim = rng.randint(250, 1000)   # pre-act up to 4000 units (k≤4)
else:
    hidden_dim = rng.randint(250, 2000)   # ReLU/Sigmoid: no expansion

hp = HParams(
    hidden_dim     = hidden_dim,
    lr             = 10 ** rng.uniform(-2.5, -1.0),   # log-uniform
    final_momentum = rng.uniform(0.5, 0.99),
    col_norm_h0    = rng.uniform(1.0, 5.0),           # max-norm per layer
    col_norm_h1    = rng.uniform(1.0, 5.0),
    col_norm_out   = rng.uniform(1.0, 5.0),
    irange         = 10 ** rng.uniform(-2.3, -1.0),   # weight init range
    # ... (momentum schedule, sparse init, bias offsets)
)
```

### Verification

| Check | Method |
|---|---|
| 8 trials per condition | `len(d['trial_summaries'][cond]) == 8` for all conditions |
| Checkpoint saves correctly | Interrupt mid-run — on restart it resumes from the last saved condition |
| HP sampling is reproducible | Fix `seed=42` and re-run — identical trial order and results |
| No trial silently fails | No `best_joint = inf` or `nan` in any trial summary |

```python
# Verify trial counts and check for diverged trials after the run
import torch

d = torch.load("results_repro/scenario_1_repro.pt", weights_only=False)
for cond, trials in d["trial_summaries"].items():
    joints = [t["best_joint"] for t in trials]
    bad    = [j for j in joints if j != j or j == float("inf")]
    print(f"{cond}: {len(trials)} trials, bad={len(bad)}, best={min(joints):.4f}")
```

---

## Stage 4 — Sequential Training and Frontier Construction

**Goal:** For each trial, train on Task A until convergence, then train on Task B while recording `(old_error, new_error)` at each epoch. Construct the Pareto frontier from all trial point clouds.

**Code location:** `final_experiment_repro.py` — `train_task1()`, `train_task2_and_log()`, `pareto_lower_left()`

### Training protocol

1. Train on Task A with early stopping (patience=15 epochs on validation loss)
2. Freeze nothing — all weights remain trainable during Task B training
3. At each Task B epoch: evaluate Task A error and Task B error, record the pair as a frontier point
4. After all 8 trials: compute the Pareto lower-left frontier from the union of all point clouds

```python
# Core sequential training loop (simplified from train_task2_and_log)
for epoch in range(MAX_EPOCHS):
    train_one_epoch(model, t2_train, optimizer, hp, epoch)
    old_val  = evaluate_error(model, t1_val)    # Task A val — used for early stopping
    new_val  = evaluate_error(model, t2_val)    # Task B val — used for early stopping
    old_test = evaluate_error(model, t1_test)   # Task A test — logged for frontier
    new_test = evaluate_error(model, t2_test)   # Task B test — logged for frontier
    trajectory.append((old_test, new_test))
    if old_val + new_val < best_joint:
        best_joint = old_val + new_val
        best_state = copy_weights(model)        # save best joint checkpoint
```

The Pareto frontier is computed from the union of all trial trajectories:

```python
# pareto_lower_left() — core logic
log_pts = np.log10(points)                      # work in log space (matches paper axes)
log_pts = log_pts[np.argsort(log_pts[:, 0])]   # sort left-to-right by old_error

frontier, min_y = [], float("inf")
for p in log_pts:
    if p[1] < min_y:        # only keep points that improve the running y-minimum
        min_y = p[1]
        frontier.append(p)  # this point is not dominated by anything to its left

return 10 ** np.array(frontier)   # back to linear scale
```

### Verification

| Check | Method |
|---|---|
| Task A converges before Task B starts | Task A val error at end of Stage 1 should be well below 0.5 |
| Forgetting is measurable | Task A error should increase after Task B training begins for SGD conditions |
| Frontier is monotone | No dominated points — y values strictly decrease left-to-right |
| Results saved in checkpoint | `.pt` file contains `trial_summaries` and `results` keys |

```python
# Verify the Pareto frontier is strictly monotone (no dominated points)
import torch
from final_experiment_repro import pareto_lower_left

d = torch.load("results_repro/scenario_1_repro.pt", weights_only=False)
for cond, trials in d["trial_summaries"].items():
    all_pts  = [pt for t in trials for pt in t["points"]]
    frontier = pareto_lower_left(all_pts)
    if len(frontier) > 1:
        assert all(frontier[i,1] > frontier[i+1,1] for i in range(len(frontier)-1)), \
            f"Non-monotone frontier in {cond}"
    print(f"{cond}: {len(frontier)} frontier points — OK")
```

---

## Stage 5 — Figure Generation

**Goal:** Produce 2 figure types per scenario: Frontier curve and model-size bar chart.

```bash
python plot_results.py
```

**Output:** `results_repro/fig_s{1,3,5}_{frontier,params}.png`

### Figure types

| Figure | X axis | Y axis | Content |
|---|---|---|---|
| Frontier | Old task error | New task error | Pareto lower-left curve per condition |
| Model sizes | Condition | Parameter count | Winning model size per condition |

### Verification

| Check | Method |
|---|---|
| All 6 figures generated | `fig_s{1,3,5}_{frontier,params}.png` exist in `results_repro/` |
| Frontier is non-empty | All 8 conditions appear as labeled curves |
| Qualitative ordering matches paper | Compare to `paper_figures/Fig{1,3,5}_*.png` |

```python
# Verify all expected figure files were produced
import os
expected = [
    f"results_repro/fig_s{s}_{t}.png"
    for s in [1, 3, 5]
    for t in ["frontier", "params"]
]
for path in expected:
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"{status}  {path}")
```

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

```python
# Forgetting metric used in the ablation (ablation_study.py)
baseline_old = evaluate_error(model, t1_test)   # Task A error before Task B training
# ... train on Task B ...
final_old    = evaluate_error(model, t1_test)   # Task A error after Task B training
forgetting   = final_old - baseline_old         # positive = catastrophic forgetting
```

### Verification

| Check | Method |
|---|---|
| Results saved | `ablation_results.pt` contains keys `dropout`, `weight_decay` |
| Monotonic dropout trend | `forgetting_mean` decreases as dropout increases (0.0 → 0.2 → 0.5) |
| Weight decay effect negligible | Differences across `wd` values smaller than 1 std |
| No diverged trials | `best_joint_std` should not exceed `best_joint_mean` |

```python
# Verify ablation results: monotonic dropout trend + no diverged trials
import torch

abl = torch.load("results_repro/ablation_results.pt", weights_only=False)

prev_forg = float("inf")
for label, v in abl["dropout"].items():
    forg     = v["forgetting_mean"]
    diverged = "DIVERGED" if v["best_joint_std"] > v["best_joint_mean"] else "OK"
    print(f"  {label}: forgetting={forg:.4f}  [{diverged}]")
    assert forg <= prev_forg, f"Non-monotonic forgetting at {label}"
    prev_forg = forg
```

---

## Deviations from the Original Paper

| Parameter | Paper | This Reproduction | Impact on Validity |
|---|---|---|---|
| Trials per condition | 25 | 8 | High — partial HP space coverage |
| Early-stopping patience | 100 epochs | 15 epochs | High — see Patience Bias note below |
| Framework | Theano / Pylearn2 | PyTorch | Medium — numerical precision differences |
| Batch size | 128 | 256 | Medium — different gradient noise |
| Seed | not specified | 42 only | High — no variance estimation |
| Scenario 3 input (Amazon) | Full vocabulary | TruncatedSVD → 784 dims | Medium — see note below |
| Frontier construction | Lower convex hull (algorithm unspecified) | Strict lower-left Pareto frontier | Medium — see note below |

> **Patience Bias:** Patience=15 (vs. 100) creates a systematic bias against Dropout — Dropout converges more slowly than SGD, so short patience gives a relative advantage to SGD. The findings on Dropout superiority are therefore **conservative**: under full patience, the performance gap is expected to be larger.

> **Scenario 3 deviation:** MNIST (784 pixels) and Amazon Reviews (5000+ features) have incompatible input sizes. We apply `TruncatedSVD` to compress Amazon to 784 dims, fit on training data only. Absolute error values in Scenario 3 are not directly comparable to the paper, but method rankings remain valid.

> **Single seed:** Using seed=42 for all experiments means results represent one specific random initialization. Rankings close in value (Δ < 0.005) should be treated as effectively tied.

> **Frontier construction:** The paper uses a lower convex hull (algorithm unspecified). We implemented a strict lower-left Pareto frontier — a point is included only if no other point is simultaneously better on both axes. A convex hull can include dominated points in concave regions; Pareto never does. This means Figures 1, 3, and 5 are not directly comparable to the originals on a point-by-point basis, though the relative ordering of methods remains interpretable.

---

## Checkpoint and Resume

The script saves a checkpoint after each condition. If the run is interrupted it automatically resumes from the last saved point:

```python
# run_hyperparameter_search() — checkpoint logic
ckpt_path = f"results_repro/ckpt_{scenario_name}_{label}.pt"

if os.path.exists(ckpt_path):
    saved = torch.load(ckpt_path, weights_only=False)
    all_results[label]     = saved["results"]
    trial_summaries[label] = saved["trial_summaries"]
    continue   # skip this condition entirely

# ... run trials ...

torch.save({"results": ..., "trial_summaries": ..., "winning_model": ...}, ckpt_path)
```
