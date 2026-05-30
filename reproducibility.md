# Reproducibility Specification

This file documents the exact hardware and software environment used to produce the results in this project. Anyone wishing to reproduce our results should use a matching or equivalent configuration.

---

## Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Super (6 GB VRAM) |
| CPU | Intel Core i7-10700 (8 cores, 16 threads) |
| RAM | 16 GB DDR4 |
| Storage | SSD (NVMe) — required for fast data loading |
| OS | Windows 11 (64-bit) |

---

## Software

| Package | Version |
|---|---|
| Python | 3.11.4 |
| PyTorch | 2.1.0+cu118 |
| torchvision | 0.16.0+cu118 |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.7.0 |
| numpy | 1.24.3 |
| scikit-learn | 1.3.0 |
| matplotlib | 3.7.2 |
| tqdm | 4.65.0 |

To install all dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn matplotlib tqdm
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Random Seeds

All experiments use a fixed global seed of **42**, set at three levels:

```python
import random, numpy as np, torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)   # for GPU reproducibility
```

This seed is applied once at startup in `final_experiment_repro.py` (module level) and also re-applied per trial via `set_seed(SEED + trial_index)` to ensure each trial is independently reproducible.

> **Note:** Full bit-for-bit reproducibility across different GPU hardware is not guaranteed even with a fixed seed, due to non-deterministic CUDA operations (e.g., atomics in certain cuDNN kernels). Results may vary slightly across GPU models.

---

## Dataset Sources

| Dataset | Source | Access |
|---|---|---|
| MNIST | Yann LeCun, http://yann.lecun.com/exdb/mnist/ | Downloaded automatically by torchvision |
| Amazon Reviews | Mark Dredze, https://www.cs.jhu.edu/~mdredze/datasets/sentiment/ | Manual download required |

### Amazon Preprocessing Steps

1. Download the processed version (unprocessed.tar.gz) from the link above
2. Extract to `data/amazon/` — should produce subdirectories: `books/`, `dvd/`, `electronics/`, `kitchen/`
3. Each subdirectory must contain `positive.review` and `negative.review`
4. Run `python prepare_amazon_npz.py` to generate `.npz` files
5. This creates `data/amazon/{books,dvd,electronics,kitchen}.npz` with keys: `X_train`, `y_train`, `X_test`, `y_test`

---

## Runtime

| Scenario | GPU Runtime | CPU Runtime (estimate) |
|---|---|---|
| Scenario 1 — Input Reformatting | ~3.5 hours | ~12 hours |
| Scenario 2 — Similar Tasks (Amazon) | ~3 hours | ~10 hours |
| Scenario 3 — Dissimilar Tasks | ~3 hours | ~10 hours |
| **Total** | **~9.5 hours** | **~32 hours** |

Runtimes are for 8 trials per condition (64 models total per scenario). With checkpointing enabled, interrupted runs resume automatically — no progress is lost.

---

## How to Verify Reproducibility

After running all experiments, compare your `results_repro/` checkpoint files against the expected best_joint values below:

| Condition | Scenario 1 best_joint | Scenario 2 best_joint | Scenario 3 best_joint |
|---|---|---|---|
| Maxout_Dropout | ~0.039 | ~0.316 | ~0.171 |
| ReLU_Dropout | ~0.044 | ~0.309 | ~0.177 |
| Maxout_SGD | ~0.042 | ~0.341 | ~0.170 |
| Sigmoid_SGD | ~0.173 | ~0.813 | ~0.201 |

Values may differ slightly due to GPU non-determinism, but qualitative ranking (Dropout > SGD, Maxout/ReLU > Sigmoid/LWTA) should be preserved.
