"""
final_experiment_repro.py
=========================
Faithful reproduction of:
  "An Empirical Investigation of Catastrophic Forgetting in
   Gradient-Based Neural Networks"
  Goodfellow, Mirza, Xiao, Courville, Bengio (arXiv:1312.6211v3, 2015)

Colab-ready version:
  - saves a checkpoint after every condition (not only at end of scenario)
  - if it crashes mid-run — resumes automatically from where it stopped
  - real-time progress bars in CMD / Colab
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# =============================================================================
# Basic settings
# =============================================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results_repro"
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRIALS_PER_CONDITION = 8     # paper: 25 — reduced for home environment (covers HP space)
PATIENCE_OLD = 15            # paper: 100 — sufficient with LR/momentum that converge quickly
PATIENCE_NEW = 15
MAX_EPOCHS_OLD = 150         # with patience=15 almost no model reaches 150
MAX_EPOCHS_NEW = 150
BATCH_SIZE = 256             # better GPU utilization; fewer batches → faster epoch

# total conditions: 4 activations × 2 algorithms = 8
# total models per scenario: 8 × 25 = 200
TOTAL_CONDITIONS = 8
TOTAL_MODELS_PER_SCENARIO = TOTAL_CONDITIONS * TRIALS_PER_CONDITION


# =============================================================================
# Section 1 – Utility helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def apply_max_norm_constraint(model: nn.Module, hp: "HParams") -> None:
    norms_cfg = [hp.col_norm_h0, hp.col_norm_h1, hp.col_norm_out]
    with torch.no_grad():
        for layer, max_norm in zip([model.fc1, model.fc2, model.fc3], norms_cfg):
            W = layer.weight
            col_norms = W.norm(2, dim=1, keepdim=True)
            W.mul_(col_norms.clamp(max=max_norm) / col_norms.clamp(min=1e-8))
            # Clip bias element-wise by the same bound to prevent unbounded growth
            layer.bias.data.clamp_(-max_norm, max_norm)


def evaluate_error(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total   += y.numel()
    return 1.0 - correct / max(total, 1)


def _split_dataset(dataset, n_val: int):
    n   = len(dataset)
    idx = torch.randperm(n).tolist()
    return Subset(dataset, idx[n_val:]), Subset(dataset, idx[:n_val])


# =============================================================================
# Section 2 – Data loaders
# =============================================================================

_PIN = torch.cuda.is_available()
_NW  = 0  # Windows + lambda transforms can't be pickled for workers; MNIST is small enough


class _PermuteFlat:
    """Picklable replacement for transforms.Lambda(lambda x: x.view(-1)[perm])."""
    def __init__(self, perm): self.perm = perm
    def __call__(self, x):    return x.view(-1)[self.perm]


def get_permuted_mnist_loaders(permutation, batch_size=BATCH_SIZE, val_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        _PermuteFlat(permutation),
    ])
    full_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds    = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_ds, val_ds = _split_dataset(full_train, n_val=val_size)
    kw = dict(pin_memory=_PIN, num_workers=_NW, persistent_workers=_NW > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


def get_padded_binary_mnist_loaders(target_dim, classes=(2, 9),
                                    batch_size=BATCH_SIZE, val_size=1000):
    def remap(y): return 0 if y == classes[0] else 1
    pad_len = target_dim - 784
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x, p=pad_len: F.pad(x.view(-1), (0, p))),
    ])
    full_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    def to_td(subset):
        xs, ys = [], []
        for x, y in subset:
            xs.append(x); ys.append(remap(int(y)))
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    tr_idx   = [i for i, y in enumerate(full_train.targets.tolist()) if y in classes]
    te_idx   = [i for i, y in enumerate(test_full.targets.tolist())  if y in classes]
    train_td = to_td(Subset(full_train, tr_idx))
    test_td  = to_td(Subset(test_full,  te_idx))
    train_td, val_td = _split_dataset(train_td, n_val=val_size)
    kw = dict(pin_memory=_PIN, num_workers=_NW, persistent_workers=_NW > 0)
    return (
        DataLoader(train_td, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_td,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_td,  batch_size=batch_size, shuffle=False, **kw),
    )


def get_amazon_from_npz(npz_path, batch_size=BATCH_SIZE, val_ratio=0.2):
    data      = np.load(npz_path, allow_pickle=True)
    X_tr_full = torch.tensor(data["X_train"], dtype=torch.float32)
    y_tr_full = torch.tensor(data["y_train"], dtype=torch.long)
    X_te      = torch.tensor(data["X_test"],  dtype=torch.float32)
    y_te      = torch.tensor(data["y_test"],  dtype=torch.long)
    n_val     = max(1, int(val_ratio * len(X_tr_full)))
    idx       = torch.randperm(len(X_tr_full))
    vi, ti    = idx[:n_val], idx[n_val:]
    kw = dict(pin_memory=_PIN, num_workers=_NW, persistent_workers=_NW > 0)
    return (
        DataLoader(TensorDataset(X_tr_full[ti], y_tr_full[ti]), batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(TensorDataset(X_tr_full[vi], y_tr_full[vi]), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(TensorDataset(X_te, y_te),                   batch_size=batch_size, shuffle=False, **kw),
        X_tr_full.shape[1],
        int(torch.unique(y_tr_full).numel()),
    )


def get_amazon_reduced(npz_path, target_dim, batch_size=BATCH_SIZE, val_ratio=0.2):
    """Load Amazon data, apply TruncatedSVD on raw arrays (before any DataLoader),
    then build loaders. This is the correct order: split → fit SVD on train only →
    transform val/test. The old reduce_feature_dim gathered from already-shuffled
    loaders, making the pipeline hard to audit and the split non-reproducible."""
    data      = np.load(npz_path, allow_pickle=True)
    X_tr_full = data["X_train"].astype(np.float32)
    y_tr_full = data["y_train"].astype(np.int64)
    X_te      = data["X_test"].astype(np.float32)
    y_te      = data["y_test"].astype(np.int64)

    rng   = np.random.default_rng(SEED)
    idx   = rng.permutation(len(X_tr_full))
    n_val = max(1, int(val_ratio * len(X_tr_full)))
    vi, ti = idx[:n_val], idx[n_val:]
    X_va, y_va = X_tr_full[vi], y_tr_full[vi]
    X_tr, y_tr = X_tr_full[ti], y_tr_full[ti]

    svd  = TruncatedSVD(n_components=target_dim, random_state=SEED)
    X_tr = svd.fit_transform(X_tr)   # fit on train only — no leakage
    X_va = svd.transform(X_va)
    X_te = svd.transform(X_te)

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        kw = dict(pin_memory=_PIN, num_workers=_NW, persistent_workers=_NW > 0)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)

    n_cls = int(np.unique(y_tr_full).size)
    return to_loader(X_tr, y_tr, True), to_loader(X_va, y_va, False), \
           to_loader(X_te, y_te, False), target_dim, n_cls


# =============================================================================
# Section 3 – Network architectures
# =============================================================================

class Maxout(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    def forward(self, x):
        b, d = x.shape
        return x.view(b, d // self.pool_size, self.pool_size).max(dim=2).values


class LWTA(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size
    def forward(self, x):
        b, d  = x.shape
        x_g   = x.view(b, d // self.group_size, self.group_size)
        noise = torch.rand_like(x_g) * 1e-6
        mask  = ((x_g + noise) >= (x_g + noise).max(dim=2, keepdim=True).values).float()
        mask  = (mask / mask.sum(dim=2, keepdim=True).clamp(min=1) > 0).float()
        return (x_g * mask).view(b, d)


class MLP(nn.Module):
    P_VIS = 0.2
    P_HID = 0.5

    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation, use_dropout, pool_size=2, hp=None):
        super().__init__()
        self.activation_name = activation
        self.drop_in  = nn.Dropout(self.P_VIS) if use_dropout else nn.Identity()
        self.drop_hid = nn.Dropout(self.P_HID) if use_dropout else nn.Identity()
        expansion = pool_size if activation in {"Maxout", "LWTA"} else 1
        pre_dim   = hidden_dim * expansion
        # After Maxout: output dim = hidden_dim  (max reduces by pool_size)
        # After LWTA:   output dim = pre_dim     (same size, half zeroed)
        # After ReLU/Sigmoid: pre_dim == hidden_dim (expansion=1)
        post_act_dim = pre_dim if activation == "LWTA" else hidden_dim
        self.fc1  = nn.Linear(input_dim,     pre_dim)
        self.fc2  = nn.Linear(post_act_dim,  pre_dim)
        self.fc3  = nn.Linear(post_act_dim,  output_dim)
        acts = {
            "ReLU":    (nn.ReLU(),        nn.ReLU()),
            "Sigmoid": (nn.Sigmoid(),      nn.Sigmoid()),
            "Maxout":  (Maxout(pool_size), Maxout(pool_size)),
            "LWTA":    (LWTA(pool_size),   LWTA(pool_size)),
        }
        if activation not in acts:
            raise ValueError(f"Unknown activation: {activation}")
        self.act1, self.act2 = acts[activation]
        self._init_weights(hp or {})

    def _init_weights(self, hp: dict):
        irange      = hp.get("irange", 0.005)
        bias_h0     = hp.get("bias_h0", 0.0)
        bias_h1     = hp.get("bias_h1", 0.0)
        sparse_init = hp.get("sparse_init", False)
        sparse_k    = hp.get("sparse_k", 15)
        for layer, bias_val in [(self.fc1, bias_h0), (self.fc2, bias_h1), (self.fc3, 0.0)]:
            nn.init.constant_(layer.bias, bias_val)
            if sparse_init and layer is not self.fc3:
                nn.init.zeros_(layer.weight)
                fan_in   = layer.weight.size(1)
                k_actual = min(sparse_k, fan_in)
                for row in range(layer.weight.size(0)):
                    cols = torch.randperm(fan_in)[:k_actual]
                    layer.weight.data[row, cols] = torch.randn(k_actual) * irange * 10
            else:
                nn.init.uniform_(layer.weight, -irange, irange)

    def forward(self, x):
        x = self.drop_in(x)
        if self.activation_name == "LWTA":
            # Dropout before the winner selection so the competition runs on a
            # thinned pre-activation — avoids double-killing losing units.
            x = self.act1(self.drop_hid(self.fc1(x)))
            x = self.act2(self.drop_hid(self.fc2(x)))
        else:
            x = self.drop_hid(self.act1(self.fc1(x)))
            x = self.drop_hid(self.act2(self.fc2(x)))
        return self.fc3(x)


# =============================================================================
# Section 4 – Hyperparameter sampling
# =============================================================================

@dataclass
class HParams:
    hidden_dim:     int
    lr:             float
    init_momentum:  float
    final_momentum: float
    momentum_sat:   int
    lr_sat:         int
    lr_decay:       float
    col_norm_h0:    float
    col_norm_h1:    float
    col_norm_out:   float
    k:              int
    irange:         float
    sparse_init:    bool
    sparse_k:       int
    bias_h0:        float
    bias_h1:        float


def sample_hparams(activation: str, rng: random.Random) -> HParams:
    # hidden_dim is always the POST-activation width.
    # For Maxout/LWTA the pre-activation layer is hidden_dim * k, so we cap
    # hidden_dim tighter there to keep parameter counts sane (≤ ~5M).
    if activation in {"Maxout", "LWTA"}:
        hidden_dim = rng.randint(250, 1000)   # pre-act up to 4 000 (k≤4)
        k          = rng.randint(2, 4)
    else:
        hidden_dim = rng.randint(250, 2000)   # fc1/fc2 up to 2 000 units
        k          = 2

    hp = dict(
        hidden_dim     = hidden_dim,
        # LR: 0.003–0.1  (log-uniform). Upper tail cut: 0.316 + momentum blows up.
        lr             = 10 ** rng.uniform(-2.5, -1.0),
        init_momentum  = 0.5,
        final_momentum = rng.uniform(0.5, 0.99),
        # Saturation epochs must be reachable within the patience window (~20-60 ep).
        momentum_sat   = rng.randint(2, 40),
        lr_sat         = rng.randint(20, 100),
        lr_decay       = 10 ** rng.uniform(-3.0, -1.0),
        col_norm_h0    = rng.uniform(1.0, 5.0),
        col_norm_h1    = rng.uniform(1.0, 5.0),
        col_norm_out   = rng.uniform(1.0, 5.0),
    )
    if activation in {"Maxout", "LWTA"}:
        hp.update(k=k, irange=10 ** rng.uniform(-2.3, -1.0),
                  sparse_init=False, sparse_k=0, bias_h0=0.0, bias_h1=0.0)
    else:
        hp.update(k=k, irange=10 ** rng.uniform(-2.3, -1.0),
                  sparse_init=rng.random() < 0.5,
                  sparse_k=rng.randint(10, 30),
                  bias_h0=rng.uniform(0.0, 0.3) if rng.random() < 0.5 else 0.0,
                  bias_h1=rng.uniform(0.0, 0.3) if rng.random() < 0.5 else 0.0)
    return HParams(**hp)


def _get_lr_momentum(hp: HParams, epoch: int) -> Tuple[float, float]:
    frac_lr  = min(1.0, epoch / max(hp.lr_sat, 1))
    lr       = hp.lr * (1 - frac_lr) + hp.lr * hp.lr_decay * frac_lr
    frac_m   = min(1.0, epoch / max(hp.momentum_sat, 1))
    momentum = hp.init_momentum + (hp.final_momentum - hp.init_momentum) * frac_m
    return lr, momentum


def build_model(input_dim, output_dim, activation, use_dropout, hp):
    return MLP(input_dim, hp.hidden_dim, output_dim,
               activation, use_dropout, pool_size=hp.k,
               hp=hp.__dict__).to(DEVICE)


# =============================================================================
# Section 5 – Training loops
# =============================================================================

def train_one_epoch(model, loader, optimizer, hp, epoch, epoch_bar=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    lr, momentum = _get_lr_momentum(hp, epoch)
    for pg in optimizer.param_groups:
        pg['lr']       = lr
        pg['momentum'] = momentum
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    apply_max_norm_constraint(model, hp)  # once per epoch, not per batch
    if epoch_bar is not None:
        epoch_bar.update(1)


def train_task1(model, train_ldr, val_ldr, optimizer, hp,
                max_epochs=MAX_EPOCHS_OLD, patience=PATIENCE_OLD,
                desc="Task1"):
    best_val, best_state, stale = float("inf"), {k: v.clone() for k, v in model.state_dict().items()}, 0
    # progress bar over epochs — updated every epoch
    with tqdm(total=max_epochs, desc=f"    {desc}", unit="ep",
              leave=False, dynamic_ncols=True) as pbar:
        for epoch in range(max_epochs):
            train_one_epoch(model, train_ldr, optimizer, hp, epoch)
            val_err = evaluate_error(model, val_ldr)
            pbar.set_postfix(val_err=f"{val_err:.4f}", stale=stale)
            pbar.update(1)
            if val_err < best_val:
                best_val, best_state, stale = val_err, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


def train_task2_and_log(model, t1_val, t1_test, t2_train, t2_val, t2_test,
                        optimizer, hp,
                        max_epochs=MAX_EPOCHS_NEW, patience=PATIENCE_NEW,
                        desc="Task2"):
    best_joint, best_state, stale = float("inf"), {k: v.clone() for k, v in model.state_dict().items()}, 0
    trajectory = []
    LOG_INTERVAL = 1  # every epoch — MLP is small, eval is negligible; Pareto needs density
    with tqdm(total=max_epochs, desc=f"    {desc}", unit="ep",
              leave=False, dynamic_ncols=True) as pbar:
        for epoch in range(max_epochs):
            train_one_epoch(model, t2_train, optimizer, hp, epoch)
            old_val  = evaluate_error(model, t1_val)
            new_val  = evaluate_error(model, t2_val)
            joint    = old_val + new_val
            if epoch % LOG_INTERVAL == 0:
                old_test = evaluate_error(model, t1_test)
                new_test = evaluate_error(model, t2_test)
                trajectory.append((old_test, new_test))
                pbar.set_postfix(old=f"{old_test:.3f}", new=f"{new_test:.3f}", stale=stale)
            pbar.update(1)
            if joint < best_joint:
                best_joint, best_state, stale = joint, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return trajectory, best_joint


# =============================================================================
# Section 6 – Hyperparameter search
# =============================================================================

ACTIVATIONS   = ["Sigmoid", "ReLU", "Maxout", "LWTA"]
DROPOUT_FLAGS = [False, True]


def _condition_ckpt_path(scenario_name: str, label: str) -> str:
    safe_label = label.replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"ckpt_{scenario_name}_{safe_label}.pt")


def run_hyperparameter_search(scenario_name, t1_train, t1_val, t1_test,
                               t2_train, t2_val, t2_test,
                               input_dim, output_dim,
                               trials_per_condition=TRIALS_PER_CONDITION):
    all_results, trial_summaries, winning_models = {}, {}, {}

    # main progress bar — over all models in the scenario
    total_models = TOTAL_CONDITIONS * trials_per_condition
    main_bar = tqdm(total=total_models, desc=f"[{scenario_name}] total models",
                    unit="model", dynamic_ncols=True, colour="green")

    for activation in ACTIVATIONS:
        for use_dropout in DROPOUT_FLAGS:
            label     = f"{activation}_{'Dropout' if use_dropout else 'SGD'}"
            ckpt_path = _condition_ckpt_path(scenario_name, label)

            if os.path.exists(ckpt_path):
                tqdm.write(f"  ✅ Skipping {label} — loaded from checkpoint")
                saved = torch.load(ckpt_path, weights_only=False)
                all_results[label]     = saved["results"]
                trial_summaries[label] = saved["trial_summaries"]
                winning_models[label]  = saved["winning_model"]
                main_bar.update(trials_per_condition)
                continue

            tqdm.write(f"\n── {label} ──────────────────────────")
            rng = random.Random(SEED)
            best_joint_global, best_param_count = float("inf"), 0
            trials = []

            # secondary progress bar — over the trials within a condition
            trial_bar = tqdm(range(trials_per_condition),
                             desc=f"  {label}", unit="trial",
                             leave=False, dynamic_ncols=True)

            for trial in trial_bar:
                set_seed(SEED + trial)
                hp    = sample_hparams(activation, rng)
                model = build_model(input_dim, output_dim, activation, use_dropout, hp)
                opt   = optim.SGD(model.parameters(), lr=hp.lr, momentum=hp.init_momentum)

                model = train_task1(model, t1_train, t1_val, opt, hp,
                                    desc=f"T1 trial{trial+1}")
                traj, best_joint = train_task2_and_log(
                    model, t1_val, t1_test, t2_train, t2_val, t2_test, opt, hp,
                    desc=f"T2 trial{trial+1}")

                pc = count_parameters(model)
                trial_bar.set_postfix(joint=f"{best_joint:.4f}", params=f"{pc:,}")
                trials.append({"points": traj, "hp": hp.__dict__,
                               "best_joint": best_joint, "param_count": pc})
                if best_joint < best_joint_global:
                    best_joint_global, best_param_count = best_joint, pc
                main_bar.update(1)

            trial_bar.close()

            all_results[label]     = [pt for t in trials for pt in t["points"]]
            trial_summaries[label] = trials
            winning_models[label]  = best_param_count

            torch.save({
                "results":         all_results[label],
                "trial_summaries": trial_summaries[label],
                "winning_model":   winning_models[label],
            }, ckpt_path)
            tqdm.write(f"  💾 Checkpoint saved: {ckpt_path}")

    main_bar.close()
    return {"scenario_name": scenario_name, "results": all_results,
            "trial_summaries": trial_summaries, "winning_models": winning_models}


# =============================================================================
# Section 7 – Visualisation
# =============================================================================

STYLE_MAP = {
    "Sigmoid_SGD":     {"color": "blue",    "marker": "D", "label": "SGD, Sigmoid"},
    "Sigmoid_Dropout": {"color": "cyan",    "marker": "D", "label": "Dropout, Sigmoid"},
    "ReLU_SGD":        {"color": "red",     "marker": "s", "label": "SGD, ReLUs"},
    "ReLU_Dropout":    {"color": "magenta", "marker": "s", "label": "Dropout, ReLUs"},
    "Maxout_SGD":      {"color": "green",   "marker": "^", "label": "SGD, Maxout"},
    "Maxout_Dropout":  {"color": "lime",    "marker": "^", "label": "Dropout, Maxout"},
    "LWTA_SGD":        {"color": "black",   "marker": ">", "label": "SGD, LWTA"},
    "LWTA_Dropout":    {"color": "gray",    "marker": ">", "label": "Dropout, LWTA"},
}


def pareto_lower_left(points) -> np.ndarray:
    pts = np.array(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 0] > 0) & (pts[:, 1] > 0)]
    if len(pts) == 0:
        return np.empty((0, 2))
    log_pts = np.log10(pts)
    idx     = np.lexsort((log_pts[:, 1], log_pts[:, 0]))
    log_pts = log_pts[idx]
    frontier, min_y = [], float("inf")
    for p in log_pts:
        if p[1] < min_y:
            min_y = p[1]
            frontier.append(p)
    return 10 ** np.array(frontier) if frontier else np.empty((0, 2))


def plot_frontier(trial_summaries, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, trials in trial_summaries.items():
        all_pts  = [pt for t in trials for pt in t["points"]]
        frontier = pareto_lower_left(np.array(all_pts))
        if len(frontier) == 0:
            continue
        s = STYLE_MAP.get(label, {"color": "gray", "marker": "o", "label": label})
        ax.plot(frontier[:, 0], frontier[:, 1],
                color=s["color"], marker=s["marker"],
                markersize=4, linewidth=1.5, alpha=0.9, label=s["label"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Test Error – Old Task", fontsize=13)
    ax.set_ylabel("Test Error – New Task", fontsize=13)
    ax.set_title(title, fontsize=14, pad=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    tqdm.write(f"  Saved {save_path}")


def plot_model_sizes(winning_models, title, save_path):
    x   = np.arange(len(ACTIVATIONS))
    w   = 0.35
    sgd = [winning_models.get(f"{a}_SGD",     0) for a in ACTIVATIONS]
    do  = [winning_models.get(f"{a}_Dropout", 0) for a in ACTIVATIONS]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w/2, sgd, width=w, label="SGD",     color="steelblue", alpha=0.85)
    ax.bar(x + w/2, do,  width=w, label="Dropout", color="tomato",    alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ACTIVATIONS, fontsize=11)
    ax.set_ylabel("Model size (# parameters)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for xi, (s, d) in enumerate(zip(sgd, do)):
        ax.text(xi - w/2, s, f"{s:,}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, d, f"{d:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    tqdm.write(f"  Saved {save_path}")


def save_and_plot(ckpt, fig_num, frontier_title, size_title):
    path = os.path.join(RESULTS_DIR, f"scenario_{fig_num}_repro.pt")
    torch.save(ckpt, path)
    tqdm.write(f"Checkpoint → {path}")
    plot_frontier(ckpt["trial_summaries"], frontier_title,
                  os.path.join(RESULTS_DIR, f"fig{fig_num}_frontier_repro.png"))
    plot_model_sizes(ckpt["winning_models"], size_title,
                     os.path.join(RESULTS_DIR, f"fig{fig_num}_sizes_repro.png"))


# =============================================================================
# Section 8 – Scenario runners
# =============================================================================

def run_scenario_1():
    tqdm.write("\n" + "=" * 60)
    tqdm.write("SCENARIO 1 – Input Reformatting (Permuted MNIST)")
    tqdm.write("=" * 60)
    rng   = np.random.default_rng(SEED)
    perm1 = torch.from_numpy(rng.permutation(784))
    perm2 = torch.from_numpy(rng.permutation(784))
    while torch.equal(perm1, perm2):
        perm2 = torch.from_numpy(rng.permutation(784))
    t1_tr, t1_va, t1_te = get_permuted_mnist_loaders(perm1)
    t2_tr, t2_va, t2_te = get_permuted_mnist_loaders(perm2)
    ckpt = run_hyperparameter_search(
        "s1_input_reformatting",
        t1_tr, t1_va, t1_te, t2_tr, t2_va, t2_te,
        input_dim=784, output_dim=10)
    save_and_plot(ckpt, 1,
                  "Figure 1 – Input Reformatting: Old MNIST → New Permutation",
                  "Figure 2 – Optimal Model Size (Input Reformatting)")


def run_scenario_2_paper_pair():
    tqdm.write("\n" + "=" * 60)
    tqdm.write("SCENARIO 2 – Similar Tasks (Amazon Kitchen → DVD)")
    tqdm.write("=" * 60)
    base = os.path.join("data", "amazon")
    t1_tr, t1_va, t1_te, dim1, cls1 = get_amazon_from_npz(os.path.join(base, "kitchen.npz"))
    t2_tr, t2_va, t2_te, dim2, cls2 = get_amazon_from_npz(os.path.join(base, "dvd.npz"))
    assert dim1 == dim2 and cls1 == cls2
    ckpt = run_hyperparameter_search(
        "s2_similar_kitchen_dvd",
        t1_tr, t1_va, t1_te, t2_tr, t2_va, t2_te,
        input_dim=dim1, output_dim=cls1)
    save_and_plot(ckpt, 3,
                  "Figure 3 – Similar Tasks: Amazon Kitchen → DVD",
                  "Figure 4 – Optimal Model Size (Similar Tasks)")


def run_scenario_3():
    tqdm.write("\n" + "=" * 60)
    tqdm.write("SCENARIO 3 – Dissimilar Tasks (MNIST 2/9 → Amazon DVD)")
    tqdm.write("=" * 60)
    base = os.path.join("data", "amazon")
    # Use Amazon at full feature size (5000), pad MNIST to match — same as original paper
    a_tr, a_va, a_te, amazon_dim, amazon_cls = get_amazon_from_npz(
        os.path.join(base, "dvd.npz"))
    assert amazon_cls == 2
    m_tr, m_va, m_te = get_padded_binary_mnist_loaders(target_dim=amazon_dim, classes=(2, 9))
    ckpt = run_hyperparameter_search(
        "s3_dissimilar_mnist29_amazon_dvd",
        m_tr, m_va, m_te, a_tr, a_va, a_te,
        input_dim=amazon_dim, output_dim=2)
    save_and_plot(ckpt, 5,
                  "Figure 5 – Dissimilar Tasks: MNIST(2,9) → Amazon DVD",
                  "Figure 6 – Optimal Model Size (Dissimilar Tasks)")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    tqdm.write(f"Device            : {DEVICE}")
    tqdm.write(f"Trials/condition  : {TRIALS_PER_CONDITION}")
    tqdm.write(f"Patience Task 1/2 : {PATIENCE_OLD} / {PATIENCE_NEW} epochs")
    tqdm.write(f"Output dir        : {RESULTS_DIR}\n")
    run_scenario_1()
    run_scenario_2_paper_pair()
    run_scenario_3()
    tqdm.write("\nDone. All figures saved to " + RESULTS_DIR)
