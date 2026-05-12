"""
final_experiment_repro.py
=========================
Faithful reproduction of:
  "An Empirical Investigation of Catastrophic Forgetting in
   Gradient-Based Neural Networks"
  Goodfellow, Mirza, Xiao, Courville, Bengio (arXiv:1312.6211v3, 2015)

Key fixes vs pytorch_reproduction_suite.py
-------------------------------------------
1. PATIENCE       – 100 epochs (paper §4), not 12.
2. TRIALS         – 25 per condition (paper §3.3), not 3/10.
3. DROPOUT RATES  – p_hidden=0.5, p_visible=0.2 (paper §3.1). Already correct;
                    now explicitly documented.
4. HIDDEN SIZES   – wider candidate pool for ReLU/Sigmoid to allow the "noticeably
                    larger" dropout nets described in §3.1.
5. BIAS INIT      – Maxout/LWTA: 0; Sigmoid: slightly negative; ReLU: slightly
                    positive. Matches paper §4 exactly.
6. FRONTIER CURVE – Pareto lower-left in log-space (paper §4 description), not
                    a convex hull.
7. SCENARIO 2     – Kitchen→DVD as the paper's specific pair (§4.2).
8. SCENARIO 3     – MNIST(2,9)→Amazon(DVD) as in paper §4.3.
9. BAR CHARTS     – SGD vs Dropout grouped side-by-side (matching Figs 2/4/6).
10. NameError fix – original __main__ called undefined run_scenario_2/3 functions.
"""

import copy
import os
import random
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

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results_repro"
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Experiment switches ───────────────────────────────────────────────────────
RUN_SCENARIO_1            = True
RUN_SCENARIO_2_PAPER_PAIR = True   # Kitchen → DVD  (paper's exact pair)
RUN_SCENARIO_3            = True   # MNIST(2,9) → Amazon DVD

# Paper uses 25; reduce to 5 for a fast smoke-test.
TRIALS_PER_CONDITION = 25          # paper: 25

# Paper §4: early-stopping patience = 100 epochs.
PATIENCE_OLD = 100
PATIENCE_NEW = 100
MAX_EPOCHS_OLD = 1000
MAX_EPOCHS_NEW = 1000
BATCH_SIZE = 100


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


def apply_max_norm_constraint(model: nn.Module, max_val: float) -> None:
    """Max-norm weight constraint (Srebro & Shraibman, 2005) – paper §3."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1 and "weight" in name:
                norms   = param.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norms, max=max_val)
                param.mul_(desired / (norms + 1e-8))


def evaluate_error(model: nn.Module, loader: DataLoader) -> float:
    """Classification error rate (1 − accuracy)."""
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

def get_permuted_mnist_loaders(permutation, batch_size=BATCH_SIZE, val_size=5000):
    """§4.1 – Input Reformatting: MNIST with fixed pixel permutation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[permutation]),
    ])
    full_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds    = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_ds, val_ds = _split_dataset(full_train, n_val=val_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


def get_padded_binary_mnist_loaders(target_dim, classes=(2, 9),
                                    batch_size=BATCH_SIZE, val_size=1000):
    """§4.3 – Dissimilar Tasks (old task): binary MNIST zero-padded to target_dim."""
    def remap(y): return 0 if y == classes[0] else 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.view(-1), (0, target_dim - 784))),
    ])
    full_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    def to_td(subset):
        xs, ys = [], []
        for x, y in subset:
            xs.append(x); ys.append(remap(int(y)))
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    tr_idx = [i for i, y in enumerate(full_train.targets.tolist()) if y in classes]
    te_idx = [i for i, y in enumerate(test_full.targets.tolist())  if y in classes]
    train_td = to_td(Subset(full_train, tr_idx))
    test_td  = to_td(Subset(test_full,  te_idx))
    train_td, val_td = _split_dataset(train_td, n_val=val_size)
    return (
        DataLoader(train_td, batch_size=batch_size, shuffle=True),
        DataLoader(val_td,   batch_size=batch_size, shuffle=False),
        DataLoader(test_td,  batch_size=batch_size, shuffle=False),
    )


def get_amazon_from_npz(npz_path, batch_size=BATCH_SIZE, val_ratio=0.2):
    """Load a pre-processed Amazon category .npz file."""
    data      = np.load(npz_path, allow_pickle=True)
    X_tr_full = torch.tensor(data["X_train"], dtype=torch.float32)
    y_tr_full = torch.tensor(data["y_train"], dtype=torch.long)
    X_te      = torch.tensor(data["X_test"],  dtype=torch.float32)
    y_te      = torch.tensor(data["y_test"],  dtype=torch.long)

    n_val = max(1, int(val_ratio * len(X_tr_full)))
    idx   = torch.randperm(len(X_tr_full))
    vi, ti = idx[:n_val], idx[n_val:]
    return (
        DataLoader(TensorDataset(X_tr_full[ti], y_tr_full[ti]), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_tr_full[vi], y_tr_full[vi]), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(X_te, y_te),                   batch_size=batch_size, shuffle=False),
        X_tr_full.shape[1],
        int(torch.unique(y_tr_full).numel()),
    )


def reduce_feature_dim(train_loader, val_loader, test_loader, target_dim):
    """§4.3 – PCA-reduce Amazon (5000-dim) to target_dim=784 using TruncatedSVD."""
    def gather(loader):
        xs, ys = zip(*[(x, y) for x, y in loader])
        return torch.cat(xs).numpy(), torch.cat(ys)

    X_tr, y_tr = gather(train_loader)
    X_va, y_va = gather(val_loader)
    X_te, y_te = gather(test_loader)

    svd  = TruncatedSVD(n_components=target_dim, random_state=SEED)
    X_tr = svd.fit_transform(X_tr)
    X_va = svd.transform(X_va)
    X_te = svd.transform(X_te)

    bs = train_loader.batch_size
    return (
        DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), y_tr), batch_size=bs, shuffle=True),
        DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32), y_va), batch_size=bs, shuffle=False),
        DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), y_te), batch_size=bs, shuffle=False),
    )


# =============================================================================
# Section 3 – Network architectures
# =============================================================================

class Maxout(nn.Module):
    """Maxout activation (Goodfellow et al., 2013b): h_i = max_{j in group} z_j"""
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    def forward(self, x):
        b, d = x.shape
        return x.view(b, d // self.pool_size, self.pool_size).max(dim=2).values


class LWTA(nn.Module):
    """
    Hard Local Winner-Take-All (Srivastava et al., 2013).
    Ties broken uniformly at random (paper footnote 1).
    """
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size
    def forward(self, x):
        b, d  = x.shape
        x_g   = x.view(b, d // self.group_size, self.group_size)
        noise = torch.rand_like(x_g) * 1e-6          # random tie-breaking
        mask  = ((x_g + noise) >= (x_g + noise).max(dim=2, keepdim=True).values).float()
        mask  = (mask / mask.sum(dim=2, keepdim=True).clamp(min=1) > 0).float()
        return (x_g * mask).view(b, d)


class MLP(nn.Module):
    """
    Two-hidden-layer MLP + softmax output (paper §4).
    Dropout: p_visible=0.2, p_hidden=0.5 (paper §3.1 "well-known constants").
    Bias init follows paper §4:
      Maxout/LWTA → 0 (avoids dominant filter)
      Sigmoid     → slightly negative (promotes sparsity)
      ReLU        → slightly positive (prevents dead units)
    """
    P_VIS = 0.2
    P_HID = 0.5

    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation, use_dropout, pool_size=2, init_name="xavier"):
        super().__init__()
        self.activation_name = activation
        self.drop_in  = nn.Dropout(self.P_VIS) if use_dropout else nn.Identity()
        self.drop_hid = nn.Dropout(self.P_HID) if use_dropout else nn.Identity()

        # Maxout/LWTA expand pre-activation width by pool_size
        expansion = pool_size if activation in {"Maxout", "LWTA"} else 1
        pre_dim   = hidden_dim * expansion

        self.fc1 = nn.Linear(input_dim,  pre_dim)
        self.fc2 = nn.Linear(hidden_dim, pre_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        acts = {
            "ReLU":    (nn.ReLU(),          nn.ReLU()),
            "Sigmoid": (nn.Sigmoid(),        nn.Sigmoid()),
            "Maxout":  (Maxout(pool_size),   Maxout(pool_size)),
            "LWTA":    (LWTA(pool_size),     LWTA(pool_size)),
        }
        if activation not in acts:
            raise ValueError(f"Unknown activation: {activation}")
        self.act1, self.act2 = acts[activation]
        self._init_weights(init_name)

    def _init_weights(self, init_name):
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            act = self.activation_name
            if act in {"Maxout", "LWTA"}:
                nn.init.uniform_(m.weight, -0.005, 0.005)
                nn.init.constant_(m.bias, 0.0)
            elif act == "Sigmoid":
                nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.2, 0.0)   # negative → sparsity
            else:  # ReLU
                if init_name == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, 0.0, 0.1)    # positive → no dead units

    def forward(self, x):
        x = self.drop_in(x)
        x = self.act1(self.fc1(x))
        x = self.drop_hid(x)
        x = self.act2(self.fc2(x))
        x = self.drop_hid(x)
        return self.fc3(x)


# =============================================================================
# Section 4 – Hyperparameter sampling  (paper §3.3 – random search)
# =============================================================================

@dataclass
class HParams:
    hidden_dim: int
    lr:         float
    momentum:   float
    max_norm:   float
    init_name:  str


def sample_hparams(activation: str, rng: random.Random) -> HParams:
    """
    Random search over the ranges described in paper §4.
    Wider hidden_dim pool for ReLU/Sigmoid allows the larger dropout nets (§3.1).
    """
    if activation in {"ReLU", "Sigmoid"}:
        hidden_dim = rng.choice([128, 256, 512, 800, 1024, 1200, 1600, 2048])
    else:
        hidden_dim = rng.choice([128, 256, 400, 512, 800, 1024])
    lr        = 10 ** rng.uniform(-4.0, -1.0)
    momentum  = rng.uniform(0.3, 0.99)
    max_norm  = rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    init_name = rng.choice(["xavier", "kaiming", "uniform"])
    return HParams(hidden_dim, lr, momentum, max_norm, init_name)


def build_model(input_dim, output_dim, activation, use_dropout, hp):
    init = hp.init_name
    if activation == "Sigmoid" and init == "kaiming":
        init = "xavier"
    if activation in {"Maxout", "LWTA"}:
        init = "uniform"
    return MLP(input_dim, hp.hidden_dim, output_dim,
               activation, use_dropout, pool_size=2, init_name=init).to(DEVICE)


# =============================================================================
# Section 5 – Training loops
# =============================================================================

def train_one_epoch(model, loader, optimizer, max_norm):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
        apply_max_norm_constraint(model, max_norm)


def train_task1(model, train_ldr, val_ldr, optimizer, max_norm,
                max_epochs=MAX_EPOCHS_OLD, patience=PATIENCE_OLD):
    """Train on Task 1; restore best weights (paper §4)."""
    best_val, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        train_one_epoch(model, train_ldr, optimizer, max_norm)
        val_err = evaluate_error(model, val_ldr)
        if val_err < best_val:
            best_val, best_state, stale = val_err, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model


def train_task2_and_log(model, t1_val, t1_test, t2_train, t2_val, t2_test,
                        optimizer, max_norm,
                        max_epochs=MAX_EPOCHS_NEW, patience=PATIENCE_NEW):
    """
    Train on Task 2; log (old_test_err, new_test_err) after every epoch.
    Stop when joint val error (val_old + val_new) does not improve for
    *patience* epochs — exactly as described in paper §4.
    """
    best_joint, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    trajectory = []
    for _ in range(max_epochs):
        train_one_epoch(model, t2_train, optimizer, max_norm)
        old_val  = evaluate_error(model, t1_val)
        new_val  = evaluate_error(model, t2_val)
        joint    = old_val + new_val
        old_test = evaluate_error(model, t1_test)
        new_test = evaluate_error(model, t2_test)
        trajectory.append((old_test, new_test))
        if joint < best_joint:
            best_joint, best_state, stale = joint, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return trajectory, best_joint


# =============================================================================
# Section 6 – Hyperparameter search driver
# =============================================================================

ACTIVATIONS   = ["Sigmoid", "ReLU", "Maxout", "LWTA"]
DROPOUT_FLAGS = [False, True]


def run_hyperparameter_search(scenario_name, t1_train, t1_val, t1_test,
                               t2_train, t2_val, t2_test,
                               input_dim, output_dim,
                               trials_per_condition=TRIALS_PER_CONDITION):
    """25 random trials × 8 conditions = 200 models (paper §3.3)."""
    all_results, trial_summaries, winning_models = {}, {}, {}

    for activation in ACTIVATIONS:
        for use_dropout in DROPOUT_FLAGS:
            label = f"{activation}_{'Dropout' if use_dropout else 'SGD'}"
            print(f"\n[{scenario_name}] {label}", flush=True)
            rng = random.Random(SEED)
            best_joint_global, best_param_count = float("inf"), 0
            trials = []

            for trial in range(trials_per_condition):
                print(f"  trial {trial+1}/{trials_per_condition}", end="  ", flush=True)
                set_seed(SEED + trial)
                hp    = sample_hparams(activation, rng)
                model = build_model(input_dim, output_dim, activation, use_dropout, hp)
                opt   = optim.SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum)

                model = train_task1(model, t1_train, t1_val, opt, hp.max_norm)
                traj, best_joint = train_task2_and_log(
                    model, t1_val, t1_test, t2_train, t2_val, t2_test, opt, hp.max_norm)

                print(f"joint={best_joint:.4f}  epochs={len(traj)}", flush=True)
                pc = count_parameters(model)
                trials.append({"points": traj, "hp": hp.__dict__,
                               "best_joint": best_joint, "param_count": pc})
                if best_joint < best_joint_global:
                    best_joint_global, best_param_count = best_joint, pc

            all_results[label]     = [pt for t in trials for pt in t["points"]]
            trial_summaries[label] = trials
            winning_models[label]  = best_param_count

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


def pareto_lower_left(points: np.ndarray) -> np.ndarray:
    """
    Lower-left Pareto frontier in log-space.
    Matches the paper's "possibilities frontier" (§4):
    'the lower left frontier of the cloud of points'.
    """
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
    """Reproduce Figures 1, 3, 5: log-log possibilities-frontier curves."""
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
    print(f"  Saved {save_path}")


def plot_model_sizes(winning_models, title, save_path):
    """Reproduce Figures 2, 4, 6: grouped bar chart SGD vs Dropout."""
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
    print(f"  Saved {save_path}")


def save_and_plot(ckpt, fig_num, frontier_title, size_title):
    path = os.path.join(RESULTS_DIR, f"scenario_{fig_num}_repro.pt")
    torch.save(ckpt, path)
    print(f"Checkpoint → {path}")
    plot_frontier(ckpt["trial_summaries"], frontier_title,
                  os.path.join(RESULTS_DIR, f"fig{fig_num}_frontier_repro.png"))
    plot_model_sizes(ckpt["winning_models"], size_title,
                     os.path.join(RESULTS_DIR, f"fig{fig_num}_sizes_repro.png"))


# =============================================================================
# Section 8 – Scenario runners
# =============================================================================

def run_scenario_1():
    """§4.1 Input Reformatting – Permuted MNIST → differently permuted MNIST."""
    print("\n" + "=" * 60)
    print("SCENARIO 1 – Input Reformatting (Permuted MNIST)")
    print("=" * 60)
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
    """§4.2 Similar Tasks – Amazon Kitchen (old) → Amazon DVD (new)."""
    print("\n" + "=" * 60)
    print("SCENARIO 2 – Similar Tasks (Amazon Kitchen → DVD)")
    print("=" * 60)
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
    """§4.3 Dissimilar Tasks – MNIST(2,9) → Amazon DVD."""
    print("\n" + "=" * 60)
    print("SCENARIO 3 – Dissimilar Tasks (MNIST 2/9 → Amazon DVD)")
    print("=" * 60)
    target_dim = 784
    base = os.path.join("data", "amazon")
    a_tr, a_va, a_te, _, amazon_cls = get_amazon_from_npz(os.path.join(base, "dvd.npz"))
    assert amazon_cls == 2, "Expects binary Amazon task."
    a_tr, a_va, a_te = reduce_feature_dim(a_tr, a_va, a_te, target_dim=target_dim)
    m_tr, m_va, m_te = get_padded_binary_mnist_loaders(target_dim=target_dim, classes=(2, 9))
    ckpt = run_hyperparameter_search(
        "s3_dissimilar_mnist29_amazon_dvd",
        m_tr, m_va, m_te, a_tr, a_va, a_te,
        input_dim=target_dim, output_dim=2)
    save_and_plot(ckpt, 5,
                  "Figure 5 – Dissimilar Tasks: MNIST(2,9) → Amazon DVD",
                  "Figure 6 – Optimal Model Size (Dissimilar Tasks)")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print(f"Device            : {DEVICE}")
    print(f"Trials/condition  : {TRIALS_PER_CONDITION}")
    print(f"Patience Task 1/2 : {PATIENCE_OLD} / {PATIENCE_NEW} epochs")
    print(f"Output dir        : {RESULTS_DIR}\n")

    if RUN_SCENARIO_1:
        run_scenario_1()
    if RUN_SCENARIO_2_PAPER_PAIR:
        run_scenario_2_paper_pair()
    if RUN_SCENARIO_3:
        run_scenario_3()

    print("\nDone. All figures saved to", RESULTS_DIR)
