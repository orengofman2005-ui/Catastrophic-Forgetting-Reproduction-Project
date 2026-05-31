"""
ablation_study.py
=================
Ablation experiments on Scenario 1 (Permuted MNIST) to isolate the contribution
of individual components to catastrophic forgetting resistance.

Three ablations are run:
  1. Dropout rate: 0.0, 0.2, 0.5 (paper value)
  2. Weight decay: none, 1e-4, 1e-3
  3. BatchNorm: without vs with (bonus improvement)

For each configuration we run N_TRIALS trials with random HP search,
record best_joint and forgetting rate, and report mean ± std.

Forgetting rate is defined as:
  forgetting = old_task_error_after_task2 - old_task_error_before_task2
  (higher = more forgetting)

Usage:
  python ablation_study.py

Output:
  results_repro/ablation_results.pt   — raw results
  results_repro/ablation_dropout.png  — dropout ablation figure
  results_repro/ablation_wd.png       — weight decay ablation figure
  results_repro/ablation_batchnorm.png — batchnorm ablation figure
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from final_experiment_repro import (
    SEED, DEVICE, BATCH_SIZE,
    get_permuted_mnist_loaders,
    evaluate_error,
    set_seed,
    apply_max_norm_constraint,
    _get_lr_momentum,
    HParams,
    sample_hparams,
)

RESULTS_DIR = "results_repro"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRIALS = 4          # trials per ablation config
MAX_EPOCHS = 150
PATIENCE = 15


# =============================================================================
# Ablation MLP — supports optional BatchNorm and configurable Dropout
# =============================================================================

class AblationMLP(nn.Module):
    """
    Two-hidden-layer MLP with ReLU activation.
    Supports configurable dropout rates and optional BatchNorm.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 p_input=0.2, p_hidden=0.5, use_batchnorm=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.drop_in  = nn.Dropout(p_input)  if p_input  > 0 else nn.Identity()
        self.drop_hid = nn.Dropout(p_hidden) if p_hidden > 0 else nn.Identity()

        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn1 = self.bn2 = nn.Identity()

        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(layer.weight, -0.005, 0.005)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.drop_in(x)
        x = self.drop_hid(self.act(self.bn1(self.fc1(x))))
        x = self.drop_hid(self.act(self.bn2(self.fc2(x))))
        return self.fc3(x)


# =============================================================================
# Training helpers
# =============================================================================

def _make_hp(hidden_dim=512, lr=0.01, weight_decay=0.0):
    """Return a minimal HParams-like namespace for the ablation trainer."""
    from types import SimpleNamespace
    return SimpleNamespace(
        hidden_dim     = hidden_dim,
        lr             = lr,
        init_momentum  = 0.5,
        final_momentum = 0.9,
        momentum_sat   = 20,
        lr_sat         = 60,
        lr_decay       = 0.01,
        col_norm_h0    = 3.0,
        col_norm_h1    = 3.0,
        col_norm_out   = 3.0,
        weight_decay   = weight_decay,
    )


def train_one_epoch_abl(model, loader, optimizer, hp, epoch):
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
    apply_max_norm_constraint(model, hp)


def run_sequential(model, t1_train, t1_val, t1_test, t2_train, t2_val, t2_test, hp):
    """Train on Task1, record baseline, then train on Task2, record forgetting."""
    optimizer = optim.SGD(model.parameters(), lr=hp.lr,
                          momentum=hp.init_momentum,
                          weight_decay=getattr(hp, 'weight_decay', 0.0))

    # --- Task 1 training ---
    best_val, best_state, stale = float("inf"), {k: v.clone() for k, v in model.state_dict().items()}, 0
    for epoch in range(MAX_EPOCHS):
        train_one_epoch_abl(model, t1_train, optimizer, hp, epoch)
        val_err = evaluate_error(model, t1_val)
        if val_err < best_val:
            best_val, best_state, stale = val_err, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    baseline_old = evaluate_error(model, t1_test)

    # --- Task 2 training ---
    best_joint, stale = float("inf"), 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(MAX_EPOCHS):
        train_one_epoch_abl(model, t2_train, optimizer, hp, epoch)
        old_val = evaluate_error(model, t1_val)
        new_val = evaluate_error(model, t2_val)
        joint = old_val + new_val
        if joint < best_joint:
            best_joint, best_state, stale = joint, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)

    final_old  = evaluate_error(model, t1_test)
    final_new  = evaluate_error(model, t2_test)
    forgetting = final_old - baseline_old   # positive = forgot Task A

    return final_old + final_new, forgetting


# =============================================================================
# Ablation 1: Dropout Rate
# =============================================================================

def ablation_dropout():
    """Compare dropout_hidden in {0.0, 0.2, 0.5} on Scenario 1 (Permuted MNIST)."""
    dropout_rates = [0.0, 0.2, 0.5]
    rng = np.random.default_rng(SEED)
    perm1 = torch.from_numpy(rng.permutation(784))
    perm2 = torch.from_numpy(rng.permutation(784))

    t1_tr, t1_va, t1_te = get_permuted_mnist_loaders(perm1)
    t2_tr, t2_va, t2_te = get_permuted_mnist_loaders(perm2)

    results = {}
    for p_hid in dropout_rates:
        label = f"dropout={p_hid}"
        joints, forgettings = [], []

        for trial in tqdm(range(N_TRIALS), desc=label, leave=False):
            set_seed(SEED + trial)
            hp = _make_hp(hidden_dim=random.randint(256, 1024),
                          lr=10 ** random.uniform(-2.5, -1.0))
            p_in = 0.2 if p_hid > 0 else 0.0
            model = AblationMLP(784, hp.hidden_dim, 10,
                                p_input=p_in, p_hidden=p_hid).to(DEVICE)
            bj, forg = run_sequential(model, t1_tr, t1_va, t1_te,
                                      t2_tr, t2_va, t2_te, hp)
            joints.append(bj)
            forgettings.append(forg)

        results[label] = {
            "best_joint_mean": float(np.mean(joints)),
            "best_joint_std":  float(np.std(joints)),
            "forgetting_mean": float(np.mean(forgettings)),
            "forgetting_std":  float(np.std(forgettings)),
        }
        print(f"{label}: best_joint={np.mean(joints):.3f}±{np.std(joints):.3f}  "
              f"forgetting={np.mean(forgettings):.3f}±{np.std(forgettings):.3f}")

    return results, dropout_rates


# =============================================================================
# Ablation 2: Weight Decay
# =============================================================================

def ablation_weight_decay():
    """Compare weight_decay in {0, 1e-4, 1e-3} on Scenario 1."""
    wd_values = [0.0, 1e-4, 1e-3]
    rng = np.random.default_rng(SEED + 100)
    perm1 = torch.from_numpy(rng.permutation(784))
    perm2 = torch.from_numpy(rng.permutation(784))

    t1_tr, t1_va, t1_te = get_permuted_mnist_loaders(perm1)
    t2_tr, t2_va, t2_te = get_permuted_mnist_loaders(perm2)

    results = {}
    for wd in wd_values:
        label = f"wd={wd}"
        joints, forgettings = [], []

        for trial in tqdm(range(N_TRIALS), desc=label, leave=False):
            set_seed(SEED + trial + 200)
            hp = _make_hp(hidden_dim=random.randint(256, 1024),
                          lr=10 ** random.uniform(-2.5, -1.0),
                          weight_decay=wd)
            model = AblationMLP(784, hp.hidden_dim, 10,
                                p_input=0.2, p_hidden=0.5).to(DEVICE)
            bj, forg = run_sequential(model, t1_tr, t1_va, t1_te,
                                      t2_tr, t2_va, t2_te, hp)
            joints.append(bj)
            forgettings.append(forg)

        results[label] = {
            "best_joint_mean": float(np.mean(joints)),
            "best_joint_std":  float(np.std(joints)),
            "forgetting_mean": float(np.mean(forgettings)),
            "forgetting_std":  float(np.std(forgettings)),
        }
        print(f"{label}: best_joint={np.mean(joints):.3f}±{np.std(joints):.3f}  "
              f"forgetting={np.mean(forgettings):.3f}±{np.std(forgettings):.3f}")

    return results, wd_values


# =============================================================================
# Ablation 3: BatchNorm (Bonus)
# =============================================================================

def ablation_batchnorm():
    """Compare without BatchNorm vs with BatchNorm on Scenario 1."""
    rng = np.random.default_rng(SEED + 200)
    perm1 = torch.from_numpy(rng.permutation(784))
    perm2 = torch.from_numpy(rng.permutation(784))

    t1_tr, t1_va, t1_te = get_permuted_mnist_loaders(perm1)
    t2_tr, t2_va, t2_te = get_permuted_mnist_loaders(perm2)

    results = {}
    for use_bn in [False, True]:
        label = "BatchNorm=On" if use_bn else "BatchNorm=Off"
        joints, forgettings = [], []

        for trial in tqdm(range(N_TRIALS), desc=label, leave=False):
            set_seed(SEED + trial + 400)
            hp = _make_hp(hidden_dim=random.randint(256, 1024),
                          lr=10 ** random.uniform(-2.5, -1.0))
            model = AblationMLP(784, hp.hidden_dim, 10,
                                p_input=0.2, p_hidden=0.5,
                                use_batchnorm=use_bn).to(DEVICE)
            bj, forg = run_sequential(model, t1_tr, t1_va, t1_te,
                                      t2_tr, t2_va, t2_te, hp)
            joints.append(bj)
            forgettings.append(forg)

        results[label] = {
            "best_joint_mean": float(np.mean(joints)),
            "best_joint_std":  float(np.std(joints)),
            "forgetting_mean": float(np.mean(forgettings)),
            "forgetting_std":  float(np.std(forgettings)),
        }
        print(f"{label}: best_joint={np.mean(joints):.3f}±{np.std(joints):.3f}  "
              f"forgetting={np.mean(forgettings):.3f}±{np.std(forgettings):.3f}")

    return results


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_ablation_bars(labels, means, stds, ylabel, title, save_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=color, alpha=0.82,
                  error_kw={"elinewidth": 1.8, "ecolor": "black"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ablation 1: Dropout Rate")
    print("=" * 60)
    do_results, do_rates = ablation_dropout()

    print("\n" + "=" * 60)
    print("Ablation 2: Weight Decay")
    print("=" * 60)
    wd_results, wd_vals = ablation_weight_decay()

    print("\n" + "=" * 60)
    print("Ablation 3: BatchNorm (Bonus)")
    print("=" * 60)
    bn_results = ablation_batchnorm()

    # Save all raw results
    torch.save({
        "dropout": do_results,
        "weight_decay": wd_results,
        "batchnorm": bn_results,
    }, os.path.join(RESULTS_DIR, "ablation_results.pt"))

    # --- Plot dropout ablation ---
    do_labels = [f"p={p}" for p in do_rates]
    do_forg_means = [do_results[f"dropout={p}"]["forgetting_mean"] for p in do_rates]
    do_forg_stds  = [do_results[f"dropout={p}"]["forgetting_std"]  for p in do_rates]
    _plot_ablation_bars(
        do_labels, do_forg_means, do_forg_stds,
        ylabel="Forgetting Rate (old-task error increase)",
        title="Ablation 1: Effect of Dropout Rate on Forgetting",
        save_path=os.path.join(RESULTS_DIR, "ablation_dropout.png"),
        color="steelblue",
    )

    # --- Plot weight decay ablation ---
    wd_labels = [f"wd={v}" for v in wd_vals]
    wd_forg_means = [wd_results[f"wd={v}"]["forgetting_mean"] for v in wd_vals]
    wd_forg_stds  = [wd_results[f"wd={v}"]["forgetting_std"]  for v in wd_vals]
    _plot_ablation_bars(
        wd_labels, wd_forg_means, wd_forg_stds,
        ylabel="Forgetting Rate (old-task error increase)",
        title="Ablation 2: Effect of Weight Decay on Forgetting",
        save_path=os.path.join(RESULTS_DIR, "ablation_wd.png"),
        color="tomato",
    )

    # --- Plot batchnorm ablation ---
    bn_labels = list(bn_results.keys())
    bn_forg_means = [bn_results[l]["forgetting_mean"] for l in bn_labels]
    bn_forg_stds  = [bn_results[l]["forgetting_std"]  for l in bn_labels]
    _plot_ablation_bars(
        bn_labels, bn_forg_means, bn_forg_stds,
        ylabel="Forgetting Rate (old-task error increase)",
        title="Ablation 3 (Bonus): Effect of BatchNorm on Forgetting",
        save_path=os.path.join(RESULTS_DIR, "ablation_batchnorm.png"),
        color="mediumseagreen",
    )

    print("\nAll ablation experiments complete.")
    print(f"Results saved to {RESULTS_DIR}/ablation_results.pt")
