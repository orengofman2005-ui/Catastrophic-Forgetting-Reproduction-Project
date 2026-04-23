import copy
import os
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

# We set a fixed "seed" number. 
# This guarantees that every time we run our "random" experiments, 
# we get the exact same results. It makes our science predictable!
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results_fixed"
AMAZON_CATEGORIES = ["books", "dvd", "electronics", "kitchen"]
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# This helper function locks all the random number generators to our chosen seed.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# This function counts how many adjustable parts (parameters) our AI model has.
# You can think of these as the "knobs and dials" the computer turns to learn.
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Sometimes the AI's "knobs and dials" (parameters) get turned up way too high during learning.
# This function acts like a safety limit, turning them back down if they exceed a certain maximum.
def apply_max_norm_constraint(model: nn.Module, max_val: float) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1 and "weight" in name:
                norms = param.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norms, max=max_val)
                param.mul_(desired / (norms + 1e-8))


# This is like giving the AI a final exam. It checks how many mistakes
# the model makes on a batch of test data. A lower score means it did better.
def evaluate_error(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 1.0 - (correct / max(total, 1))


# This function hides a small portion of our study materials (the "validation" set).
# The AI uses the rest to study, and uses the hidden part to test itself 
# to make sure it's actually learning, not just memorizing the answers.
def _split_dataset(dataset, n_val: int):
    n = len(dataset)
    indices = torch.randperm(n).tolist()
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# MNIST is a famous set of handwritten digits (0-9) used to train AI.
# This function loads these images but scrambles the pixels in a specific, repeatable way.
# It's like asking the AI to read a secret code where the letters have been shuffled.
def get_permuted_mnist_loaders(
    permutation: torch.Tensor,
    batch_size: int = 100,
    val_size: int = 5000,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[permutation]),
    ])
    full_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_ds, val_ds = _split_dataset(full_train, n_val=val_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# This function also loads handwritten digits, but only two specific numbers (like 2 and 9).
# It also adds blank space ("padding") around the images so they match the size of completely different tasks later.
def get_padded_binary_mnist_loaders(
    target_dim: int,
    classes=(2, 9),
    batch_size: int = 100,
    val_size: int = 1000,
):
    assert target_dim >= 784

    def remap_label(y: int) -> int:
        return 0 if y == classes[0] else 1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.view(-1), (0, target_dim - 784))),
    ])
    full_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_full = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_idx = [i for i, y in enumerate(full_train.targets.tolist()) if y in classes]
    test_idx = [i for i, y in enumerate(test_full.targets.tolist()) if y in classes]

    train_subset = Subset(full_train, train_idx)
    test_subset = Subset(test_full, test_idx)

    def subset_to_tensor_dataset(subset: Subset) -> TensorDataset:
        xs, ys = [], []
        for x, y in subset:
            xs.append(x)
            ys.append(remap_label(int(y)))
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    train_tensor = subset_to_tensor_dataset(train_subset)
    test_tensor = subset_to_tensor_dataset(test_subset)
    train_tensor, val_tensor = _split_dataset(train_tensor, n_val=val_size)

    return (
        DataLoader(train_tensor, batch_size=batch_size, shuffle=True),
        DataLoader(val_tensor, batch_size=batch_size, shuffle=False),
        DataLoader(test_tensor, batch_size=batch_size, shuffle=False),
    )


# This function loads the Amazon review data we prepared earlier.
# It splits the reviews into "study" (train), "practice exam" (validation), and "final exam" (test) groups.
def get_amazon_from_npz(npz_path: str, batch_size: int = 100, val_ratio: float = 0.2):
    data = np.load(npz_path, allow_pickle=True)
    required = ["X_train", "y_train", "X_test", "y_test"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{npz_path} is missing keys: {missing}")

    X_train_full = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_full = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    n_train = len(X_train_full)
    n_val = max(1, int(val_ratio * n_train))
    indices = torch.randperm(n_train)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    input_dim = X_train.shape[1]
    output_dim = int(torch.unique(y_train_full).numel())
    return (
        DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
        input_dim,
        output_dim,
    )


# Sometimes our data has too many details (like thousands of words). 
# This function magically "squishes" the data down to a smaller size, 
# keeping only the most important patterns so it's easier for the AI to handle.
def reduce_feature_dim(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_dim: int,
):
    def gather(loader):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        return torch.cat(xs, dim=0).numpy(), torch.cat(ys, dim=0)

    X_train, y_train = gather(train_loader)
    X_val, y_val = gather(val_loader)
    X_test, y_test = gather(test_loader)

    svd = TruncatedSVD(n_components=target_dim, random_state=SEED)
    X_train_r = svd.fit_transform(X_train)
    X_val_r = svd.transform(X_val)
    X_test_r = svd.transform(X_test)

    return (
        DataLoader(
            TensorDataset(torch.tensor(X_train_r, dtype=torch.float32), y_train),
            batch_size=train_loader.batch_size,
            shuffle=True,
        ),
        DataLoader(
            TensorDataset(torch.tensor(X_val_r, dtype=torch.float32), y_val),
            batch_size=val_loader.batch_size,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(torch.tensor(X_test_r, dtype=torch.float32), y_test),
            batch_size=test_loader.batch_size,
            shuffle=False,
        ),
    )


# A special type of "brain cell" (activation function) for our AI. 
# It looks at a group of signals and only passes forward the strongest one.
class Maxout(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d = x.shape
        x = x.view(b, d // self.pool_size, self.pool_size)
        x, _ = x.max(dim=2)
        return x


# Another special type of "brain cell" called "Local Winner-Take-All".
# In a group of cells, only the one with the highest score gets to fire; the rest are forced to be silent.
class LWTA(nn.Module):
    def __init__(self, group_size: int):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d = x.shape
        x = x.view(b, d // self.group_size, self.group_size)
        max_val, _ = x.max(dim=2, keepdim=True)
        mask = (x == max_val).float()
        x = x * mask
        return x.view(b, d)


# This is the blueprint for our Artificial Intelligence model.
# MLP stands for Multi-Layer Perceptron. Think of it as a simple digital brain
# where information passes through different layers to make a decision.
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
        use_dropout: bool,
        k: int = 2,
        init_name: str = "xavier",
    ):
        super().__init__()
        self.activation_name = activation
        self.drop_in = nn.Dropout(0.2) if use_dropout else nn.Identity()
        self.drop_hid = nn.Dropout(0.5) if use_dropout else nn.Identity()

        expansion = k if activation in {"Maxout", "LWTA"} else 1
        first_out = hidden_dim * expansion
        second_in = hidden_dim if activation == "Maxout" else hidden_dim * expansion

        self.fc1 = nn.Linear(input_dim, first_out)
        self.fc2 = nn.Linear(second_in, hidden_dim * expansion)
        self.fc3 = nn.Linear(hidden_dim if activation == "Maxout" else hidden_dim * expansion, output_dim)

        if activation == "ReLU":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation == "Sigmoid":
            self.act1 = nn.Sigmoid()
            self.act2 = nn.Sigmoid()
        elif activation == "Maxout":
            self.act1 = Maxout(k)
            self.act2 = Maxout(k)
        elif activation == "LWTA":
            self.act1 = LWTA(k)
            self.act2 = LWTA(k)
        else:
            raise ValueError(f"Unknown activation {activation}")

        self._initialize(init_name)

    # This function sets the initial, random starting positions of all the knobs and dials.
    # A good starting position helps the AI learn much faster.
    def _initialize(self, init_name: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in {"Maxout", "LWTA"}:
                    nn.init.uniform_(m.weight, -0.005, 0.005)
                    nn.init.constant_(m.bias, 0.0)
                elif self.activation_name == "Sigmoid":
                    if init_name == "xavier":
                        nn.init.xavier_uniform_(m.weight)
                    else:
                        nn.init.uniform_(m.weight, -0.05, 0.05)
                    nn.init.constant_(m.bias, -0.1)
                else:
                    if init_name == "kaiming":
                        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_in(x)
        x = self.act1(self.fc1(x))
        x = self.drop_hid(x)
        x = self.act2(self.fc2(x))
        x = self.drop_hid(x)
        return self.fc3(x)


@dataclass
class HParams:
    hidden_dim: int
    lr: float
    momentum: float
    max_norm: float
    init_name: str


# This is a simple digital notebook. 
# It records the AI's test scores after every round of studying so we can graph them later.
class EpochLogger:
    def __init__(self):
        self.history = []

    def log(self, epoch: int, old_test_err: float, new_test_err: float, old_val_err: float, new_val_err: float):
        self.history.append({
            "epoch": epoch,
            "old_test_err": old_test_err,
            "new_test_err": new_test_err,
            "old_val_err": old_val_err,
            "new_val_err": new_val_err,
        })


# This function represents one full round of studying (an "epoch").
# The AI looks at the examples, makes guesses, sees its mistakes, 
# and updates its "knobs and dials" to do better next time.
def train_one_epoch(model, loader, optimizer, criterion, max_norm):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        apply_max_norm_constraint(model, max_norm)


# This function lets the AI study over and over until it stops improving.
# If its test scores stop getting better, it stops studying 
# so it doesn't overthink and just memorize the answers.
def train_until_early_stop(
    model,
    train_loader,
    val_loader,
    optimizer,
    max_epochs: int = 120,
    patience: int = 12,
    max_norm: float = 2.0,
):
    criterion = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for _ in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, max_norm)
        val_err = evaluate_error(model, val_loader)
        if val_err < best_val:
            best_val = val_err
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return model


# This is the core of our experiment! It forces the AI to study one subject first,
# and then forces it to study a completely new subject. 
# We track its scores the whole time to see if it "forgets" the first subject.
def sequential_train_and_log(
    model,
    t1_train,
    t1_val,
    t1_test,
    t2_train,
    t2_val,
    t2_test,
    optimizer,
    max_epochs_old,
    max_epochs_new,
    patience_old,
    patience_new,
    max_norm,
):
    model = train_until_early_stop(model, t1_train, t1_val, optimizer, max_epochs_old, patience_old, max_norm)

    criterion = nn.CrossEntropyLoss()
    best_joint_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    logger = EpochLogger()

    for epoch in range(max_epochs_new):
        print(f"task2 epoch {epoch + 1}/{max_epochs_new}", flush=True)
        train_one_epoch(model, t2_train, optimizer, criterion, max_norm)

        old_val = evaluate_error(model, t1_val)
        new_val = evaluate_error(model, t2_val)
        joint_val = old_val + new_val

        old_test = evaluate_error(model, t1_test)
        new_test = evaluate_error(model, t2_test)
        logger.log(epoch, old_test, new_test, old_val, new_val)

        if joint_val < best_joint_val:
            best_joint_val = joint_val
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience_new:
            break

    model.load_state_dict(best_state)
    return logger, best_joint_val


# Finding the perfect settings for an AI is tricky.
# This function acts like a scientist, trying out many different random combinations
# (like how fast to learn) to see which settings create the smartest model.
def sample_hparams(activation: str) -> HParams:
    if activation in {"ReLU", "Sigmoid"}:
        hidden_dim = random.choice([128, 256, 512, 800, 1000, 1200, 1600])
    else:
        hidden_dim = random.choice([128, 256, 400, 512, 800])

    lr = 10 ** random.uniform(-4.0, -1.0)
    momentum = random.uniform(0.3, 0.99)
    max_norm = random.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    init_name = random.choice(["xavier", "kaiming", "uniform"])
    return HParams(hidden_dim, lr, momentum, max_norm, init_name)


# This function takes our blueprint (the MLP class) and actually builds the AI model
# using the specific settings (like size and activation type) we give it.
def build_model(input_dim, output_dim, activation, use_dropout, hp: HParams):
    init_name = hp.init_name
    if activation == "Sigmoid" and init_name == "kaiming":
        init_name = "xavier"
    if activation in {"Maxout", "LWTA"}:
        init_name = "uniform"
    return MLP(input_dim, hp.hidden_dim, output_dim, activation, use_dropout, k=2, init_name=init_name).to(DEVICE)


# This is the master control center for our experiments.
# It tries out many different settings (like learning speeds and model sizes),
# trains an AI for each one, and keeps a record of which settings produced the smartest models.
def run_hyperparameter_search(
    scenario_name: str,
    t1_train,
    t1_val,
    t1_test,
    t2_train,
    t2_val,
    t2_test,
    input_dim: int,
    output_dim: int,
    trials_per_condition: int = 25,
    max_epochs_old: int = 120,
    max_epochs_new: int = 120,
    patience_old: int = 12,
    patience_new: int = 12,
):
    activations = ["ReLU", "Sigmoid", "Maxout", "LWTA"]
    dropout_flags = [False, True]

    all_results = {}
    trial_summaries = {}
    winning_models = {}

    for activation in activations:
        for use_dropout in dropout_flags:
            cond = f"{activation}_{'Dropout' if use_dropout else 'SGD'}"
            print(f"[{scenario_name}] {cond}")
            all_results[cond] = []
            trial_summaries[cond] = []
            best_joint_val_for_condition = float("inf")
            best_param_count = 0

            for trial in range(trials_per_condition):
                print(f"{cond} | trial {trial + 1}/{trials_per_condition}", flush=True)
                set_seed(SEED + trial)
                hp = sample_hparams(activation)
                model = build_model(input_dim, output_dim, activation, use_dropout, hp)
                optimizer = optim.SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum)

                logger, best_joint_val = sequential_train_and_log(
                    model,
                    t1_train, t1_val, t1_test,
                    t2_train, t2_val, t2_test,
                    optimizer,
                    max_epochs_old=max_epochs_old,
                    max_epochs_new=max_epochs_new,
                    patience_old=patience_old,
                    patience_new=patience_new,
                    max_norm=hp.max_norm,
                )

                points = [(h["old_test_err"], h["new_test_err"]) for h in logger.history]
                all_results[cond].extend(points)

                param_count = count_parameters(model)
                trial_summaries[cond].append({
                    "params": hp.__dict__,
                    "points": points,
                    "best_joint_val": best_joint_val,
                    "best_model_param_count": param_count,
                })

                if best_joint_val < best_joint_val_for_condition:
                    best_joint_val_for_condition = best_joint_val
                    best_param_count = param_count

            winning_models[cond] = best_param_count

    return {
        "scenario_name": scenario_name,
        "results": all_results,
        "trial_summaries": trial_summaries,
        "winning_models": winning_models,
        "meta": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "trials_per_condition": trials_per_condition,
            "max_epochs_old": max_epochs_old,
            "max_epochs_new": max_epochs_new,
            "patience_old": patience_old,
            "patience_new": patience_new,
        },
    }


# When we run many small experiments, we end up with lots of separate result files.
# This function gathers all those scattered results and combines them into one big summary.
def _merge_checkpoints(checkpoints: List[dict], scenario_name: str) -> dict:
    merged_results = {}
    merged_trial_summaries = {}
    merged_winning_models = {}

    for ckpt in checkpoints:
        for cond, pts in ckpt["results"].items():
            merged_results.setdefault(cond, []).extend(pts)

        for cond, summaries in ckpt["trial_summaries"].items():
            merged_trial_summaries.setdefault(cond, []).extend(summaries)

        for cond, param_count in ckpt["winning_models"].items():
            current = merged_winning_models.get(cond, 0)
            merged_winning_models[cond] = max(current, param_count)

    return {
        "scenario_name": scenario_name,
        "results": merged_results,
        "trial_summaries": merged_trial_summaries,
        "winning_models": merged_winning_models,
        "meta": {"merged_from": len(checkpoints)},
    }


# Experiment 1: We teach the AI to read scrambled numbers. 
# Then, we change the scramble pattern completely and teach it again to see if it forgets the first pattern.
def run_scenario_1(batch_size=100):
    perm1 = torch.randperm(784)
    perm2 = torch.randperm(784)
    while torch.equal(perm1, perm2):
        perm2 = torch.randperm(784)

    t1_train, t1_val, t1_test = get_permuted_mnist_loaders(perm1, batch_size=batch_size)
    t2_train, t2_val, t2_test = get_permuted_mnist_loaders(perm2, batch_size=batch_size)

    ckpt = run_hyperparameter_search(
        "scenario_1_reformatted_mnist",
        t1_train, t1_val, t1_test,
        t2_train, t2_val, t2_test,
        input_dim=784,
        output_dim=10,
        trials_per_condition=3,
        max_epochs_old=15,
        max_epochs_new=15,
        patience_old=3,
        patience_new=3,
    )
    path = os.path.join(RESULTS_DIR, "scenario_1_checkpoint.pt")
    torch.save(ckpt, path)
    print(f"Saved {path}")


# Experiment 2: We teach the AI to review Amazon products (like Books). 
# Then, we switch it to a different product category (like DVDs) and see what it remembers.
def run_scenario_2_all_pairs(base_path="data/amazon", batch_size=100):
    print("Running Scenario 2 (all Amazon pairs)")
    checkpoints = []

    for old_cat, new_cat in combinations(AMAZON_CATEGORIES, 2):
        print(f"\nScenario 2 pair: {old_cat} -> {new_cat}")
        old_npz = os.path.join(base_path, f"{old_cat}.npz")
        new_npz = os.path.join(base_path, f"{new_cat}.npz")

        t1_train, t1_val, t1_test, input_dim1, output_dim1 = get_amazon_from_npz(old_npz, batch_size=batch_size)
        t2_train, t2_val, t2_test, input_dim2, output_dim2 = get_amazon_from_npz(new_npz, batch_size=batch_size)

        if input_dim1 != input_dim2:
            raise ValueError(f"Feature dims differ: {input_dim1} vs {input_dim2}")
        if output_dim1 != output_dim2:
            raise ValueError(f"Class counts differ: {output_dim1} vs {output_dim2}")

        ckpt_pair = run_hyperparameter_search(
            f"scenario_2_{old_cat}_to_{new_cat}",
            t1_train, t1_val, t1_test,
            t2_train, t2_val, t2_test,
            input_dim=input_dim1,
            output_dim=output_dim1,
            trials_per_condition=10,
            max_epochs_old=50,
            max_epochs_new=50,
            patience_old=20,
            patience_new=20,
        )
        checkpoints.append(ckpt_pair)

    merged = _merge_checkpoints(checkpoints, "scenario_2_similar_amazon_all_pairs")
    path = os.path.join(RESULTS_DIR, "scenario_2_checkpoint.pt")
    torch.save(merged, path)
    print(f"Saved {path}")


# Experiment 3: The ultimate test! We teach the AI to recognize handwritten numbers. 
# Then, we suddenly force it to analyze text from Amazon reviews to see how it handles a completely different task.
def run_scenario_3_all_amazon(base_path="data/amazon", batch_size=100):
    print("Running Scenario 3 (MNIST vs all Amazon categories)")
    checkpoints = []

    for cat in AMAZON_CATEGORIES:
        print(f"\nScenario 3 category: MNIST -> {cat}")
        amazon_npz = os.path.join(base_path, f"{cat}.npz")

        a_train, a_val, a_test, _, amazon_classes = get_amazon_from_npz(amazon_npz, batch_size=batch_size)
        if amazon_classes != 2:
            raise ValueError("Scenario 3 expects a binary Amazon task.")

        target_dim = 784
        a_train, a_val, a_test = reduce_feature_dim(a_train, a_val, a_test, target_dim=target_dim)
        m_train, m_val, m_test = get_padded_binary_mnist_loaders(
            target_dim=target_dim,
            classes=(2, 9),
            batch_size=batch_size,
        )

        ckpt_cat = run_hyperparameter_search(
            f"scenario_3_mnist_to_{cat}",
            m_train, m_val, m_test,
            a_train, a_val, a_test,
            input_dim=target_dim,
            output_dim=2,
            trials_per_condition=10,
            max_epochs_old=50,
            max_epochs_new=50,
            patience_old=20,
            patience_new=20,
        )
        checkpoints.append(ckpt_cat)

    merged = _merge_checkpoints(checkpoints, "scenario_3_dissimilar_mnist_vs_all_amazon")
    path = os.path.join(RESULTS_DIR, "scenario_3_checkpoint.pt")
    torch.save(merged, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    RUN_SCENARIO_1 = True
    RUN_SCENARIO_2 = False
    RUN_SCENARIO_3 = False

    if RUN_SCENARIO_1:
        print("Running Scenario 1")
        run_scenario_1(batch_size=100)

    if RUN_SCENARIO_2:
        run_scenario_2(
            os.path.join("data", "amazon", "books.npz"),
            os.path.join("data", "amazon", "dvd.npz"),
            batch_size=100,
        )

    if RUN_SCENARIO_3:
        run_scenario_3(
            os.path.join("data", "amazon", "books.npz"),
            batch_size=100,
        )