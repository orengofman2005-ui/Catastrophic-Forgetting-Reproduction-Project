"""
prepare_amazon_npz.py
=====================
Preprocesses the Amazon product review dataset (Dredze et al.) into .npz files
for use in Scenarios 2 and 3 of the catastrophic forgetting reproduction.

Each category (books, dvd, electronics, kitchen) is vectorized using a shared
DictVectorizer fitted on the union of all categories, then projected onto the
top-5000 features by corpus frequency. Each category is split 80/20
(train/test) with stratification, and saved as a compressed .npz file.

Usage:
    python prepare_amazon_npz.py

Expected input layout:
    data/amazon/{books,dvd,electronics,kitchen}/{positive,negative}.review

Output:
    data/amazon/{books,dvd,electronics,kitchen}.npz
    Each file contains: X_train, y_train, X_test, y_test
"""

import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

CATEGORIES = ["books", "dvd", "electronics", "kitchen"]


def parse_review_line(line: str):
    """Parse one line of the Dredze feature-value format into a dict."""
    feats = {}
    for item in line.strip().split():
        if ":" not in item:
            continue
        key, val = item.split(":", 1)
        try:
            feats[key] = float(val)
        except ValueError:
            continue
    return feats


def load_category_rows(category_path):
    """Load positive (label=1) and negative (label=0) reviews for one category."""
    rows = []
    labels = []

    for label, fname in [(1, "positive.review"), (0, "negative.review")]:
        file_path = os.path.join(category_path, fname)
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                rows.append(parse_review_line(line))
                labels.append(label)

    return rows, np.array(labels, dtype=np.int64)


def build_shared_vectorizer(base_path="data/amazon", max_features=5000):
    """Fit a DictVectorizer on all categories; return it and the top-k feature indices."""
    all_rows = []
    for cat in CATEGORIES:
        cat_path = os.path.join(base_path, cat)
        rows, _ = load_category_rows(cat_path)
        all_rows.extend(rows)

    vectorizer = DictVectorizer(sparse=True)
    X_sparse = vectorizer.fit_transform(all_rows)

    # Select top-max_features features by aggregate corpus frequency.
    freqs = np.asarray(X_sparse.sum(axis=0)).ravel()
    keep_idx = np.argsort(freqs)[::-1][:max_features]
    keep_idx = np.sort(keep_idx)

    return vectorizer, keep_idx


def vectorize_category(category_path, vectorizer, keep_idx):
    """Project one category onto the shared feature space and split train/test."""
    rows, y = load_category_rows(category_path)

    X = vectorizer.transform(rows)
    X = X[:, keep_idx].toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, y_train, X_test, y_test


def save_all_npz(base_path="data/amazon", max_features=5000):
    vectorizer, keep_idx = build_shared_vectorizer(
        base_path=base_path,
        max_features=max_features,
    )

    for cat in CATEGORIES:
        cat_path = os.path.join(base_path, cat)
        X_train, y_train, X_test, y_test = vectorize_category(
            cat_path,
            vectorizer,
            keep_idx,
        )

        out_path = os.path.join(base_path, f"{cat}.npz")
        np.savez(
            out_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        print(f"Saved {out_path} | train={X_train.shape} | test={X_test.shape}")


if __name__ == "__main__":
    save_all_npz()
    print("Preprocessing complete. Files saved to data/amazon/*.npz")