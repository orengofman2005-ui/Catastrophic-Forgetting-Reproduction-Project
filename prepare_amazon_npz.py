import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# This is a list of the different types of Amazon products we want to look at.
CATEGORIES = ["books", "dvd", "electronics", "kitchen"]


# This function takes a single line of text from a review file.
# It breaks the text apart to find important words and their scores.
def parse_review_line(line: str):
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


# This function opens the folders for a specific product type.
# It reads both the positive reviews (label 1) and negative reviews (label 0).
def load_category_rows(category_path):
    rows = []
    labels = []

    for label, fname in [(1, "positive.review"), (0, "negative.review")]:
        file_path = os.path.join(category_path, fname)
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                rows.append(parse_review_line(line))
                labels.append(label)

    return rows, np.array(labels, dtype=np.int64)


# This function looks at all reviews across all categories.
# It figures out which 5000 words are the most common overall.
def build_shared_vectorizer(base_path="data/amazon", max_features=5000):
    all_rows = []
    for cat in CATEGORIES:
        cat_path = os.path.join(base_path, cat)
        rows, _ = load_category_rows(cat_path)
        all_rows.extend(rows)

    # This tool turns our text words into a mathematical format (numbers) 
    # that a computer can easily understand and calculate.
    vectorizer = DictVectorizer(sparse=True)
    X_sparse = vectorizer.fit_transform(all_rows)

    # Top features by total frequency across all domains
    # We sort the words by how often they appear and keep only the top ones.
    freqs = np.asarray(X_sparse.sum(axis=0)).ravel()
    keep_idx = np.argsort(freqs)[::-1][:max_features]
    keep_idx = np.sort(keep_idx)

    return vectorizer, keep_idx


# This function takes the reviews for one specific category and converts
# them into numbers. Then, it splits them into a "practice" group for learning
# and a "test" group to check how well the computer actually learned.
def vectorize_category(category_path, vectorizer, keep_idx):
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


# This is the main starting point. It goes through every category,
# prepares the text into numbers, and saves them into ready-to-use files.
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
        print(f"Saved {out_path} | train {X_train.shape} | test {X_test.shape}")


if __name__ == "__main__":
    save_all_npz()