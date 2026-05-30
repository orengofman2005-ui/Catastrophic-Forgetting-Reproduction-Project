import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR


# =========================================================
# Files and directories
# =========================================================
FEATURES_CSV = "data/processed/features.csv"
RESULTS_DIR = "results"
MODELS_DIR = "results/models"
METRICS_CSV = "results/metrics.csv"

# Metadata columns that are not used as model features.
# Including cup_id would teach the model "cup identity" rather than
# the real relationship between sound and fill percentage.
META_COLS = [
    "audio_path", "cup_id", "take_id", "window_id",
    "start_time", "end_time", "fill_percent"
]


# =========================================================
# Load data
# =========================================================
def load_data():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [col for col in df.columns if col not in META_COLS]

    X = df[feature_cols].values
    y = df["fill_percent"].values

    # Groups for validation:
    # all samples from the same cup get the same group — essential for GroupKFold
    groups = df["cup_id"].values

    return df, X, y, groups, feature_cols


# =========================================================
# Define classical models
# All are machine learning only — no neural networks, per project requirements
# =========================================================
def get_models():
    return {
        # Ridge Regression: simple linear baseline.
        # Checks whether there is a basic linear relationship between features and fill%.
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        # SVR: non-linear model with RBF kernel.
        # epsilon=1.0 (changed from 2.0 in previous version) —
        # penalizes error above 1% instead of 2%, improving accuracy.
        "svr_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, epsilon=1.0))
        ]),

        # Random Forest: robust and stable model.
        # max_depth=None lets trees grow to the end of the data (better than a hard limit).
        # min_samples_leaf=2 prevents overfitting on single samples.
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        # Gradient Boosting: sometimes gives more accurate predictions than RF.
        # subsample=0.8 adds stochasticity that reduces overfitting.
        # max_depth=4 (changed from 3) allows capturing more complex relationships.
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ),

        # Extra Trees: similar to Random Forest but splits are fully random.
        # Faster to train, and sometimes gives lower variance on small data.
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
    }


# =========================================================
# Train and evaluate a single model within a fold
# =========================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return pred, mse, rmse, mae, r2


# =========================================================
# Save feature importances to CSV
#
# Available only for tree-based models (RF, GB, ET).
# Important for the report: shows what the model "learns" and how it decides.
# =========================================================
def save_feature_importances(model, feature_cols, model_name):
    # Pipeline does not expose feature_importances_ directly — access the inner model
    if isinstance(model, Pipeline):
        inner = model.named_steps.get("model")
    else:
        inner = model

    if not hasattr(inner, "feature_importances_"):
        return

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": inner.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    path = f"results/feature_importances_{model_name}.csv"
    imp_df.to_csv(path, index=False)
    print(f"Saved feature importances: {path}")


# =========================================================
# Training with GroupKFold
#
# GroupKFold ensures that an entire cup stays in the test set only in each fold —
# prevents data leakage where the model would "see" the cup in both training and testing.
# =========================================================
def train_with_groupkfold():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df, X, y, groups, feature_cols = load_data()
    models = get_models()

    # Number of folds cannot exceed the number of cups
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)

    if n_splits < 2:
        raise ValueError("Need at least 2 different cups for GroupKFold")

    gkf = GroupKFold(n_splits=n_splits)
    all_metrics = []

    for model_name, model in models.items():

        print()
        print("======================================")
        print(f"Training model: {model_name}")
        print("======================================")

        fold_predictions = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_cups = np.unique(groups[test_idx])

            pred, mse, rmse, mae, r2 = evaluate_model(
                model, X_train, y_train, X_test, y_test
            )

            print(f"Fold {fold + 1} | Test cups: {test_cups}")
            print(f"  RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.3f}")

            all_metrics.append({
                "model": model_name,
                "fold": fold + 1,
                "test_cups": ",".join(test_cups),
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

            # Save predictions for each window — needed for Ground Truth vs Prediction plots
            fold_df = df.iloc[test_idx].copy()
            fold_df["prediction"] = pred
            fold_df["model"] = model_name
            fold_df["fold"] = fold + 1
            fold_predictions.append(fold_df)

        # Save all predictions for this model
        predictions_df = pd.concat(fold_predictions, ignore_index=True)
        pred_path = f"results/predictions_{model_name}.csv"
        predictions_df.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

        # -------------------------------------------------
        # Final training on all data (after CV).
        #
        # Dual purpose:
        # 1. Compute feature importances on all data (more stable than a single fold)
        # 2. Save a model ready for use in realtime_decision.py
        #
        # Note: this model is NOT used for performance evaluation —
        # correct evaluation is always done via GroupKFold.
        # -------------------------------------------------
        print(f"Fitting final model on all data...")
        final_model = clone(model)
        final_model.fit(X, y)

        save_feature_importances(final_model, feature_cols, model_name)

        model_path = f"results/models/{model_name}.pkl"
        joblib.dump(final_model, model_path)
        print(f"Saved final model: {model_path}")

    # =========================================================
    # Performance summary
    # =========================================================
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print()
    print("======================================")
    print("Summary — mean ± std across folds:")
    print("======================================")

    # Showing mean and std: allows seeing not just average performance
    # but also consistency — a model with high std suggests sensitivity to which cup is tested
    summary = metrics_df.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    print(summary.round(3))

    best_model = metrics_df.groupby("model")["RMSE"].mean().idxmin()
    best_rmse = metrics_df.groupby("model")["RMSE"].mean().min()

    print()
    print(f"Best model by mean RMSE: {best_model} ({best_rmse:.3f}%)")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    train_with_groupkfold()
