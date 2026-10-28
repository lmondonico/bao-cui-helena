"""
Nested CV + Ensemble Training for Random Forest Baseline.
Uses same experimental setup as HELENA for fair comparison.

Configuration:
- 5 outer folds (stratified by current density + rare moieties)
- 8 ensemble models per outer fold (40 total)
- Predictions averaged across ensemble
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Baseline_Utils.PNAS_featurizer import PNASFeaturizer
from Baseline_Utils.nested_cv_utils import (
    MASTER_SEED,
    N_OUTER_FOLDS,
    N_ENSEMBLE,
    VAL_SPLIT,
    TeeLogger,
    set_seed,
    get_stratification_groups,
    get_results_dir,
    calculate_regression_metrics,
    save_results,
    plot_parity,
    print_final_results,
)

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}


def extract_features(dataset, featurizer):
    """Extract PNAS features plus current and method from dataset."""
    features_list = []
    for _, sample in dataset.iterrows():
        features = featurizer.calculate_features(sample)
        features_list.append([
            features["sO"],
            features["FO"],
            features["InOr"],
            features["aC"],
            features["log_current"],
            features["method"],
        ])
    return np.array(features_list)


def train_single_model(X_train_full, y_train_full, X_test, seed):
    """Train a single Random Forest model with train/val split."""
    set_seed(seed)

    # Inner train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_full)), test_size=VAL_SPLIT, random_state=seed
    )
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    # Train Random Forest
    params = RF_PARAMS.copy()
    params["random_state"] = seed

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Get predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)

    metrics = {
        "train_r2": train_r2,
        "train_rmse": np.sqrt(np.mean((y_train - train_preds) ** 2)),
        "val_r2": val_r2,
        "val_rmse": np.sqrt(np.mean((y_val - val_preds) ** 2)),
    }

    return model, test_preds, metrics


def main():
    set_seed(MASTER_SEED)
    start_time = time.time()

    # Setup results directory and logging
    results_dir = get_results_dir("RandomForest")
    log_path = os.path.join(results_dir, "training.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"Results directory: {results_dir}")
    print(f"\n{'=' * 60}")
    print("RANDOM FOREST NESTED CV + ENSEMBLE TRAINING")
    print(f"{'=' * 60}")
    print(f"Outer folds: {N_OUTER_FOLDS}")
    print(f"Ensemble size: {N_ENSEMBLE}")
    print(f"Total models: {N_OUTER_FOLDS * N_ENSEMBLE}")

    # Load dataset
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "HELENA-CE-26.csv")
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=MASTER_SEED).reset_index(drop=True)
    print(f"\nDataset: {len(data)} samples")

    # Extract features
    featurizer = PNASFeaturizer()
    X_all = extract_features(data, featurizer)
    y_all = data["LCE"].values
    row_ids = np.arange(len(data))

    # Get stratification groups
    strat_groups = get_stratification_groups(data)

    # Setup outer CV
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=False)
    folds = list(outer_cv.split(X_all, strat_groups))

    # Storage for results
    all_predictions = {}
    outer_fold_metrics = []
    inner_fold_records = []
    individual_model_test_metrics = []

    for outer_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"OUTER FOLD {outer_idx + 1}/{N_OUTER_FOLDS}")
        print(f"{'=' * 60}")

        X_train_full = X_all[train_idx]
        y_train_full = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        test_row_ids = row_ids[test_idx]

        print(f"Train: {len(X_train_full)}, Test: {len(X_test)}")

        ensemble_test_preds = []

        for m in range(N_ENSEMBLE):
            seed = MASTER_SEED + outer_idx * 100 + m

            model, test_preds, metrics = train_single_model(
                X_train_full, y_train_full, X_test, seed
            )

            ensemble_test_preds.append(test_preds)

            # Calculate individual model test metrics
            model_test_r2 = r2_score(y_test, test_preds)
            model_test_rmse = np.sqrt(np.mean((y_test - test_preds) ** 2))
            individual_model_test_metrics.append({
                "outer_fold": outer_idx + 1,
                "model_idx_in_fold": m + 1,
                "test_r2": model_test_r2,
                "test_rmse": model_test_rmse,
            })

            inner_fold_records.append({
                "outer_fold": outer_idx + 1,
                "model_idx": m + 1,
                "seed": seed,
                "train_r2": metrics["train_r2"],
                "train_rmse": metrics["train_rmse"],
                "val_r2": metrics["val_r2"],
                "val_rmse": metrics["val_rmse"],
            })

            print(f"  Model {m + 1}: Train R²={metrics['train_r2']:.3f}, Val R²={metrics['val_r2']:.3f}")

        # Average ensemble predictions
        ensemble_preds = np.array(ensemble_test_preds)
        mean_preds = np.mean(ensemble_preds, axis=0)

        # Store predictions
        for rid, pred, preds_list in zip(test_row_ids, mean_preds, ensemble_preds.T):
            if rid not in all_predictions:
                all_predictions[rid] = []
            all_predictions[rid].extend(preds_list.tolist())

        # Calculate fold metrics
        fold_metrics = calculate_regression_metrics(mean_preds, y_test)
        outer_fold_metrics.append({
            "outer_fold": outer_idx + 1,
            "n_test": len(y_test),
            "r2": fold_metrics["r2"],
            "rmse": fold_metrics["rmse"],
            "mae": fold_metrics["mae"],
        })
        print(f"Fold {outer_idx + 1} Ensemble: R²={fold_metrics['r2']:.3f}, RMSE={fold_metrics['rmse']:.3f}")

    # Aggregate final predictions
    final_row_ids = []
    final_targets = []
    final_predictions = []
    n_preds_list = []

    for rid in sorted(all_predictions.keys()):
        preds = all_predictions[rid]
        final_row_ids.append(rid)
        final_targets.append(y_all[rid])
        final_predictions.append(np.mean(preds))
        n_preds_list.append(len(preds))

    final_targets = np.array(final_targets)
    final_predictions = np.array(final_predictions)

    # Calculate final metrics
    final_metrics = calculate_regression_metrics(final_predictions, final_targets)
    final_metrics["n_samples"] = len(final_targets)
    final_metrics["avg_preds_per_sample"] = np.mean(n_preds_list)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "row_id": final_row_ids,
        "actual_lce": final_targets,
        "predicted_lce": final_predictions,
        "n_predictions": n_preds_list,
        "error": final_targets - final_predictions,
        "abs_error": np.abs(final_targets - final_predictions),
    })

    # Save results
    save_results(results_dir, predictions_df, final_metrics, outer_fold_metrics, inner_fold_records, individual_model_test_metrics)

    # Generate parity plot
    plot_parity(
        final_targets,
        final_predictions,
        final_metrics,
        os.path.join(results_dir, "nested_cv_parity_plot.png"),
        model_name="Random Forest"
    )

    # Print final results
    print_final_results(final_metrics, "Random Forest")

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {results_dir}")

    # Close logging
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
