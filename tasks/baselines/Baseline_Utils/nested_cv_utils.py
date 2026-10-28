"""
Shared utilities for nested CV + ensemble training across all baseline models.
Matches the configuration used in HELENA's train_ensemble_regression.py.
"""

import sys
import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tasks.training.train_parameters import get_combined_stratification_groups
from modules.tokenizer.tokenizer_helena import Tokenizer

# Constants matching HELENA ensemble training
MASTER_SEED = 142
N_OUTER_FOLDS = 5
N_ENSEMBLE = 8
VAL_SPLIT = 0.1


class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, filename, mode="w"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_stratification_groups(data):
    """Get stratification groups for CV splits (by current density + rare moieties)."""
    tokenizer = Tokenizer()
    return get_combined_stratification_groups(data, tokenizer)


def get_results_dir(model_name):
    """Create a unique results directory for the model."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results_base = os.path.join(project_root, "results")

    results_dir = os.path.join(results_base, f"Baseline_{model_name}_Ensemble_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def calculate_regression_metrics(predictions, targets):
    """Calculate regression metrics."""
    return {
        "r2": r2_score(targets, predictions),
        "rmse": np.sqrt(mean_squared_error(targets, predictions)),
        "mae": np.mean(np.abs(targets - predictions)),
    }


def save_results(results_dir, predictions_df, metrics, outer_fold_metrics, inner_fold_records=None, individual_model_test_metrics=None):
    """Save all results to files."""
    # Save predictions
    predictions_df.to_csv(
        os.path.join(results_dir, "nested_cv_predictions.csv"), index=False, float_format="%.3f"
    )

    # Round metrics for JSON
    rounded_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            rounded_metrics[k] = round(v, 3)
        else:
            rounded_metrics[k] = v

    # Save metrics JSON
    with open(os.path.join(results_dir, "nested_cv_metrics.json"), "w") as f:
        json.dump(rounded_metrics, f, indent=2)

    # Save outer fold metrics
    pd.DataFrame(outer_fold_metrics).to_csv(
        os.path.join(results_dir, "outer_fold_metrics.csv"), index=False, float_format="%.3f"
    )

    # Save inner fold records if provided
    if inner_fold_records is not None:
        pd.DataFrame(inner_fold_records).to_csv(
            os.path.join(results_dir, "inner_fold_results.csv"), index=False, float_format="%.3f"
        )

    # Save individual model test metrics if provided
    if individual_model_test_metrics is not None:
        pd.DataFrame(individual_model_test_metrics).to_csv(
            os.path.join(results_dir, "individual_model_test_metrics.csv"), index=False, float_format="%.3f"
        )


def plot_parity(targets, predictions, metrics, path, model_name="Model"):
    """Generate parity plot for actual vs predicted values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5, s=50, edgecolors="none", c="green")

    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    plt.xlabel("Actual LCE", fontsize=14)
    plt.ylabel("Predicted LCE", fontsize=14)
    plt.title(
        f"{model_name} - R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def print_final_results(metrics, model_name):
    """Print final results summary."""
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS - {model_name} (Nested CV + Ensemble)")
    print(f"{'=' * 60}")
    print(f"R²:   {metrics['r2']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"N samples: {metrics['n_samples']}")
    print(f"Avg predictions per sample: {metrics['avg_preds_per_sample']:.1f}")
