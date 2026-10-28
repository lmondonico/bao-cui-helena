"""
Visualization script for nested CV ensemble regression results.
Generates: parity plot (5x2 grid) and loss curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.metrics import r2_score, mean_squared_error
import os
import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "data")

# Output settings
DPI = 900

# Default split colors
TRAIN_COLOR = "#1f77b4"  # Blue
VAL_COLOR = "#ff7f0e"    # Orange
TEST_COLOR = "#2ca02c"   # Green


def load_data():
    """Load all CSV data from results directory."""
    import json
    parity_df = pd.read_csv(os.path.join(RESULTS_DIR, "all_parity_data.csv"))
    loss_df = pd.read_csv(os.path.join(RESULTS_DIR, "all_loss_data.csv"))
    inner_results_df = pd.read_csv(os.path.join(RESULTS_DIR, "inner_fold_results.csv"))
    outer_metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "outer_fold_metrics.csv"))
    individual_test_df = pd.read_csv(os.path.join(RESULTS_DIR, "individual_model_test_metrics.csv"))
    with open(os.path.join(RESULTS_DIR, "nested_cv_metrics.json"), "r") as f:
        final_metrics = json.load(f)

    return parity_df, loss_df, inner_results_df, outer_metrics_df, individual_test_df, final_metrics


def compute_ensemble_predictions_per_fold(parity_df):
    """
    Compute ensemble predictions (mean of 8 models) per outer fold.
    Returns dict with fold -> {train_df, val_df, test_df}
    """
    fold_data = {}

    for fold in range(1, 6):
        fold_parity = parity_df[parity_df['outer_fold'] == fold]

        # Train: mean of 8 models for this fold
        train_df = fold_parity[fold_parity['split'] == 'train'].copy()
        train_agg = train_df.groupby('row_id').agg({
            'prediction': 'mean',
            'target': 'first'
        }).reset_index()

        # Val: mean of 8 models for this fold
        val_df = fold_parity[fold_parity['split'] == 'val'].copy()
        val_agg = val_df.groupby('row_id').agg({
            'prediction': 'mean',
            'target': 'first'
        }).reset_index()

        # Test: mean of 8 models for this fold
        test_df = fold_parity[fold_parity['split'] == 'test'].copy()
        test_agg = test_df.groupby('row_id').agg({
            'prediction': 'mean',
            'target': 'first'
        }).reset_index()

        fold_data[fold] = {
            'train': train_agg,
            'val': val_agg,
            'test': test_agg
        }

    return fold_data


def compute_combined_ensemble(parity_df):
    """
    Compute combined ensemble predictions across all folds for the combined plot.
    """
    # For combined plot: each test sample appears in exactly 1 outer fold
    test_df = parity_df[parity_df['split'] == 'test'].copy()
    test_ensemble = test_df.groupby('row_id').agg({
        'prediction': 'mean',
        'target': 'first'
    }).reset_index()

    # Train/val: average across all folds
    train_df = parity_df[parity_df['split'] == 'train'].copy()
    train_agg = train_df.groupby('row_id').agg({
        'prediction': 'mean',
        'target': 'first'
    }).reset_index()

    val_df = parity_df[parity_df['split'] == 'val'].copy()
    val_agg = val_df.groupby('row_id').agg({
        'prediction': 'mean',
        'target': 'first'
    }).reset_index()

    return {'train': train_agg, 'val': val_agg, 'test': test_ensemble}


def plot_parity_grid(fold_data, combined_data, output_path):
    """
    Generate 5x2 grid parity plot.
    - 5 rows for each outer fold
    - Column 1: Train + Val (ensemble average)
    - Column 2: Test (ensemble average)
    Colors: Train=orange, Val=blue, Test=green (matching train_ensemble_regression.py)
    """
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))

    # Point sizes and alphas
    train_alpha = 0.4
    val_alpha = 0.5
    test_alpha = 0.5
    point_size = 12

    # Plot each fold
    for fold in range(1, 6):
        ax_trainval = axes[fold - 1, 0]
        ax_test = axes[fold - 1, 1]
        fd = fold_data[fold]

        # Left column: Train + Val
        ax_trainval.scatter(fd['train']['target'], fd['train']['prediction'],
                            color='#ff6600', alpha=train_alpha, s=point_size,
                            edgecolors='none', label='Train')
        ax_trainval.scatter(fd['val']['target'], fd['val']['prediction'],
                            color='blue', alpha=val_alpha, s=point_size,
                            edgecolors='none', label='Val')

        # Diagonal line
        ax_trainval.plot([0.5, 3.0], [0.5, 3.0], 'k--', linewidth=1.5, alpha=0.5)
        ax_trainval.set_xlim(0.5, 3.0)
        ax_trainval.set_ylim(0.5, 3.0)
        ax_trainval.set_aspect('equal')
        ax_trainval.tick_params(axis='both', labelsize=10)
        ax_trainval.legend(frameon=False, fontsize=10, loc='upper left')
        ax_trainval.set_xlabel('Ground Truth LCE', fontsize=12)
        ax_trainval.set_ylabel('Predicted LCE', fontsize=12)

        # Add fold label on the left side
        ax_trainval.text(-0.25, 0.5, f'Outer Fold {fold}', transform=ax_trainval.transAxes,
                         fontsize=12, fontweight='bold', rotation=90, va='center', ha='center')

        # Right column: Test only
        ax_test.scatter(fd['test']['target'], fd['test']['prediction'],
                        color='green', alpha=test_alpha, s=point_size,
                        edgecolors='none')

        ax_test.plot([0.5, 3.0], [0.5, 3.0], 'k--', linewidth=1.5, alpha=0.5)
        ax_test.set_xlim(0.5, 3.0)
        ax_test.set_ylim(0.5, 3.0)
        ax_test.set_aspect('equal')
        ax_test.tick_params(axis='both', labelsize=10)
        ax_test.set_xlabel('Ground Truth LCE', fontsize=12)
        ax_test.set_ylabel('Predicted LCE', fontsize=12)

    # Column titles
    axes[0, 0].set_title('Train + Validation', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Test', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    # Compute combined test metrics for reporting
    cd = combined_data
    combined_test_r2 = r2_score(cd['test']['target'], cd['test']['prediction'])
    combined_test_rmse = np.sqrt(mean_squared_error(cd['test']['target'], cd['test']['prediction']))

    print(f"Saved parity grid to {output_path}")
    print(f"  Combined Test R² = {combined_test_r2:.3f}, RMSE = {combined_test_rmse:.3f}")

    return combined_test_r2, combined_test_rmse


def smooth_curve(values, weight=0.6):
    """Exponential moving average smoothing for visualization."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss_curves(loss_df, output_path):
    """
    Generate loss curves mosaic: 5 rows (one per outer fold) x 1 column.
    Each subplot shows averaged loss across 8 models for that fold.
    Style matches train_ensemble_regression.py (orange=train, blue=val, smoothed lines).
    """
    fig, axes = plt.subplots(5, 1, figsize=(8, 15))

    for fold in range(1, 6):
        ax = axes[fold - 1]

        # Filter data for this fold
        fold_data = loss_df[loss_df['outer_fold'] == fold]

        # Average across all 8 models per epoch
        stats = fold_data.groupby('epoch').agg({
            'train_loss': 'mean',
            'val_loss': 'mean'
        }).reset_index()

        epochs = stats['epoch'].values
        train_raw = stats['train_loss'].values
        val_raw = stats['val_loss'].values

        # Apply smoothing
        train_smooth = smooth_curve(train_raw.tolist())
        val_smooth = smooth_curve(val_raw.tolist())

        # Faint raw data behind
        ax.plot(epochs, train_raw, color='orange', alpha=0.2, linewidth=1)
        ax.plot(epochs, val_raw, color='blue', alpha=0.2, linewidth=1)

        # Smoothed lines on top (lighter)
        ax.plot(epochs, train_smooth, color='orange', label='Train', linewidth=1.5, alpha=0.8)
        ax.plot(epochs, val_smooth, color='blue', label='Val', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(frameon=False, fontsize=10, loc='upper right')

        # Add fold label on the left side
        ax.text(-0.12, 0.5, f'Outer Fold {fold}', transform=ax.transAxes,
                fontsize=12, fontweight='bold', rotation=90, va='center', ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(SCRIPT_DIR).parents[1] / "results/si_regression")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir; existing figures are never overwritten.")
    print("Loading data...")
    parity_df, loss_df, inner_df, outer_df, individual_df, final_metrics = load_data()
    args.output_dir.mkdir(parents=True)

    print(f"Loaded {len(parity_df)} parity records, {len(loss_df)} loss records")
    print(f"Inner results: {len(inner_df)} models, Outer folds: {len(outer_df)}")
    print(f"Final metrics: R²={final_metrics['r2']}, RMSE={final_metrics['rmse']}")

    print("\nComputing ensemble predictions per fold...")
    fold_data = compute_ensemble_predictions_per_fold(parity_df)
    combined_data = compute_combined_ensemble(parity_df)

    for fold in range(1, 6):
        fd = fold_data[fold]
        print(f"  Fold {fold}: Train={len(fd['train'])}, Val={len(fd['val'])}, Test={len(fd['test'])}")

    print("\nGenerating 5x2 parity grid...")
    parity_path = args.output_dir / "nested_cv_parity_plot.png"
    plot_parity_grid(fold_data, combined_data, parity_path)

    print("\nGenerating 5x1 loss curves mosaic...")
    loss_path = args.output_dir / "nested_cv_loss_curves.png"
    plot_loss_curves(loss_df, loss_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
