"""
Visualization script for nested CV ensemble binary classification results.
Generates: confusion matrix, loss-curve, and F1 mosaics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import json
import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "data")

# Output settings
DPI = 900

# Classification threshold (from train_parameters.py)
LCE_THRESHOLD = 2.0


def load_data():
    """Load all CSV data from results directory."""
    prediction_df = pd.read_csv(os.path.join(RESULTS_DIR, "all_prediction_data.csv"))
    loss_df = pd.read_csv(os.path.join(RESULTS_DIR, "all_loss_data.csv"))
    inner_results_df = pd.read_csv(os.path.join(RESULTS_DIR, "inner_fold_results.csv"))
    outer_metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "outer_fold_metrics.csv"))
    with open(os.path.join(RESULTS_DIR, "nested_cv_metrics.json"), "r") as f:
        final_metrics = json.load(f)

    return prediction_df, loss_df, inner_results_df, outer_metrics_df, final_metrics


def majority_vote(predictions):
    """Apply majority voting to a list of predictions."""
    return np.bincount(predictions.astype(int), minlength=2).argmax()


def compute_ensemble_predictions_per_fold(prediction_df):
    """
    Compute ensemble predictions (majority vote of 8 models) per outer fold.
    Returns dict with fold -> {train_df, val_df, test_df}
    """
    fold_data = {}

    for fold in range(1, 6):
        fold_preds = prediction_df[prediction_df["outer_fold"] == fold]

        # Train: majority vote of 8 models for this fold
        train_df = fold_preds[fold_preds["split"] == "train"].copy()
        train_agg = (
            train_df.groupby("row_id")
            .agg({"prediction": majority_vote, "target": "first"})
            .reset_index()
        )

        # Val: majority vote of 8 models for this fold
        val_df = fold_preds[fold_preds["split"] == "val"].copy()
        val_agg = (
            val_df.groupby("row_id")
            .agg({"prediction": majority_vote, "target": "first"})
            .reset_index()
        )

        # Test: majority vote of 8 models for this fold
        test_df = fold_preds[fold_preds["split"] == "test"].copy()
        test_agg = (
            test_df.groupby("row_id")
            .agg({"prediction": majority_vote, "target": "first"})
            .reset_index()
        )

        fold_data[fold] = {"train": train_agg, "val": val_agg, "test": test_agg}

    return fold_data


def plot_confusion_matrix_on_ax(ax, targets, predictions, title=None):
    """Plot a confusion matrix on the given axis."""
    cm = confusion_matrix(targets, predictions)
    labels = [f"< {LCE_THRESHOLD}", f">= {LCE_THRESHOLD}"]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    return im


def plot_confusion_mosaic(fold_data, output_path):
    """
    Generate 5x3 grid confusion matrix plot.
    - 5 rows for each outer fold
    - Column 1: Train confusion matrix
    - Column 2: Validation confusion matrix
    - Column 3: Test confusion matrix
    """
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for fold in range(1, 6):
        ax_train = axes[fold - 1, 0]
        ax_val = axes[fold - 1, 1]
        ax_test = axes[fold - 1, 2]
        fd = fold_data[fold]

        # Left column: Train
        plot_confusion_matrix_on_ax(
            ax_train, fd["train"]["target"].values, fd["train"]["prediction"].values
        )

        # Add fold label on the left side
        ax_train.text(
            -0.45,
            0.5,
            f"Outer Fold {fold}",
            transform=ax_train.transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

        # Middle column: Validation
        plot_confusion_matrix_on_ax(
            ax_val, fd["val"]["target"].values, fd["val"]["prediction"].values
        )

        # Right column: Test
        plot_confusion_matrix_on_ax(
            ax_test, fd["test"]["target"].values, fd["test"]["prediction"].values
        )

    # Column titles
    axes[0, 0].set_title("Training", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("Validation", fontsize=14, fontweight="bold")
    axes[0, 2].set_title("Test", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion mosaic to {output_path}")


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
    """
    fig, axes = plt.subplots(5, 1, figsize=(8, 15))

    for fold in range(1, 6):
        ax = axes[fold - 1]

        # Filter data for this fold
        fold_data = loss_df[loss_df["outer_fold"] == fold]

        # Average across all 8 models per epoch
        stats = (
            fold_data.groupby("epoch")
            .agg({"train_loss": "mean", "val_loss": "mean"})
            .reset_index()
        )

        epochs = stats["epoch"].values
        train_raw = stats["train_loss"].values
        val_raw = stats["val_loss"].values

        # Apply smoothing
        train_smooth = smooth_curve(train_raw.tolist())
        val_smooth = smooth_curve(val_raw.tolist())

        # Faint raw data behind
        ax.plot(epochs, train_raw, color="orange", alpha=0.2, linewidth=1)
        ax.plot(epochs, val_raw, color="blue", alpha=0.2, linewidth=1)

        # Smoothed lines on top
        ax.plot(
            epochs,
            train_smooth,
            color="orange",
            label="Train",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.plot(epochs, val_smooth, color="blue", label="Val", linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(frameon=False, fontsize=10, loc="upper right")

        # Add fold label on the left side
        ax.text(
            -0.12,
            0.5,
            f"Outer Fold {fold}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curves to {output_path}")


def plot_f1_curves_from_mosaics(output_path):
    """
    Assemble the original five Model 1 panel crops into the SI9 mosaic.
    Numeric epoch-level F1 logs are unavailable; these are lossless image sources.
    """
    from PIL import Image

    fig, axes = plt.subplots(5, 1, figsize=(8, 15))

    for fold in range(1, 6):
        ax = axes[fold - 1]

        panel_path = os.path.join(RESULTS_DIR, f"f1_model1_fold_{fold}.png")
        with Image.open(panel_path) as img:
            model1_crop = np.array(img)

        ax.imshow(model1_crop)
        ax.axis("off")

        # Add fold label on the left side
        ax.text(
            -0.08,
            0.5,
            f"Outer Fold {fold}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved F1 curves mosaic to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(SCRIPT_DIR).parents[1] / "results/si_classification")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir; existing figures are never overwritten.")
    print("Loading data...")
    prediction_df, loss_df, inner_df, outer_df, final_metrics = load_data()
    args.output_dir.mkdir(parents=True)

    print(
        f"Loaded {len(prediction_df)} prediction records, {len(loss_df)} loss records"
    )
    print(f"Inner results: {len(inner_df)} models, Outer folds: {len(outer_df)}")
    print(f"Final metrics: Acc={final_metrics['accuracy']}, F1={final_metrics['f1']}")

    print("\nComputing ensemble predictions per fold...")
    fold_data = compute_ensemble_predictions_per_fold(prediction_df)

    for fold in range(1, 6):
        fd = fold_data[fold]
        print(
            f"  Fold {fold}: Train={len(fd['train'])}, Val={len(fd['val'])}, Test={len(fd['test'])}"
        )

    print("\nGenerating 5x3 confusion matrix mosaic...")
    confusion_path = args.output_dir / "nested_cv_confusion_mosaic.png"
    plot_confusion_mosaic(fold_data, confusion_path)

    print("\nGenerating 5x1 loss curves mosaic...")
    loss_path = args.output_dir / "nested_cv_binary_loss_curves.png"
    plot_loss_curves(loss_df, loss_path)

    print("\nGenerating 5x1 F1 curves mosaic (from existing mosaics)...")
    f1_path = args.output_dir / "nested_cv_f1_curves.png"
    plot_f1_curves_from_mosaics(f1_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
