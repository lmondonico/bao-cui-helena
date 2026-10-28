"""
Plotting functions for HELENA-CE classification.

This module contains confusion matrices, F1 score curves, and learning-rate
plots for CE classification.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ce_kfold_confusion_matrices(all_kfold_metrics, save_dir, num_classes=2, seed=None):
    """
    Plot confusion matrices for all CE folds in horizontal layout (5 in one row).

    Args:
        all_kfold_metrics (dict): Dictionary containing metrics for all folds
        save_dir (str): Directory to save the plots
        num_classes (int): Number of classes (2 for binary, 3 for ternary)

    Generates:
        - kfold_training_confusion_matrices.png: Training confusion matrices for all folds
        - kfold_validation_confusion_matrices.png: Validation confusion matrices for all folds
    """
    if num_classes == 2:
        class_labels = ["< 2.3", ">= 2.3"]
    elif num_classes == 3:
        class_labels = ["< 2.0", "[2.0, 2.3)", ">= 2.3"]
    else:
        class_labels = [str(i) for i in range(num_classes)]

    n_folds = len(all_kfold_metrics)

    # Training confusion matrices - HORIZONTAL LAYOUT: 1 row × n_folds columns
    fig_train, axes_train = plt.subplots(1, n_folds, figsize=(5 * n_folds, 5))
    if n_folds == 1:
        axes_train = [axes_train]  # Make it subscriptable

    for fold_idx, fold_data in all_kfold_metrics.items():
        ax = axes_train[fold_idx]
        cm_train = fold_data["ce_metrics"]["Training"]["confusion_matrix"]
        sns.heatmap(
            cm_train,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            cbar=False,
            square=True,
            annot_kws={"size": 12},
        )
        ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    plt.suptitle(
        "CE Training Confusion Matrices Across K-Folds",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    suffix = f"_{seed}" if seed is not None else ""
    train_cm_path = os.path.join(save_dir, f"kfold_training_confusion_matrices{suffix}.png")
    plt.savefig(train_cm_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"CE per-fold training confusion matrices saved to {train_cm_path}")

    # Validation confusion matrices - HORIZONTAL LAYOUT: 1 row × 5 columns
    fig_val, axes_val = plt.subplots(1, n_folds, figsize=(25, 5))

    for fold_idx, fold_data in all_kfold_metrics.items():
        ax = axes_val[fold_idx]
        cm_val = fold_data["ce_metrics"]["Validation"]["confusion_matrix"]
        sns.heatmap(
            cm_val,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            cbar=False,
            square=True,
            annot_kws={"size": 12},
        )
        ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    plt.suptitle(
        "CE Validation Confusion Matrices Across K-Folds",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    val_cm_path = os.path.join(save_dir, f"kfold_validation_confusion_matrices{suffix}.png")
    plt.savefig(val_cm_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"CE per-fold validation confusion matrices saved to {val_cm_path}")


def plot_ce_final_confusion_matrices(
    train_results, val_results, test_results, metrics, save_dir, num_classes=2, seed=None
):
    """
    Plot confusion matrices for train, val, and test sets for best CE model.

    Args:
        train_results (dict): Training set results
        val_results (dict): Validation set results
        test_results (dict): Test set results
        metrics (dict): Computed metrics for all splits
        save_dir (str): Directory to save the plot
        num_classes (int): Number of classes (2 for binary, 3 for ternary)

    Generates:
        - confusion_matrices.png: Side-by-side confusion matrices for train/val/test
    """
    if num_classes == 2:
        class_labels = ["< 2.3", ">= 2.3"]
    elif num_classes == 3:
        class_labels = ["< 2.0", "[2.0, 2.3)", ">= 2.3"]
    else:
        class_labels = [str(i) for i in range(num_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (results, split_name) in enumerate(
        [
            (train_results, "Training"),
            (val_results, "Validation"),
            (test_results, "Test"),
        ]
    ):
        cm = metrics[split_name]["confusion_matrix"]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes[idx],
            cbar=True,
            annot_kws={"size": 14},
            cbar_kws={"shrink": 0.8},
        )

        axes[idx].set_title(
            f"{split_name} Confusion Matrix", fontsize=16, fontweight="bold"
        )
        axes[idx].set_xlabel("Predicted Class (LCE)", fontsize=14)
        axes[idx].set_ylabel("True Class (LCE)", fontsize=14)
        axes[idx].tick_params(axis="both", which="major", labelsize=13)

    plt.tight_layout()
    suffix = f"_{seed}" if seed is not None else ""
    save_path = os.path.join(save_dir, f"confusion_matrices{suffix}.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"CE confusion matrices saved to {save_path}")


def plot_ce_kfold_f1_curves(all_kfold_metrics, save_dir, seed=None):
    """
    Plot CE F1 score curves for all folds in mosaic layout (3 top, 2 bottom).

    Args:
        all_kfold_metrics (dict): Dictionary containing metrics for all folds
        save_dir (str): Directory to save the plot

    Generates:
        - kfold_f1_curves.png: F1 score learning curves for all folds
    """
    # Create figure with 2 rows: 3 plots on top, 2 on bottom
    fig = plt.figure(figsize=(18, 10))

    for fold_idx, fold_data in all_kfold_metrics.items():
        # Position: first 3 in top row, last 2 in bottom row
        if fold_idx < 3:
            ax = plt.subplot(2, 3, fold_idx + 1)
        else:
            # Center the bottom row plots
            ax = plt.subplot(2, 3, fold_idx + 2)

        train_f1 = fold_data["loss_histories"].get("ce_train_f1_scores", [])
        val_f1 = fold_data["loss_histories"].get("ce_val_f1_scores", [])

        if train_f1 and val_f1:
            # Prepend 0 to start all curves from origin (0, 0)
            train_f1 = [0] + train_f1
            val_f1 = [0] + val_f1
            epochs = range(len(train_f1))
            ax.plot(epochs, train_f1, label="Training", color="blue", linewidth=2)
            ax.plot(epochs, val_f1, label="Validation", color="orange", linewidth=2)

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Weighted F1 Score", fontsize=12)
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")
            ax.legend(loc="lower right", fontsize=11, frameon=False)
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=11)
            ax.set_ylim([0, 1])  # F1 score is between 0 and 1
        else:
            ax.text(
                0.5, 0.5, "No F1 data available", ha="center", va="center", fontsize=12
            )
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")

    plt.suptitle(
        "CE Weighted F1 Score Curves Across K-Folds",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    suffix = f"_{seed}" if seed is not None else ""
    f1_curves_path = os.path.join(save_dir, f"kfold_f1_curves{suffix}.png")
    plt.savefig(f1_curves_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"CE F1 score curves saved to {f1_curves_path}")


def plot_kfold_lr_curves(all_kfold_metrics, save_dir, seed=None):
    """
    Plot learning rate curves for all folds in a single row layout.

    Shows the learning rate schedule during training.

    Args:
        all_kfold_metrics (dict): Dictionary containing metrics for all folds
        save_dir (str): Directory to save the plot

    Generates:
        - kfold_lr_curves.png: Learning rate curves for all folds
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for fold_idx, fold_data in all_kfold_metrics.items():
        ax = axes[fold_idx]
        lr_history = fold_data["loss_histories"].get("learning_rates", [])

        if lr_history:
            epochs = range(1, len(lr_history) + 1)
            ax.plot(epochs, lr_history, color="green", linewidth=2)

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")
            ax.set_yscale("log")
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=11)
        else:
            ax.text(
                0.5, 0.5, "No LR data available", ha="center", va="center", fontsize=12
            )
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")

    plt.suptitle(
        "Learning Rate Schedule",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    suffix = f"_{seed}" if seed is not None else ""
    lr_curves_path = os.path.join(save_dir, f"kfold_lr_curves{suffix}.png")
    plt.savefig(lr_curves_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Learning rate curves saved to {lr_curves_path}")
