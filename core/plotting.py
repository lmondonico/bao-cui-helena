"""
Utility functions for plotting regression results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


class PlotLossesLR:
    """Class for handling training-related plotting functions."""

    @staticmethod
    def plot_training_loss_and_lr(train_losses, val_losses, learning_rates):
        """Plot training and validation loss and learning rate over epochs."""
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Loss plot
        (train_line,) = ax1.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="Train Loss",
            color="blue",
        )
        (val_line,) = ax1.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label="Validation Loss",
            color="orange",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss over Epochs")
        ax1.legend(frameon=False)
        ax1.grid(False)

        # Learning rate plot
        (lr_line,) = ax2.plot(
            range(1, len(learning_rates) + 1),
            learning_rates,
            label="Learning Rate",
            color="green",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend(frameon=False)
        ax2.grid(False)

        plot_elements = {
            "ax1": ax1,
            "ax2": ax2,
            "train_line": train_line,
            "val_line": val_line,
            "lr_line": lr_line,
        }

        plt.tight_layout()

        return fig, plot_elements

    @staticmethod
    def update_loss_lr_plots(
        epoch, train_losses, val_losses, learning_rates, plot_elements
    ):
        """Update the training and learning rate plots."""
        ax1, ax2 = plot_elements["ax1"], plot_elements["ax2"]
        train_line, val_line, lr_line = (
            plot_elements["train_line"],
            plot_elements["val_line"],
            plot_elements["lr_line"],
        )

        # Update loss plot
        train_line.set_data(range(1, epoch + 2), train_losses)
        val_line.set_data(range(1, epoch + 2), val_losses)
        ax1.relim()
        ax1.autoscale_view()

        # Update learning rate plot
        lr_line.set_data(range(1, epoch + 2), learning_rates)
        ax2.relim()
        ax2.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    @staticmethod
    def save_loss_lr_plot(fig, save_path):
        """Save the final loss and learning rate plot."""
        plt.ioff()
        loss_plot_file_name = "regr_loss_plot.png"
        full_save_path = os.path.join(save_path, loss_plot_file_name)
        plt.savefig(full_save_path, dpi=600)
        plt.close()


def plot_results(
    train_results,
    val_results,
    test_results,
    metrics,
    save_path,
):
    """Create evaluation plots with metrics."""

    # Retrieve data
    train_pred = train_results["predictions"]
    train_true = train_results["truths"]
    val_pred = val_results["predictions"]
    val_true = val_results["truths"]
    test_pred = test_results["predictions"]
    test_true = test_results["truths"]

    train_rmse = metrics["Training"]["rmse"]
    train_r2 = metrics["Training"]["r2"]
    val_rmse = metrics["Validation"]["rmse"]
    val_r2 = metrics["Validation"]["r2"]
    test_rmse = metrics["Test"]["rmse"]
    test_r2 = metrics["Test"]["r2"]

    # Create plot
    axis_tick_size = 14
    axis_title_size = 16
    plot_title_size = 14
    plt.figure(figsize=(12, 10))
    plt.scatter(
        train_true,
        train_pred,
        color="orange",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Training",
    )
    plt.scatter(
        val_true,
        val_pred,
        color="blue",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Validation",
    )
    plt.scatter(
        test_true,
        test_pred,
        color="green",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Test",
    )

    # Add y=x line
    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    # Set labels and title with metrics
    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)

    # Build title with metrics
    title_line1 = "Parity Plot (Actual vs Predicted)"
    title_line2 = f"Train: R²={train_r2:.3f} RMSE={train_rmse:.3f} | Val: R²={val_r2:.3f} RMSE={val_rmse:.3f} | Test: R²={test_r2:.3f} RMSE={test_rmse:.3f}"
    plt.title(
        f"{title_line1}\n{title_line2}",
        fontsize=plot_title_size,
        fontweight="bold",
        pad=20,
    )

    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)

    # Add legend
    plt.legend(frameon=False, fontsize=16)
    plt.grid(False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*tight_layout.*",
            category=UserWarning,
        )
        plt.tight_layout()
    regr_plot_file_name = "regr_parity_plot.png"
    regr_plot_full_save_path = os.path.join(save_path, regr_plot_file_name)
    plt.savefig(regr_plot_full_save_path, dpi=600)
    plt.close()


def plot_kfold_cross_results(fold_metrics, save_path):
    """Plot cross-validation results focusing on RMSE and R2."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_to_plot = ["rmse", "r2"]
    # Get validation metrics for each fold - fold_metrics has nested structure: fold -> metrics -> Validation -> metric
    data_to_plot = []
    for metric in metrics_to_plot:
        metric_values = []
        for fold in fold_metrics.keys():
            # Use Validation metrics as the primary evaluation metric for kfold
            metric_values.append(fold_metrics[fold]["metrics"]["Validation"][metric])
        data_to_plot.append(metric_values)

    ax1.boxplot(data_to_plot, labels=["RMSE", "R²"])
    ax1.set_title("Metrics Distribution Across Folds")
    ax1.grid(False)

    # Plot predictions vs actual for all folds (using validation data)
    all_true = []
    all_pred = []
    for fold in fold_metrics.keys():
        all_true.extend(fold_metrics[fold]["val_truths"])
        all_pred.extend(fold_metrics[fold]["val_predictions"])

    ax2.scatter(all_true, all_pred, alpha=0.5)
    ax2.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], "r--")
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title("Predicted vs Actual Values (All Folds)")
    ax2.grid(False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*tight_layout.*",
            category=UserWarning,
        )
        plt.tight_layout()

    # Save plot
    kfold_plot_filename = "kfold_cross_val_results.png"
    plot_save_path = os.path.join(save_path, kfold_plot_filename)
    plt.savefig(plot_save_path, dpi=600)
    plt.close()
    print(f"\nK-fold cross-validation plot saved to: {plot_save_path}")


def plot_results_with_worst_labels(
    train_results,
    val_results,
    test_results,
    metrics,
    save_path,
    data_path=None,
    n_worst_samples=15,
    seed=None,
):
    """Create parity plot with labels for the n_worst_samples worst formulations and save their details to a text file."""

    # Retrieve data
    train_pred = train_results["predictions"]
    train_true = train_results["truths"]
    train_row_id = train_results["row_ids"]
    val_pred = val_results["predictions"]
    val_true = val_results["truths"]
    val_row_id = val_results["row_ids"]
    test_pred = test_results["predictions"]
    test_true = test_results["truths"]
    test_row_id = test_results["row_ids"]

    train_rmse = metrics["Training"]["rmse"]
    train_r2 = metrics["Training"]["r2"]
    val_rmse = metrics["Validation"]["rmse"]
    val_r2 = metrics["Validation"]["r2"]
    test_rmse = metrics["Test"]["rmse"]
    test_r2 = metrics["Test"]["r2"]

    # Combine all data
    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    all_row_id = np.concatenate([train_row_id, val_row_id, test_row_id])
    all_splits = np.concatenate(
        [
            ["Training"] * len(train_true),
            ["Validation"] * len(val_true),
            ["Test"] * len(test_true),
        ]
    )

    # Calculate residuals and standardized residuals
    residuals = all_pred - all_true
    residual_std = np.std(residuals)
    standardized_residuals = np.abs(residuals) / residual_std

    # Find n_worst_samples worst formulations
    worst_indices = np.argsort(standardized_residuals)[-n_worst_samples:]

    # Create plot
    axis_tick_size = 14
    axis_title_size = 16
    plot_title_size = 14
    plt.figure(figsize=(12, 10))

    # Plot points by dataset split
    plt.scatter(
        train_true,
        train_pred,
        color="orange",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Training",
    )
    plt.scatter(
        val_true,
        val_pred,
        color="blue",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Validation",
    )
    plt.scatter(
        test_true,
        test_pred,
        color="green",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Test",
    )

    # Add y=x line
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    # Label worst formulations
    for idx in worst_indices:
        row_id = all_row_id[idx]
        true_val = all_true[idx]
        pred_val = all_pred[idx]

        # Add text annotation with offset to avoid overlap
        offset_x = (max_val - min_val) * 0.02
        offset_y = (max_val - min_val) * 0.02
        plt.annotate(
            f"{int(row_id)}",
            xy=(true_val, pred_val),
            xytext=(true_val + offset_x, pred_val + offset_y),
            fontsize=8,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    # Set labels and title with metrics
    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)

    # Build title with metrics
    title_line1 = f"Parity Plot with {n_worst_samples} Worst Formulations Labeled"
    title_line2 = f"Train: R²={train_r2:.3f} RMSE={train_rmse:.3f} | Val: R²={val_r2:.3f} RMSE={val_rmse:.3f} | Test: R²={test_r2:.3f} RMSE={test_rmse:.3f}"
    plt.title(
        f"{title_line1}\n{title_line2}",
        fontsize=plot_title_size,
        fontweight="bold",
        pad=20,
    )

    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)

    # Add legend
    plt.legend(frameon=False, fontsize=16)
    plt.grid(False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*tight_layout.*",
            category=UserWarning,
        )
        plt.tight_layout()

    # Save worst formulations data to text file
    if data_path is not None:
        try:
            # Load dataset to get electrolyte names and additional data
            dataset = pd.read_csv(data_path)
            row_id_to_electrolyte = dict(zip(dataset["Row_ID"], dataset["Electrolyte"]))

            # Check if dataset has Current column (CE dataset)
            has_current = "Current" in dataset.columns

            # Create dictionaries for temperature and current lookup
            row_id_to_temperature = dict(
                zip(
                    dataset["Row_ID"], dataset.get("Temperature", [25.0] * len(dataset))
                )
            )
            row_id_to_current = (
                dict(
                    zip(dataset["Row_ID"], dataset.get("Current", [0.0] * len(dataset)))
                )
                if has_current
                else {}
            )

            # Create list of worst formulations with details
            worst_formulations = []
            for idx in worst_indices:
                row_id = int(all_row_id[idx])
                deviation = standardized_residuals[idx]
                split = all_splits[idx]
                true_val = all_true[idx]
                pred_val = all_pred[idx]
                electrolyte_name = row_id_to_electrolyte.get(row_id, "Unknown")
                temperature = row_id_to_temperature.get(row_id, 25.0)
                current = row_id_to_current.get(row_id, 0.0) if has_current else None

                formulation_data = {
                    "row_id": row_id,
                    "electrolyte": electrolyte_name,
                    "deviation": deviation,
                    "split": split,
                    "true_val": true_val,
                    "pred_val": pred_val,
                    "temperature": temperature,
                }

                if has_current:
                    formulation_data["current"] = current

                worst_formulations.append(formulation_data)

            # Sort by deviation (worst to best)
            worst_formulations.sort(key=lambda x: x["deviation"], reverse=True)

            # Save to text file
            suffix = f"_{seed}" if seed is not None else ""
            worst_file_path = os.path.join(save_path, f"worst_formulations{suffix}.txt")
            with open(worst_file_path, "w") as f:
                f.write(
                    f"{n_worst_samples} Worst Performing Formulations (Sorted by Standardized Deviation)\n"
                )
                f.write("=" * 100 + "\n\n")

                # Dynamic header based on available data
                if has_current:
                    f.write(
                        "Rank | Row_ID | Std Dev | Split      | True Val | Pred Val | Temp(°C) | Current(mA/cm2) | Electrolyte\n"
                    )
                    f.write("-" * 100 + "\n")
                else:
                    f.write(
                        "Rank | Row_ID | Std Dev | Split      | True Val | Pred Val | Temp(°C) | Electrolyte\n"
                    )
                    f.write("-" * 90 + "\n")

                # Dynamic row formatting based on available data
                for i, form in enumerate(worst_formulations, 1):
                    if has_current:
                        f.write(
                            f"{i:4d} | {form['row_id']:8d} | {form['deviation']:7.3f} | "
                            f"{form['split']:10s} | {form['true_val']:8.3f} | {form['pred_val']:8.3f} | "
                            f"{form['temperature']:7.1f} | {form['current']:14.1f} | "
                            f"{form['electrolyte']}\n"
                        )
                    else:
                        f.write(
                            f"{i:4d} | {form['row_id']:8d} | {form['deviation']:7.3f} | "
                            f"{form['split']:10s} | {form['true_val']:8.3f} | {form['pred_val']:8.3f} | "
                            f"{form['temperature']:7.1f} | {form['electrolyte']}\n"
                        )

                f.write("\n" + "=" * 100 + "\n")
                f.write("Notes:\n")
                f.write(
                    "- Std Dev: Absolute standardized residual (|predicted - actual| / residual_std)\n"
                )
                f.write("- Split: Dataset split where this formulation was evaluated\n")
                f.write("- Temp(°C): Experimental temperature in Celsius\n")
                if has_current:
                    f.write(
                        "- Current(mA/cm2): Applied current density during experiment\n"
                    )
                f.write(
                    "- Higher Std Dev indicates worse model performance for that formulation\n"
                )

            print(f"\nWorst formulations data saved to: {worst_file_path}")

        except Exception as e:
            print(f"Warning: Could not save worst formulations data: {e}")

    suffix = f"_{seed}" if seed is not None else ""
    regr_plot_file_name = f"regr_parity_plot_worst_labeled{suffix}.png"
    regr_plot_full_save_path = os.path.join(save_path, regr_plot_file_name)
    plt.savefig(regr_plot_full_save_path, dpi=600)
    plt.close()


def plot_kfold_loss_curves_mosaic(all_kfold_metrics, save_path, seed=None):
    """Plot loss curves for each fold in a single row layout."""

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    n_folds = len(all_kfold_metrics)

    for fold_idx in range(n_folds):
        ax = axes[fold_idx]
        loss_histories = all_kfold_metrics[fold_idx]["loss_histories"]

        train_losses = loss_histories["train_losses"]
        val_losses = loss_histories["val_losses"]

        if len(train_losses) > 0 and len(val_losses) > 0:
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, color="orange", label="Training", linewidth=2)
            ax.plot(epochs, val_losses, color="blue", label="Validation", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")
            ax.legend(frameon=False, fontsize=10)
            ax.grid(False)
        else:
            ax.text(
                0.5,
                0.5,
                f"No loss data\navailable for\nFold {fold_idx + 1}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"Fold {fold_idx + 1}", fontsize=14, fontweight="bold")

    plt.suptitle(
        "Training and Validation Loss Curves",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    suffix = f"_{seed}" if seed is not None else ""
    plot_path = os.path.join(save_path, f"kfold_loss_curves_mosaic{suffix}.png")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"K-fold loss curves mosaic plot saved to: {plot_path}")


def plot_kfold_parity_plots_mosaic(all_kfold_metrics, save_path, seed=None):
    """Plot parity plots for each fold in a single row layout."""

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    n_folds = len(all_kfold_metrics)

    for fold_idx in range(n_folds):
        ax = axes[fold_idx]

        train_pred = all_kfold_metrics[fold_idx]["train_predictions"]
        train_true = all_kfold_metrics[fold_idx]["train_truths"]
        val_pred = all_kfold_metrics[fold_idx]["val_predictions"]
        val_true = all_kfold_metrics[fold_idx]["val_truths"]

        # Plot training and validation data
        ax.scatter(
            train_true, train_pred, color="orange", alpha=0.6, s=20, label="Training"
        )
        ax.scatter(
            val_true, val_pred, color="blue", alpha=0.6, s=20, label="Validation"
        )

        # Calculate range for y=x line
        all_true = np.concatenate([train_true, val_true])
        all_pred = np.concatenate([train_pred, val_pred])
        min_val = min(np.min(all_true), np.min(all_pred))
        max_val = max(np.max(all_true), np.max(all_pred))

        # Add y=x line
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, alpha=0.8)

        # Get metrics for this fold
        fold_metrics = all_kfold_metrics[fold_idx]["metrics"]
        train_r2 = fold_metrics["Training"]["r2"]
        val_r2 = fold_metrics["Validation"]["r2"]

        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title(
            f"Fold {fold_idx + 1}\nTrain R²={train_r2:.3f}, Val R²={val_r2:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=10)
        ax.grid(False)

    plt.suptitle(
        "K-Fold Cross-Validation: Parity Plots (Actual vs Predicted)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    suffix = f"_{seed}" if seed is not None else ""
    plot_path = os.path.join(save_path, f"kfold_parity_plots_mosaic{suffix}.png")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"K-fold parity plots mosaic plot saved to: {plot_path}")
