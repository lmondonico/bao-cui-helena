"""Train HELENA-CE binary ensembles with nested cross-validation and majority voting."""

import sys
import os
import time
import random
import json
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tasks.training.train_parameters import (
    batch_size,
    num_epochs,
    learning_rate_classification,
    early_stopping_patience,
    LCE_THRESHOLD_BINARY,
    convert_lce_to_binary_class,
    N_ENSEMBLE,
    N_OUTER_FOLDS,
    setup_optimizer,
    setup_scheduler,
    get_combined_stratification_groups,
)
from modules.helena_ce.helena_ce_model import HELENA_CE
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE
from tasks.training.train_loss_function import get_classification_loss
from core.utils import (
    load_dataset,
    load_preprocessed_data,
    get_running_time,
    get_unique_results_dir,
    feature_engineering,
)
from modules.dataset_processing import CollateFn
from modules.tokenizer.tokenizer_helena import Tokenizer
from core.smiles_preprocessing import smiles_dict_preprocessing
from config.datasets import get_ce_dataset, MAX_MOLECULES

MASTER_SEED = 142
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def majority_vote(predictions_list):
    """Hard majority voting across ensemble predictions.

    Args:
        predictions_list: List of prediction arrays, one per model

    Returns:
        voted: Array of final predicted classes after voting
        vote_counts: Array of shape (n_samples, 2) with vote counts per class
    """
    stacked = np.array(predictions_list)
    voted = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=2).argmax(), axis=0, arr=stacked
    )
    vote_counts = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=2), axis=0, arr=stacked
    )
    return voted, vote_counts.T


def calculate_classification_metrics(predictions, truths):
    return {
        "accuracy": accuracy_score(truths, predictions),
        "f1": f1_score(truths, predictions, average="weighted"),
        "precision": precision_score(
            truths, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            truths, predictions, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truths, predictions),
    }


def train_single_model_binary(train_data, map_dict, vocab_size, device, seed, desc=""):
    """Train a single binary classification model with train/val split."""
    set_seed(seed)

    train_indices, val_indices = train_test_split(
        np.arange(len(train_data)), test_size=VAL_SPLIT, random_state=seed
    )
    inner_train_data = train_data.iloc[train_indices].reset_index(drop=True)
    inner_val_data = train_data.iloc[val_indices].reset_index(drop=True)

    train_dataset = feature_engineering(
        map_dict, 0.05, inner_train_data, label_field="LCE"
    )[0]
    val_dataset = feature_engineering(map_dict, 0.0, inner_val_data, label_field="LCE")[
        0
    ]

    collate_fn = CollateFn(MAX_MOLECULES)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = HELENA_CE(
        input_vocab_size=vocab_size,
        max_num_molecules=MAX_MOLECULES,
        task_type="classification",
        num_classes=2,
        **ARCHITECTURE,
    ).to(device)

    optimizer = setup_optimizer(model, lr=learning_rate_classification)
    warmup_scheduler, plateau_scheduler, warmup_epochs = setup_scheduler(
        optimizer, max_lr=learning_rate_classification
    )
    criterion = get_classification_loss()

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_f1_scores": [],
        "val_f1_scores": [],
        "learning_rates": [],
    }
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = model.state_dict()
    best_epoch = 0

    pbar = tqdm(range(1, num_epochs + 1), desc=desc, leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        n_batches = 0
        train_preds_epoch = []
        train_targets_epoch = []

        for batch in train_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)

            lce_values = batch["y"].cpu().numpy()
            class_labels = convert_lce_to_binary_class(
                lce_values, threshold=LCE_THRESHOLD_BINARY
            )
            class_labels_tensor = torch.tensor(class_labels, dtype=torch.long).to(
                device
            )

            loss = criterion(outputs["ce"], class_labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(outputs["ce"], dim=1).cpu().numpy()
            train_preds_epoch.extend(preds)
            train_targets_epoch.extend(class_labels)

        train_loss = epoch_loss / n_batches
        train_f1 = f1_score(train_targets_epoch, train_preds_epoch, average="weighted")

        model.eval()
        val_loss = 0
        val_batches = 0
        val_preds_epoch = []
        val_targets_epoch = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                outputs = model(batch)

                lce_values = batch["y"].cpu().numpy()
                class_labels = convert_lce_to_binary_class(
                    lce_values, threshold=LCE_THRESHOLD_BINARY
                )
                class_labels_tensor = torch.tensor(class_labels, dtype=torch.long).to(
                    device
                )

                loss = criterion(outputs["ce"], class_labels_tensor)
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(outputs["ce"], dim=1).cpu().numpy()
                val_preds_epoch.extend(preds)
                val_targets_epoch.extend(class_labels)

        val_loss = val_loss / max(val_batches, 1)
        val_f1 = f1_score(val_targets_epoch, val_preds_epoch, average="weighted")

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_f1_scores"].append(train_f1)
        history["val_f1_scores"].append(val_f1)
        history["learning_rates"].append(current_lr)

        pbar.set_postfix(
            train_f1=f"{train_f1:.3f}", val_f1=f"{val_f1:.3f}", lr=f"{current_lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_preds, train_targets, train_row_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)
            preds = torch.argmax(outputs["ce"], dim=1).cpu().numpy()
            train_preds.extend(preds)

            lce_values = batch["y"].cpu().numpy()
            class_labels = convert_lce_to_binary_class(
                lce_values, threshold=LCE_THRESHOLD_BINARY
            )
            train_targets.extend(class_labels)
            train_row_ids.extend(batch["row_id"].cpu().numpy().flatten())

    val_preds, val_targets, val_row_ids = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)
            preds = torch.argmax(outputs["ce"], dim=1).cpu().numpy()
            val_preds.extend(preds)

            lce_values = batch["y"].cpu().numpy()
            class_labels = convert_lce_to_binary_class(
                lce_values, threshold=LCE_THRESHOLD_BINARY
            )
            val_targets.extend(class_labels)
            val_row_ids.extend(batch["row_id"].cpu().numpy().flatten())

    train_preds, train_targets = np.array(train_preds), np.array(train_targets)
    train_row_ids = np.array(train_row_ids)
    val_preds, val_targets = np.array(val_preds), np.array(val_targets)
    val_row_ids = np.array(val_row_ids)

    metrics = {
        "train_accuracy": accuracy_score(train_targets, train_preds),
        "train_f1": f1_score(train_targets, train_preds, average="weighted"),
        "val_accuracy": accuracy_score(val_targets, val_preds),
        "val_f1": f1_score(val_targets, val_preds, average="weighted"),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }

    return (
        model,
        best_state,
        history,
        metrics,
        train_preds,
        train_targets,
        train_row_ids,
        val_preds,
        val_targets,
        val_row_ids,
    )


def run_inference_binary(model, data_loader, device):
    """Run inference and return predictions, targets, and row_ids."""
    model.eval()
    predictions = []
    targets = []
    row_ids = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)

            preds = torch.argmax(outputs["ce"], dim=1).cpu().numpy()
            predictions.extend(preds)

            lce_vals = batch["y"].cpu().numpy()
            class_labels = convert_lce_to_binary_class(
                lce_vals, threshold=LCE_THRESHOLD_BINARY
            )
            targets.extend(class_labels)
            row_ids.extend(batch["row_id"].cpu().numpy().flatten())

    return np.array(predictions), np.array(targets), np.array(row_ids)


def train_outer_fold_worker_binary(
    outer_idx,
    gpu_id,
    train_data,
    test_data,
    map_dict,
    vocab_size,
    moiety_key_map,
    key_moiety_map,
    weights_dir,
    result_queue,
):
    """Worker function to train all ensemble models for one outer fold on a specific GPU."""
    try:
        os.makedirs(weights_dir, exist_ok=True)

        device = torch.device(f"cuda:{gpu_id}")
        print(f"[GPU {gpu_id}] Starting Outer Fold {outer_idx + 1}")

        test_dataset = feature_engineering(map_dict, 0.0, test_data, label_field="LCE")[
            0
        ]
        collate_fn = CollateFn(MAX_MOLECULES)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        fold_result = {
            "outer_idx": outer_idx,
            "inner_fold_records": [],
            "ensemble_histories": [],
            "inner_confusion_data": [],
            "ensemble_preds": [],
            "test_targets": None,
            "row_ids": None,
            "all_loss_records": [],
            "all_lr_records": [],
            "all_prediction_records": [],
        }

        for m in range(N_ENSEMBLE):
            seed = MASTER_SEED + outer_idx * 100 + m
            desc = f"GPU{gpu_id} Fold{outer_idx + 1} M{m + 1}"

            (
                model,
                best_state,
                history,
                model_metrics,
                train_preds,
                train_targets,
                train_row_ids,
                val_preds,
                val_targets,
                val_row_ids,
            ) = train_single_model_binary(
                train_data, map_dict, vocab_size, device, seed, desc
            )

            print(
                f"  [GPU {gpu_id}] Fold {outer_idx + 1} Model {m + 1}: "
                f"Train F1={model_metrics['train_f1']:.3f} | Val F1={model_metrics['val_f1']:.3f} | "
                f"Best epoch: {model_metrics['best_epoch']}"
            )

            fold_result["inner_fold_records"].append(
                {
                    "outer_fold": outer_idx + 1,
                    "model_idx": m + 1,
                    "seed": seed,
                    "train_accuracy": model_metrics["train_accuracy"],
                    "train_f1": model_metrics["train_f1"],
                    "val_accuracy": model_metrics["val_accuracy"],
                    "val_f1": model_metrics["val_f1"],
                    "best_epoch": model_metrics["best_epoch"],
                    "best_val_f1": model_metrics["best_val_f1"],
                }
            )

            fold_result["ensemble_histories"].append(history)
            fold_result["inner_confusion_data"].append(
                {
                    "train_preds": train_preds,
                    "train_targets": train_targets,
                    "train_row_ids": train_row_ids,
                    "val_preds": val_preds,
                    "val_targets": val_targets,
                    "val_row_ids": val_row_ids,
                }
            )

            checkpoint = {
                "model_state_dict": best_state,
                "moiety_key_map": moiety_key_map,
                "key_moiety_map": key_moiety_map,
                "vocab_size": vocab_size,
                "task_type": "binary",
                "outer_fold": outer_idx + 1,
                "model_idx": m + 1,
                "seed": seed,
            }
            torch.save(
                checkpoint,
                os.path.join(weights_dir, f"outer_{outer_idx + 1}_model_{m + 1}.pth"),
            )

            preds, targets, row_ids = run_inference_binary(model, test_loader, device)
            fold_result["ensemble_preds"].append(preds)
            fold_result["test_targets"] = targets
            fold_result["row_ids"] = row_ids

            for epoch_idx, (train_loss, val_loss, lr) in enumerate(
                zip(
                    history["train_losses"],
                    history["val_losses"],
                    history["learning_rates"],
                )
            ):
                fold_result["all_loss_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "epoch": epoch_idx + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                fold_result["all_lr_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "epoch": epoch_idx + 1,
                        "learning_rate": lr,
                    }
                )

            for rid, target, pred in zip(train_row_ids, train_targets, train_preds):
                fold_result["all_prediction_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "split": "train",
                        "row_id": rid,
                        "prediction": pred,
                        "target": target,
                    }
                )
            for rid, target, pred in zip(val_row_ids, val_targets, val_preds):
                fold_result["all_prediction_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "split": "val",
                        "row_id": rid,
                        "prediction": pred,
                        "target": target,
                    }
                )
            for rid, target, pred in zip(row_ids, targets, preds):
                fold_result["all_prediction_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "split": "test",
                        "row_id": rid,
                        "prediction": pred,
                        "target": target,
                    }
                )

        result_queue.put(fold_result)
        print(f"[GPU {gpu_id}] Completed Outer Fold {outer_idx + 1}")

    except Exception as e:
        print(f"Error in worker GPU{gpu_id} Fold{outer_idx + 1}: {e}")
        import traceback

        traceback.print_exc()
        result_queue.put(None)




def plot_confusion_matrix(ax, cm, title, labels=None):
    """Plot a single confusion matrix on the given axis."""
    if labels is None:
        labels = [f"< {LCE_THRESHOLD_BINARY}", f">= {LCE_THRESHOLD_BINARY}"]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)

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


def plot_inner_confusion_mosaic(inner_data, outer_idx, results_dir, split="val"):
    """Plot confusion matrices for all models in one outer fold."""
    n_models = len(inner_data)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, data in enumerate(inner_data):
        ax = axes[m]
        if split == "val":
            cm = confusion_matrix(data["val_targets"], data["val_preds"])
        else:
            cm = confusion_matrix(data["train_targets"], data["train_preds"])
        title = f"Model {m + 1}"
        plot_confusion_matrix(ax, cm, title)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - {split.capitalize()} Confusion Matrices",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_{split}_confusion_mosaic.png"),
        dpi=300,
    )
    plt.close()


def plot_test_confusion_mosaic(test_preds_list, test_targets, outer_idx, results_dir):
    """Plot test set confusion matrices for all models in one outer fold."""
    n_models = len(test_preds_list)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, preds in enumerate(test_preds_list):
        ax = axes[m]
        cm = confusion_matrix(test_targets, preds)
        plot_confusion_matrix(ax, cm, f"Model {m + 1}")

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Test Confusion Matrices",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_test_confusion_mosaic.png"),
        dpi=300,
    )
    plt.close()


def smooth_curve(values, weight=0.6):
    """Exponential moving average smoothing for visualization."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_inner_loss_mosaic(inner_histories, outer_idx, results_dir):
    """Plot loss curves for all models in one outer fold."""
    n_models = len(inner_histories)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, history in enumerate(inner_histories):
        ax = axes[m]
        epochs = list(range(1, len(history["train_losses"]) + 1))
        train_raw = history["train_losses"]
        val_raw = history["val_losses"]
        train_smooth = smooth_curve(train_raw)
        val_smooth = smooth_curve(val_raw)
        ax.plot(epochs, train_raw, color="orange", alpha=0.3, linewidth=1)
        ax.plot(epochs, val_raw, color="blue", alpha=0.3, linewidth=1)
        ax.plot(epochs, train_smooth, color="orange", label="Train", linewidth=2)
        ax.plot(epochs, val_smooth, color="blue", label="Val", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Model {m + 1}", fontsize=14, fontweight="bold")
        ax.legend(frameon=False, fontsize=10)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Loss Curves", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_loss_mosaic.png"), dpi=300
    )
    plt.close()


def plot_inner_f1_mosaic(inner_histories, outer_idx, results_dir):
    """Plot F1 score curves for all models in one outer fold."""
    n_models = len(inner_histories)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, history in enumerate(inner_histories):
        ax = axes[m]
        train_raw = [0] + history["train_f1_scores"]
        val_raw = [0] + history["val_f1_scores"]
        train_smooth = smooth_curve(train_raw)
        val_smooth = smooth_curve(val_raw)
        epochs = list(range(len(train_raw)))
        ax.plot(epochs, train_raw, color="orange", alpha=0.3, linewidth=1)
        ax.plot(epochs, val_raw, color="blue", alpha=0.3, linewidth=1)
        ax.plot(epochs, train_smooth, color="orange", label="Train", linewidth=2)
        ax.plot(epochs, val_smooth, color="blue", label="Val", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(f"Model {m + 1}", fontsize=14, fontweight="bold")
        ax.legend(frameon=False, fontsize=10, loc="lower right")

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - F1 Score Curves", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_f1_mosaic.png"), dpi=300
    )
    plt.close()


def plot_inner_lr_mosaic(inner_histories, outer_idx, results_dir):
    """Plot learning rate curves for all models in one outer fold."""
    n_models = len(inner_histories)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, history in enumerate(inner_histories):
        ax = axes[m]
        epochs = range(1, len(history["learning_rates"]) + 1)
        ax.plot(epochs, history["learning_rates"], color="green", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title(f"Model {m + 1}", fontsize=14, fontweight="bold")
        ax.set_yscale("log")

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Learning Rate", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_lr_mosaic.png"), dpi=300
    )
    plt.close()


def plot_final_confusion_matrix(targets, predictions, metrics, path):
    """Plot final confusion matrix with metrics."""
    fig, ax = plt.subplots(figsize=(8, 8))
    cm = confusion_matrix(targets, predictions)
    plot_confusion_matrix(ax, cm, "")

    plt.title(
        f"Accuracy = {metrics['accuracy']:.3f}, F1 = {metrics['f1']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()




def main():
    mp.set_start_method("spawn", force=True)

    set_seed(MASTER_SEED)
    start_time = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    if n_gpus == 0:
        raise RuntimeError("This training entry point requires at least one CUDA GPU.")

    results_dir = get_unique_results_dir("Train_Ensemble_Binary")
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "training.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"Results: {results_dir}")
    print(f"Binary Classification: Class 0 (LCE < {LCE_THRESHOLD_BINARY}), Class 1 (LCE >= {LCE_THRESHOLD_BINARY})")

    ce_data_path = get_ce_dataset()

    process_id = os.getpid()
    preprocessed_path = smiles_dict_preprocessing(ce_data_path, process_id=process_id)
    map_dict, vocab_size, moiety_key_map, key_moiety_map = load_preprocessed_data(
        preprocessed_path
    )

    tokenizer = Tokenizer()
    ce_data = load_dataset(ce_data_path)
    ce_data = ce_data.sample(frac=1, random_state=MASTER_SEED).reset_index(drop=True)
    strat_groups = get_combined_stratification_groups(ce_data, tokenizer)
    print(f"Dataset: {len(ce_data)} samples")

    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=False)
    folds = list(outer_cv.split(ce_data, strat_groups))

    all_fold_results = [None] * N_OUTER_FOLDS

    print(f"\n{'=' * 60}")
    print(
        f"Processing all {N_OUTER_FOLDS} Outer Folds in parallel across {n_gpus} GPUs"
    )
    print(f"Ensemble size: {N_ENSEMBLE} models per fold")
    print(f"{'=' * 60}")

    result_queue = mp.Queue()
    processes = []

    for outer_idx in range(N_OUTER_FOLDS):
        gpu_id = outer_idx % n_gpus
        train_idx, test_idx = folds[outer_idx]
        train_data = ce_data.iloc[train_idx].reset_index(drop=True)
        test_data = ce_data.iloc[test_idx].reset_index(drop=True)

        print(f"  Fold {outer_idx + 1} -> GPU {gpu_id}")

        p = mp.Process(
            target=train_outer_fold_worker_binary,
            args=(
                outer_idx,
                gpu_id,
                train_data,
                test_data,
                map_dict,
                vocab_size,
                moiety_key_map,
                key_moiety_map,
                weights_dir,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    for _ in range(N_OUTER_FOLDS):
        result = result_queue.get()
        if result is not None:
            all_fold_results[result["outer_idx"]] = result

    for p in processes:
        p.join()

    all_predictions, all_targets, all_row_ids = [], [], []
    all_vote_counts = []
    outer_fold_metrics = []
    inner_fold_records = []
    all_loss_records = []
    all_lr_records = []
    all_prediction_records = []

    for fold_result in all_fold_results:
        if fold_result is None:
            continue

        outer_idx = fold_result["outer_idx"]

        plot_inner_loss_mosaic(
            fold_result["ensemble_histories"], outer_idx, results_dir
        )
        plot_inner_f1_mosaic(fold_result["ensemble_histories"], outer_idx, results_dir)
        plot_inner_lr_mosaic(fold_result["ensemble_histories"], outer_idx, results_dir)
        plot_inner_confusion_mosaic(
            fold_result["inner_confusion_data"], outer_idx, results_dir, split="train"
        )
        plot_inner_confusion_mosaic(
            fold_result["inner_confusion_data"], outer_idx, results_dir, split="val"
        )
        plot_test_confusion_mosaic(
            fold_result["ensemble_preds"],
            fold_result["test_targets"],
            outer_idx,
            results_dir,
        )

        ensemble_preds = fold_result["ensemble_preds"]
        voted_preds, vote_counts = majority_vote(ensemble_preds)
        targets = fold_result["test_targets"]

        fold_metrics = calculate_classification_metrics(voted_preds, targets)
        print(
            f"Outer Fold {outer_idx + 1} Test (Ensemble): "
            f"Acc = {fold_metrics['accuracy']:.3f}, F1 = {fold_metrics['f1']:.3f}"
        )

        outer_fold_metrics.append(
            {
                "outer_fold": outer_idx + 1,
                "n_test": len(targets),
                "accuracy": fold_metrics["accuracy"],
                "f1": fold_metrics["f1"],
                "precision": fold_metrics["precision"],
                "recall": fold_metrics["recall"],
            }
        )

        all_predictions.extend(voted_preds)
        all_targets.extend(targets)
        all_row_ids.extend(fold_result["row_ids"])
        all_vote_counts.extend(vote_counts)

        inner_fold_records.extend(fold_result["inner_fold_records"])
        all_loss_records.extend(fold_result["all_loss_records"])
        all_lr_records.extend(fold_result["all_lr_records"])
        all_prediction_records.extend(fold_result["all_prediction_records"])

    prediction_df = pd.DataFrame(all_prediction_records)
    test_df = prediction_df[prediction_df["split"] == "test"]

    def vote_group(group):
        preds = group["prediction"].values
        voted = np.bincount(preds.astype(int), minlength=2).argmax()
        vote_counts = np.bincount(preds.astype(int), minlength=2)
        return pd.Series(
            {
                "target": group["target"].iloc[0],
                "prediction": voted,
                "n_preds": len(preds),
                "votes_class_0": vote_counts[0],
                "votes_class_1": vote_counts[1],
            }
        )

    test_ensemble = test_df.groupby("row_id").apply(vote_group).reset_index()

    final_predictions = test_ensemble["prediction"].values
    final_targets = test_ensemble["target"].values
    final_row_ids = test_ensemble["row_id"].values

    final_metrics = calculate_classification_metrics(final_predictions, final_targets)

    print(f"\n{'=' * 60}\nFINAL RESULTS (Ensemble with Majority Voting)\n{'=' * 60}")
    print(f"Accuracy:    {final_metrics['accuracy']:.3f}")
    print(f"F1:          {final_metrics['f1']:.3f}")
    print(f"Precision:   {final_metrics['precision']:.3f}")
    print(f"Recall:      {final_metrics['recall']:.3f}")
    print(f"\nConfusion Matrix:")
    print(final_metrics["confusion_matrix"])

    pd.DataFrame(inner_fold_records).to_csv(
        os.path.join(results_dir, "inner_fold_results.csv"), index=False, float_format="%.3f"
    )

    predictions_df = pd.DataFrame(
        {
            "row_id": final_row_ids,
            "actual_class": final_targets,
            "predicted_class": final_predictions,
            "n_predictions": test_ensemble["n_preds"].values,
            "votes_class_0": test_ensemble["votes_class_0"].values,
            "votes_class_1": test_ensemble["votes_class_1"].values,
        }
    )
    predictions_df.to_csv(
        os.path.join(results_dir, "nested_cv_predictions.csv"), index=False
    )

    metrics_dict = {
        "accuracy": round(float(final_metrics["accuracy"]), 3),
        "f1": round(float(final_metrics["f1"]), 3),
        "precision": round(float(final_metrics["precision"]), 3),
        "recall": round(float(final_metrics["recall"]), 3),
        "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
        "avg_preds_per_sample": round(float(test_ensemble["n_preds"].mean()), 1),
        "n_samples": len(final_targets),
    }
    with open(os.path.join(results_dir, "nested_cv_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    pd.DataFrame(outer_fold_metrics).to_csv(
        os.path.join(results_dir, "outer_fold_metrics.csv"), index=False, float_format="%.3f"
    )
    pd.DataFrame(all_loss_records).to_csv(
        os.path.join(results_dir, "all_loss_data.csv"), index=False
    )
    pd.DataFrame(all_lr_records).to_csv(
        os.path.join(results_dir, "all_lr_data.csv"), index=False
    )
    pd.DataFrame(all_prediction_records).to_csv(
        os.path.join(results_dir, "all_prediction_data.csv"), index=False
    )

    plot_final_confusion_matrix(
        final_targets,
        final_predictions,
        final_metrics,
        os.path.join(results_dir, "nested_cv_final_confusion.png"),
    )

    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    get_running_time(start_time)
    print(f"\nResults saved to: {results_dir}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
