"""Train atom and drug tokenizer ensembles with nested cross-validation.

Before training, stratify rows by current-density bins and rare tokens from each
study's tokenizer; keep the resulting folds fixed during training.
Use --study atoms, --study moiety_drug, or --all-studies (default: moiety_drug)."""

import argparse
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
from sklearn.metrics import r2_score

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from tasks.training.train_parameters import (
    batch_size,
    num_epochs,
    learning_rate,
    early_stopping_patience,
    N_ENSEMBLE,
    N_OUTER_FOLDS,
    setup_optimizer,
    setup_scheduler,
)
from modules.helena_ce.helena_ce_model import HELENA_CE
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE
from tasks.training.train_loss_function import get_regression_loss
from core.utils import (
    load_dataset,
    get_running_time,
    get_unique_results_dir,
    feature_engineering,
)
from modules.dataset_processing import CollateFn
from modules.moiety_features import (
    map_atom_to_moiety,
    build_moiety_edges,
    get_adj_matrix,
    get_dist_matrix,
)
from config.datasets import get_ce_dataset, MAX_MOLECULES, MAX_SOLVENTS, MAX_SALTS

from tasks.postprocess.tokenizer_studies.tokenizer_atoms import TokenizerAtoms
from modules.tokenizer.tokenizer_helena import Tokenizer as TokenizerMoiety
from tasks.postprocess.tokenizer_studies.tokenizer_moiety_dict_drug import (
    MOIETIES as MOIETIES_DRUG,
)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MASTER_SEED = 142
VAL_SPLIT = 0.1

ALL_STUDIES = [
    {
        "name": "atoms",
        "tokenizer_class": TokenizerAtoms,
        "tokenizer_kwargs": {},
        "description": "Atoms-only tokenization",
        "use_parallel_preprocessing": False,
    },
    {
        "name": "moiety_drug",
        "tokenizer_class": TokenizerMoiety,
        "tokenizer_kwargs": {"moiety_dict": MOIETIES_DRUG},
        "description": "Drug moiety dictionary",
        "use_parallel_preprocessing": True,
    },
]
STUDIES = [study for study in ALL_STUDIES if study["name"] == "moiety_drug"]


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


def _preprocess_single_smiles(args):
    """Worker function for parallel preprocessing."""
    smiles, tokenizer_class, tokenizer_kwargs = args

    tokenizer = tokenizer_class(**tokenizer_kwargs)

    try:
        result = tokenizer.get_moieties(smiles)
        clique_atom_ids, edges, moiety_ids, matches, is_in_ring = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
        )

        atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
        moiety_edges = build_moiety_edges(edges, atom_to_moiety)

        node_list = list(range(len(clique_atom_ids)))
        moiety_edges = list(moiety_edges)

        dist_matrix = get_dist_matrix(node_list, moiety_edges)
        adj_matrix = get_adj_matrix(node_list, moiety_edges)

        return smiles, {
            "adj_matrix": adj_matrix,
            "dist_matrix": dist_matrix,
            "clique_atom_ids": clique_atom_ids,
            "edges": edges,
            "moiety_ids": moiety_ids,
            "is_in_ring": is_in_ring,
        }
    except Exception as e:
        return smiles, {
            "adj_matrix": np.array([[]], dtype=np.float32),
            "dist_matrix": np.array([[]], dtype=np.float32),
            "clique_atom_ids": [],
            "edges": [],
            "moiety_ids": [],
            "is_in_ring": [],
            "error": str(e),
        }


def preprocess_with_tokenizer(
    data_path,
    tokenizer,
    study_name,
    tokenizer_class=None,
    tokenizer_kwargs=None,
    use_parallel=False,
):
    """Preprocess molecules using the specified tokenizer."""
    print(f"Preprocessing molecules for study: {study_name}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")

    max_num_solvent_molecules = MAX_SOLVENTS
    max_num_salt_molecules = MAX_SALTS

    dtypes = {}
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str
    dtypes["Row_ID"] = int
    dtypes["Method"] = str
    dtypes["CE"] = float
    dtypes["LCE"] = float

    import csv

    dataset = pd.read_csv(
        data_path,
        header=0,
        dtype=dtypes,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    solvent_columns = [f"Solvent_{i}" for i in range(1, max_num_solvent_molecules + 1)]
    salt_columns = [f"Salt_{i}" for i in range(1, max_num_salt_molecules + 1)]

    unique_SMILES_in_dataset = []

    for mol_col in solvent_columns + salt_columns:
        if mol_col in dataset.columns:
            for value in dataset[mol_col].dropna():
                if pd.notna(value) and value != "":
                    try:
                        parts = str(value).split(":")
                        if len(parts) == 2:
                            smiles = parts[0].strip()
                            if smiles:
                                unique_SMILES_in_dataset.append(smiles)
                    except (ValueError, AttributeError):
                        pass

    unique_SMILES_in_toy_dataset = sorted(set(unique_SMILES_in_dataset))
    print(f"Total unique SMILES to preprocess: {len(unique_SMILES_in_toy_dataset)}")

    all_mols_dict = {}

    if use_parallel and tokenizer_class is not None and tokenizer_kwargs is not None:
        import multiprocessing as mp_cpu

        n_workers = mp_cpu.cpu_count()
        print(f"Using parallel preprocessing with {n_workers} CPU cores")

        args_list = [
            (smi, tokenizer_class, tokenizer_kwargs)
            for smi in unique_SMILES_in_toy_dataset
        ]

        with mp_cpu.Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_preprocess_single_smiles, args_list),
                    total=len(args_list),
                    desc="Preprocessing (parallel)",
                )
            )

        for smiles, data in results:
            if "error" in data:
                print(f"Error processing mol {smiles}: {data['error']}")
                del data["error"]
            all_mols_dict[smiles] = data

    else:
        for smiles in tqdm(unique_SMILES_in_toy_dataset, desc="Preprocessing"):
            try:
                result = tokenizer.get_moieties(smiles)
                clique_atom_ids, edges, moiety_ids, matches, is_in_ring = (
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[4],
                )

                atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
                moiety_edges = build_moiety_edges(edges, atom_to_moiety)

                node_list = list(range(len(clique_atom_ids)))
                moiety_edges = list(moiety_edges)

                dist_matrix = get_dist_matrix(node_list, moiety_edges)
                adj_matrix = get_adj_matrix(node_list, moiety_edges)

                all_mols_dict[smiles] = {
                    "adj_matrix": adj_matrix,
                    "dist_matrix": dist_matrix,
                    "clique_atom_ids": clique_atom_ids,
                    "edges": edges,
                    "moiety_ids": moiety_ids,
                    "is_in_ring": is_in_ring,
                }

            except Exception as e:
                print(f"Error processing mol {smiles}: {e}")
                all_mols_dict[smiles] = {
                    "adj_matrix": np.array([[]], dtype=np.float32),
                    "dist_matrix": np.array([[]], dtype=np.float32),
                    "clique_atom_ids": [],
                    "edges": [],
                    "moiety_ids": [],
                    "is_in_ring": [],
                }

    vocab_size = tokenizer.get_vocab_size()
    moiety_key_map = tokenizer.moiety_key_map
    key_moiety_map = tokenizer.key_moiety_map

    print(f"Vocabulary size: {vocab_size}")
    print("+" * 60)
    print("Molecules preprocessing COMPLETE!")
    print("+" * 60, "\n")

    return all_mols_dict, vocab_size, moiety_key_map, key_moiety_map


def get_stratification_groups_fast(df, map_dict, key_moiety_map):
    """
    Fast stratification using pre-computed map_dict instead of re-tokenizing.
    Stratify by current density bins and rare moiety presence.
    """
    from config.datasets import SOLVENT_COLUMNS, SALT_COLUMNS
    from collections import Counter

    RARE_MOIETY_THRESHOLD = 0.05

    sample_moieties = {}
    for idx, row in df.iterrows():
        moieties = set()
        for col in SOLVENT_COLUMNS + SALT_COLUMNS:
            if col in df.columns and pd.notna(row[col]):
                value = str(row[col])
                if ":" in value:
                    smiles = value.split(":")[0].strip()
                    if smiles and smiles in map_dict:
                        moiety_ids = map_dict[smiles].get("moiety_ids", [])
                        for mid in moiety_ids:
                            name = key_moiety_map.get(mid, f"Unk_{mid}")
                            moieties.add(name)
        sample_moieties[idx] = moieties

    moiety_counts = Counter()
    for moieties in sample_moieties.values():
        moiety_counts.update(moieties)
    rare = {m for m, c in moiety_counts.items() if c / len(df) < RARE_MOIETY_THRESHOLD}

    print(
        f"Stratification: {len(rare)} rare moieties (<{RARE_MOIETY_THRESHOLD * 100:.0f}%)"
    )

    groups = np.zeros(len(df), dtype=np.int64)
    for idx, row in df.iterrows():
        current = row["Current"]
        current_bin = 0 if current < 0.5 else (1 if current < 1.0 else 2)
        has_rare = int(bool(sample_moieties[idx] & rare))
        groups[idx] = current_bin * 2 + has_rare

    labels = [
        "Low+common",
        "Low+rare",
        "Med+common",
        "Med+rare",
        "High+common",
        "High+rare",
    ]
    print("Groups: " + " ".join(f"{labels[g]}={np.sum(groups == g)}" for g in range(6)))

    return groups


def train_single_model(train_data, map_dict, vocab_size, device, seed, desc=""):
    """Train a single model with train/val split, returning model and training history."""
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
        task_type="regression",
        num_classes=None,
        **ARCHITECTURE,
    ).to(device)

    with torch.no_grad():
        model.ce_head[-1].bias.fill_(inner_train_data["LCE"].mean())

    optimizer = setup_optimizer(model, lr=learning_rate)
    warmup_scheduler, plateau_scheduler, warmup_epochs = setup_scheduler(
        optimizer, max_lr=learning_rate
    )
    criterion = get_regression_loss()

    history = {"train_losses": [], "val_losses": [], "learning_rates": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = model.state_dict()
    best_epoch = 0

    pbar = tqdm(range(1, num_epochs + 1), desc=desc, leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)
            loss = criterion(outputs["ce"].squeeze(), batch["y"].squeeze())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                outputs = model(batch)
                loss = criterion(outputs["ce"].squeeze(), batch["y"].squeeze())
                val_loss += loss.item()
                val_batches += 1
        val_loss = val_loss / max(val_batches, 1)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["learning_rates"].append(current_lr)

        pbar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            train_preds.extend(outputs["ce"].cpu().numpy().flatten())
            train_targets.extend(batch["y"].cpu().numpy().flatten())
            train_row_ids.extend(batch["row_id"].cpu().numpy().flatten())

    val_preds, val_targets, val_row_ids = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)
            val_preds.extend(outputs["ce"].cpu().numpy().flatten())
            val_targets.extend(batch["y"].cpu().numpy().flatten())
            val_row_ids.extend(batch["row_id"].cpu().numpy().flatten())

    train_preds, train_targets = np.array(train_preds), np.array(train_targets)
    train_row_ids = np.array(train_row_ids)
    val_preds, val_targets = np.array(val_preds), np.array(val_targets)
    val_row_ids = np.array(val_row_ids)

    metrics = {
        "train_r2": r2_score(train_targets, train_preds),
        "train_rmse": np.sqrt(np.mean((train_targets - train_preds) ** 2)),
        "val_r2": r2_score(val_targets, val_preds),
        "val_rmse": np.sqrt(np.mean((val_targets - val_preds) ** 2)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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


def run_inference(model, data_loader, device):
    model.eval()
    preds, targets, row_ids = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)
            preds.extend(outputs["ce"].cpu().numpy().flatten())
            targets.extend(batch["y"].cpu().numpy().flatten())
            row_ids.extend(batch["row_id"].cpu().numpy().flatten())

    return np.array(preds), np.array(targets), np.array(row_ids)


def train_outer_fold_worker(
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
            "inner_parity_data": [],
            "ensemble_preds": [],
            "test_targets": None,
            "row_ids": None,
            "all_loss_records": [],
            "all_lr_records": [],
            "all_parity_records": [],
            "individual_model_test_metrics": [],
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
            ) = train_single_model(train_data, map_dict, vocab_size, device, seed, desc)

            print(
                f"  [GPU {gpu_id}] Fold {outer_idx + 1} Model {m + 1}: "
                f"Train R²={model_metrics['train_r2']:.3f} | Val R²={model_metrics['val_r2']:.3f} | "
                f"Best epoch: {model_metrics['best_epoch']}"
            )

            fold_result["inner_fold_records"].append(
                {
                    "outer_fold": outer_idx + 1,
                    "model_idx": m + 1,
                    "seed": seed,
                    "train_r2": model_metrics["train_r2"],
                    "train_rmse": model_metrics["train_rmse"],
                    "val_r2": model_metrics["val_r2"],
                    "val_rmse": model_metrics["val_rmse"],
                    "best_epoch": model_metrics["best_epoch"],
                    "best_val_loss": model_metrics["best_val_loss"],
                }
            )

            fold_result["ensemble_histories"].append(history)
            fold_result["inner_parity_data"].append(
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
                "outer_fold": outer_idx + 1,
                "model_idx": m + 1,
                "seed": seed,
            }
            torch.save(
                checkpoint,
                os.path.join(weights_dir, f"outer_{outer_idx + 1}_model_{m + 1}.pth"),
            )

            preds, targets, row_ids = run_inference(model, test_loader, device)
            fold_result["ensemble_preds"].append(preds)
            fold_result["test_targets"] = targets
            fold_result["row_ids"] = row_ids

            model_test_r2 = r2_score(targets, preds)
            model_test_rmse = np.sqrt(np.mean((targets - preds) ** 2))
            fold_result["individual_model_test_metrics"].append(
                {
                    "outer_fold": outer_idx + 1,
                    "model_idx_in_fold": m + 1,
                    "test_r2": model_test_r2,
                    "test_rmse": model_test_rmse,
                }
            )

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

            for i, (pred, target, rid) in enumerate(
                zip(train_preds, train_targets, train_row_ids)
            ):
                fold_result["all_parity_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "split": "train",
                        "row_id": rid,
                        "prediction": pred,
                        "target": target,
                    }
                )
            for i, (pred, target, rid) in enumerate(
                zip(val_preds, val_targets, val_row_ids)
            ):
                fold_result["all_parity_records"].append(
                    {
                        "outer_fold": outer_idx + 1,
                        "model_idx": m + 1,
                        "split": "val",
                        "row_id": rid,
                        "prediction": pred,
                        "target": target,
                    }
                )
            for i, (pred, target, rid) in enumerate(zip(preds, targets, row_ids)):
                fold_result["all_parity_records"].append(
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


def plot_parity(targets, predictions, metrics, path):
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5, s=50, edgecolors="none", c="green")
    min_val, max_val = (
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    )
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    plt.xlabel("Actual LCE", fontsize=14)
    plt.ylabel("Predicted LCE", fontsize=14)
    plt.title(
        f"R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300)
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
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title(f"Model {m + 1}", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(frameon=False, fontsize=12)

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Loss Curves", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_loss_mosaic.png"), dpi=300
    )
    plt.close()


def plot_inner_parity_mosaic(inner_data, outer_idx, results_dir):
    n_models = len(inner_data)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, data in enumerate(inner_data):
        ax = axes[m]
        ax.scatter(
            data["train_targets"],
            data["train_preds"],
            color="orange",
            alpha=0.5,
            s=20,
            label="Train",
        )
        ax.scatter(
            data["val_targets"],
            data["val_preds"],
            color="blue",
            alpha=0.6,
            s=25,
            label="Val",
        )
        all_vals = np.concatenate(
            [
                data["train_targets"],
                data["train_preds"],
                data["val_targets"],
                data["val_preds"],
            ]
        )
        ax.plot(
            [all_vals.min(), all_vals.max()],
            [all_vals.min(), all_vals.max()],
            "k--",
            lw=1.5,
        )
        ax.set_title(f"Model {m + 1}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Actual LCE", fontsize=14)
        ax.set_ylabel("Predicted LCE", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(frameon=False, fontsize=12, loc="upper left")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Parity Plots", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_parity_mosaic.png"), dpi=300
    )
    plt.close()


def plot_test_parity_mosaic(test_preds_list, test_targets, outer_idx, results_dir):
    n_models = len(test_preds_list)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for m, preds in enumerate(test_preds_list):
        ax = axes[m]
        ax.scatter(test_targets, preds, color="green", alpha=0.6, s=30)
        all_vals = np.concatenate([test_targets, preds])
        ax.plot(
            [all_vals.min(), all_vals.max()],
            [all_vals.min(), all_vals.max()],
            "k--",
            lw=1.5,
        )
        ax.set_title(f"Model {m + 1}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Actual LCE", fontsize=14)
        ax.set_ylabel("Predicted LCE", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Test Set Predictions",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_test_parity_mosaic.png"),
        dpi=300,
    )
    plt.close()


def plot_inner_lr_mosaic(inner_histories, outer_idx, results_dir):
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
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Learning Rate", fontsize=14)
        ax.set_title(f"Model {m + 1}", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.set_yscale("log")

    plt.suptitle(
        f"Outer Fold {outer_idx + 1} - Learning Rate", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"outer_{outer_idx + 1}_lr_mosaic.png"), dpi=300
    )
    plt.close()


def run_single_study(study_config, ce_data_path, ce_data, base_results_dir, study_idx):
    """Run a single tokenizer study."""
    study_name = study_config["name"]
    study_desc = study_config["description"]

    print(f"\n{'#' * 80}")
    print(f"# STUDY {study_idx + 1}: {study_name.upper()}")
    print(f"# Description: {study_desc}")
    print(f"{'#' * 80}\n")

    study_start_time = time.time()

    study_results_dir = os.path.join(
        base_results_dir, f"study_{study_idx + 1}_{study_name}"
    )
    weights_dir = os.path.join(study_results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    log_path = os.path.join(study_results_dir, "training.log")
    logger = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = logger

    print(f"Study: {study_name}")
    print(f"Description: {study_desc}")
    print(f"Results: {study_results_dir}")

    tokenizer_class = study_config["tokenizer_class"]
    tokenizer_kwargs = study_config["tokenizer_kwargs"]
    use_parallel = study_config.get("use_parallel_preprocessing", False)

    tokenizer = tokenizer_class(**tokenizer_kwargs)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")

    map_dict, vocab_size, moiety_key_map, key_moiety_map = preprocess_with_tokenizer(
        ce_data_path,
        tokenizer,
        study_name,
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        use_parallel=use_parallel,
    )

    strat_groups = get_stratification_groups_fast(ce_data, map_dict, key_moiety_map)
    print(f"Dataset: {len(ce_data)} samples")
    print(f"Vocabulary size: {vocab_size}")

    n_gpus = torch.cuda.device_count()
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=False)
    folds = list(outer_cv.split(ce_data, strat_groups))

    all_fold_results = [None] * N_OUTER_FOLDS

    print(f"\n{'=' * 60}")
    print(
        f"Processing all {N_OUTER_FOLDS} Outer Folds in parallel across {n_gpus} GPUs"
    )
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
            target=train_outer_fold_worker,
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

    all_predictions, all_uncertainties, all_targets, all_row_ids = [], [], [], []
    outer_fold_metrics = []
    inner_fold_records = []
    all_loss_records = []
    all_lr_records = []
    all_parity_records = []
    individual_model_test_metrics = []

    for fold_result in all_fold_results:
        if fold_result is None:
            continue

        outer_idx = fold_result["outer_idx"]

        plot_inner_loss_mosaic(
            fold_result["ensemble_histories"], outer_idx, study_results_dir
        )
        plot_inner_parity_mosaic(
            fold_result["inner_parity_data"], outer_idx, study_results_dir
        )
        plot_inner_lr_mosaic(
            fold_result["ensemble_histories"], outer_idx, study_results_dir
        )
        plot_test_parity_mosaic(
            fold_result["ensemble_preds"],
            fold_result["test_targets"],
            outer_idx,
            study_results_dir,
        )

        ensemble_preds = np.array(fold_result["ensemble_preds"])
        mean_preds = np.mean(ensemble_preds, axis=0)
        std_preds = np.std(ensemble_preds, axis=0)
        targets = fold_result["test_targets"]

        fold_r2 = r2_score(targets, mean_preds)
        fold_rmse = np.sqrt(np.mean((targets - mean_preds) ** 2))
        print(
            f"Outer Fold {outer_idx + 1} Test: R² = {fold_r2:.3f}, RMSE = {fold_rmse:.3f}"
        )

        outer_fold_metrics.append(
            {
                "outer_fold": outer_idx + 1,
                "n_test": len(targets),
                "r2": fold_r2,
                "rmse": fold_rmse,
            }
        )

        all_predictions.extend(mean_preds)
        all_uncertainties.extend(std_preds)
        all_targets.extend(targets)
        all_row_ids.extend(fold_result["row_ids"])

        inner_fold_records.extend(fold_result["inner_fold_records"])
        all_loss_records.extend(fold_result["all_loss_records"])
        all_lr_records.extend(fold_result["all_lr_records"])
        all_parity_records.extend(fold_result["all_parity_records"])
        individual_model_test_metrics.extend(
            fold_result["individual_model_test_metrics"]
        )

    parity_df = pd.DataFrame(all_parity_records)
    parity_test = parity_df[parity_df["split"] == "test"]

    test_ensemble = (
        parity_test.groupby("row_id")
        .agg({"target": "first", "prediction": "mean"})
        .reset_index()
    )
    test_ensemble["n_preds"] = parity_test.groupby("row_id").size().values

    final_predictions = test_ensemble["prediction"].values
    final_targets = test_ensemble["target"].values
    final_row_ids = test_ensemble["row_id"].values

    final_r2 = r2_score(final_targets, final_predictions)
    final_rmse = np.sqrt(np.mean((final_targets - final_predictions) ** 2))
    final_mae = np.mean(np.abs(final_targets - final_predictions))

    study_elapsed_time = time.time() - study_start_time
    study_time_str = get_running_time(study_start_time)

    print(f"\n{'=' * 60}\nFINAL RESULTS (Ensemble) - {study_name}\n{'=' * 60}")
    print(f"R²:   {final_r2:.3f}")
    print(f"RMSE: {final_rmse:.3f}")
    print(f"MAE:  {final_mae:.3f}")
    print(
        f"Training time: {study_elapsed_time:.1f} seconds ({study_elapsed_time / 60:.1f} minutes)"
    )

    pd.DataFrame(inner_fold_records).to_csv(
        os.path.join(study_results_dir, "inner_fold_results.csv"),
        index=False,
        float_format="%.3f",
    )

    predictions_df = pd.DataFrame(
        {
            "row_id": final_row_ids,
            "actual_lce": final_targets,
            "predicted_lce": final_predictions,
            "n_predictions": test_ensemble["n_preds"].values,
            "error": final_targets - final_predictions,
            "abs_error": np.abs(final_targets - final_predictions),
        }
    )
    predictions_df.to_csv(
        os.path.join(study_results_dir, "nested_cv_predictions.csv"),
        index=False,
        float_format="%.3f",
    )

    metrics = {
        "study_name": study_name,
        "description": study_desc,
        "r2": round(float(final_r2), 3),
        "rmse": round(float(final_rmse), 3),
        "mae": round(float(final_mae), 3),
        "avg_preds_per_sample": round(float(test_ensemble["n_preds"].mean()), 1),
        "n_samples": len(final_targets),
        "vocab_size": vocab_size,
        "training_time_seconds": round(study_elapsed_time, 1),
        "training_time_minutes": round(study_elapsed_time / 60, 1),
    }
    with open(os.path.join(study_results_dir, "nested_cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(outer_fold_metrics).to_csv(
        os.path.join(study_results_dir, "outer_fold_metrics.csv"),
        index=False,
        float_format="%.3f",
    )
    pd.DataFrame(all_loss_records).to_csv(
        os.path.join(study_results_dir, "all_loss_data.csv"), index=False
    )
    pd.DataFrame(all_lr_records).to_csv(
        os.path.join(study_results_dir, "all_lr_data.csv"), index=False
    )
    pd.DataFrame(all_parity_records).to_csv(
        os.path.join(study_results_dir, "all_parity_data.csv"),
        index=False,
        float_format="%.3f",
    )
    pd.DataFrame(individual_model_test_metrics).to_csv(
        os.path.join(study_results_dir, "individual_model_test_metrics.csv"),
        index=False,
        float_format="%.3f",
    )

    plot_parity(
        final_targets,
        final_predictions,
        {"r2": final_r2, "rmse": final_rmse},
        os.path.join(study_results_dir, "nested_cv_parity_plot.png"),
    )

    print(f"\nResults saved to: {study_results_dir}")

    sys.stdout = old_stdout
    logger.close()

    return metrics


def main(study_names=None):
    if study_names is None:
        studies = STUDIES
    else:
        available = {study["name"]: study for study in ALL_STUDIES}
        unknown = set(study_names) - available.keys()
        if unknown:
            raise ValueError(f"Unknown tokenizer studies: {sorted(unknown)}")
        studies = [available[name] for name in dict.fromkeys(study_names)]

    mp.set_start_method("spawn", force=True)

    set_seed(MASTER_SEED)
    total_start_time = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    if n_gpus == 0:
        raise RuntimeError("This training entry point requires at least one CUDA GPU.")

    base_results_dir = get_unique_results_dir("Tokenizer_Ablations")
    os.makedirs(base_results_dir, exist_ok=True)

    print(f"\n{'*' * 80}")
    print(f"* TOKENIZER ABLATION STUDIES")
    print(f"* Running {len(studies)} studies sequentially")
    print(f"* Results directory: {base_results_dir}")
    print(f"{'*' * 80}\n")

    ce_data_path = get_ce_dataset()
    ce_data = load_dataset(ce_data_path)
    ce_data = ce_data.sample(frac=1, random_state=MASTER_SEED).reset_index(drop=True)
    print(f"Loaded dataset: {len(ce_data)} samples")

    all_study_results = []
    for study_idx, study_config in enumerate(studies):
        study_metrics = run_single_study(
            study_config, ce_data_path, ce_data, base_results_dir, study_idx
        )
        all_study_results.append(study_metrics)

    total_elapsed_time = time.time() - total_start_time

    summary = {
        "total_studies": len(studies),
        "total_time_seconds": round(total_elapsed_time, 1),
        "total_time_minutes": round(total_elapsed_time / 60, 1),
        "total_time_hours": round(total_elapsed_time / 3600, 2),
        "studies": all_study_results,
    }

    with open(os.path.join(base_results_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'*' * 80}")
    print(f"* ALL TOKENIZER ABLATION STUDIES COMPLETE")
    print(f"{'*' * 80}")
    print(f"\nSummary:")
    print(f"{'=' * 60}")
    for result in all_study_results:
        print(
            f"  {result['study_name']:20s}: R²={result['r2']:.3f}, RMSE={result['rmse']:.3f}, "
            f"Vocab={result['vocab_size']}, Time={result['training_time_minutes']:.1f}min"
        )
    print(f"{'=' * 60}")
    print(f"Total time: {total_elapsed_time / 3600:.2f} hours")
    print(f"\nAll results saved to: {base_results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--study", action="append", choices=[study["name"] for study in ALL_STUDIES],
        help="Select a tokenizer study; repeat to run several in the given order.",
    )
    selection.add_argument("--all-studies", action="store_true", help="Run both tokenizer studies.")
    args = parser.parse_args()
    main([study["name"] for study in ALL_STUDIES] if args.all_studies else args.study)
