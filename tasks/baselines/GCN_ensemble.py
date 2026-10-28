"""
Nested CV + Ensemble Training for GCN Baseline.
Uses same experimental setup as HELENA for fair comparison.

Configuration:
- 5 outer folds (stratified by current density + rare moieties)
- 8 ensemble models per outer fold (40 total)
- Predictions averaged across ensemble
- Outer folds run in parallel across GPUs
"""

import sys
import os
import time
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from Baseline_Utils.GNN_modules import MoleculeEncoder, BaselineFormulationModel
from Baseline_Utils.GNN_regr_utils import (
    GNNFormulationDataset,
    collate_gnn_formulations,
    load_dataset,
    NODE_FEATURES_DIM,
    EDGE_FEATURES_DIM,
    GNN_HIDDEN_DIM,
    GNN_NUM_LAYERS,
    GAT_HEADS,
)
from Baseline_Utils.nested_cv_utils import (
    MASTER_SEED,
    N_OUTER_FOLDS,
    N_ENSEMBLE,
    VAL_SPLIT,
    TeeLogger,
    get_stratification_groups,
    get_results_dir,
    calculate_regression_metrics,
    save_results,
    plot_parity,
    print_final_results,
)

from config.datasets import MAX_SOLVENTS, MAX_SALTS

# GNN-specific settings
GNN_TYPE = "GCN"
NUM_EPOCHS = 600
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
EARLY_STOPPING_PATIENCE = 150  # num_epochs // 4


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_gnn_model(gnn_type, device):
    """Create a GNN model."""
    model = BaselineFormulationModel(
        gnn_type=gnn_type,
        node_features_dim=NODE_FEATURES_DIM,
        edge_features_dim=EDGE_FEATURES_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=1,
        num_layers=GNN_NUM_LAYERS,
        heads=GAT_HEADS,
        max_molecules=MAX_SOLVENTS + MAX_SALTS,
    )
    return model.to(device)


def train_single_gnn_model(
    train_data_df, val_data_df, molecule_encoder, device, seed, desc=""
):
    """Train a single GNN model."""
    set_seed(seed)

    # Create datasets
    train_dataset = GNNFormulationDataset(
        train_data_df,
        molecule_encoder,
        label_field="LCE",
        max_num_solvent_molecules=MAX_SOLVENTS,
        max_num_salt_molecules=MAX_SALTS,
        concentration_noise=0.03,
    )
    val_dataset = GNNFormulationDataset(
        val_data_df,
        molecule_encoder,
        label_field="LCE",
        max_num_solvent_molecules=MAX_SOLVENTS,
        max_num_salt_molecules=MAX_SALTS,
        concentration_noise=0.0,
    )

    # Create dataloaders
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_gnn_formulations,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_gnn_formulations,
    )

    # Create model
    model = create_gnn_model(GNN_TYPE, device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy="cos",
    )
    criterion = torch.nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    history = {"train_losses": [], "val_losses": [], "learning_rates": []}

    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=desc, leave=False)
    for epoch in pbar:
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            labels = batch["labels"].to(device).view(-1, 1)
            concentrations = batch["concentrations"].to(device)
            currents = batch["currents"].to(device)
            methods = batch["methods"].to(device)
            mol_graphs = batch["mol_graphs"]

            optimizer.zero_grad()
            outputs = model(mol_graphs, concentrations, currents, methods)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(device).view(-1, 1)
                concentrations = batch["concentrations"].to(device)
                currents = batch["currents"].to(device)
                methods = batch["methods"].to(device)
                mol_graphs = batch["mol_graphs"]

                outputs = model(mol_graphs, concentrations, currents, methods)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["learning_rates"].append(current_lr)

        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_state, history, best_epoch, best_val_loss


def run_inference(model, data_loader, device):
    """Run inference and return predictions."""
    model.eval()
    predictions = []
    targets = []
    elf_codes = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device).view(-1, 1)
            concentrations = batch["concentrations"].to(device)
            currents = batch["currents"].to(device)
            methods = batch["methods"].to(device)
            mol_graphs = batch["mol_graphs"]
            batch_elf_codes = batch["elf_codes"]

            outputs = model(mol_graphs, concentrations, currents, methods)

            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())
            elf_codes.extend(batch_elf_codes.cpu().numpy())

    return np.array(predictions), np.array(targets), np.array(elf_codes)


def train_outer_fold_worker(
    outer_idx,
    gpu_id,
    train_data_df,
    test_data_df,
    weights_dir,
    result_queue,
):
    """Worker function to train all ensemble models for one outer fold."""
    try:
        os.makedirs(weights_dir, exist_ok=True)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[GPU {gpu_id}] Starting Outer Fold {outer_idx + 1}")

        molecule_encoder = MoleculeEncoder()

        # Create test dataset and loader
        test_dataset = GNNFormulationDataset(
            test_data_df,
            molecule_encoder,
            label_field="LCE",
            max_num_solvent_molecules=MAX_SOLVENTS,
            max_num_salt_molecules=MAX_SALTS,
            concentration_noise=0.0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_gnn_formulations,
        )

        fold_result = {
            "outer_idx": outer_idx,
            "inner_fold_records": [],
            "ensemble_preds": [],
            "test_targets": None,
            "test_elf_codes": None,
        }

        for m in range(N_ENSEMBLE):
            seed = MASTER_SEED + outer_idx * 100 + m
            set_seed(seed)

            # Inner train/val split
            train_idx, val_idx = train_test_split(
                np.arange(len(train_data_df)), test_size=VAL_SPLIT, random_state=seed
            )
            inner_train_df = train_data_df.iloc[train_idx].reset_index(drop=True)
            inner_val_df = train_data_df.iloc[val_idx].reset_index(drop=True)

            desc = f"GPU{gpu_id} Fold{outer_idx + 1} M{m + 1}"

            model, best_state, history, best_epoch, best_val_loss = (
                train_single_gnn_model(
                    inner_train_df, inner_val_df, molecule_encoder, device, seed, desc
                )
            )

            # Run inference on test set
            test_preds, test_targets, test_elf_codes = run_inference(
                model, test_loader, device
            )
            fold_result["ensemble_preds"].append(test_preds)
            fold_result["test_targets"] = test_targets
            fold_result["test_elf_codes"] = test_elf_codes

            # Calculate metrics
            test_r2 = r2_score(test_targets, test_preds)
            test_rmse = np.sqrt(np.mean((test_targets - test_preds) ** 2))

            fold_result["inner_fold_records"].append(
                {
                    "outer_fold": outer_idx + 1,
                    "model_idx": m + 1,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                }
            )

            print(
                f"  [GPU {gpu_id}] Fold {outer_idx + 1} Model {m + 1}: "
                f"Test R²={test_r2:.3f}, RMSE={test_rmse:.3f}, Best epoch={best_epoch}"
            )

            # Save checkpoint
            checkpoint = {
                "model_state_dict": best_state,
                "gnn_type": GNN_TYPE,
                "outer_fold": outer_idx + 1,
                "model_idx": m + 1,
                "seed": seed,
            }
            torch.save(
                checkpoint,
                os.path.join(weights_dir, f"outer_{outer_idx + 1}_model_{m + 1}.pth"),
            )

        result_queue.put(fold_result)
        print(f"[GPU {gpu_id}] Completed Outer Fold {outer_idx + 1}")

    except Exception as e:
        print(f"Error in worker GPU{gpu_id} Fold{outer_idx + 1}: {e}")
        import traceback

        traceback.print_exc()
        result_queue.put(None)


def main():
    mp.set_start_method("spawn", force=True)
    set_seed(MASTER_SEED)
    start_time = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    if n_gpus == 0:
        raise RuntimeError("This training entry point requires at least one CUDA GPU.")

    # Setup results directory and logging
    results_dir = get_results_dir(GNN_TYPE)
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "training.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"Results directory: {results_dir}")
    print(f"\n{'=' * 60}")
    print(f"{GNN_TYPE} NESTED CV + ENSEMBLE TRAINING")
    print(f"{'=' * 60}")
    print(f"Outer folds: {N_OUTER_FOLDS}")
    print(f"Ensemble size: {N_ENSEMBLE}")
    print(f"Total models: {N_OUTER_FOLDS * N_ENSEMBLE}")
    print(f"GPUs available: {n_gpus}")

    # Load dataset
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_path = os.path.join(project_root, "data", "HELENA-CE-26.csv")
    data, _ = load_dataset(data_path)
    data = data.sample(frac=1, random_state=MASTER_SEED).reset_index(drop=True)
    print(f"\nDataset: {len(data)} samples")

    # Get stratification groups
    strat_groups = get_stratification_groups(data)

    # Setup outer CV
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=False)
    folds = list(outer_cv.split(data, strat_groups))

    # Storage for results
    all_fold_results = [None] * N_OUTER_FOLDS

    print(f"\n{'=' * 60}")
    print(f"Processing {N_OUTER_FOLDS} Outer Folds in parallel across {n_gpus} GPUs")
    print(f"{'=' * 60}")

    # Launch parallel workers
    result_queue = mp.Queue()
    processes = []

    for outer_idx in range(N_OUTER_FOLDS):
        gpu_id = outer_idx % n_gpus
        train_idx, test_idx = folds[outer_idx]
        train_data_df = data.iloc[train_idx].reset_index(drop=True)
        test_data_df = data.iloc[test_idx].reset_index(drop=True)

        print(f"  Fold {outer_idx + 1} -> GPU {gpu_id}")

        p = mp.Process(
            target=train_outer_fold_worker,
            args=(
                outer_idx,
                gpu_id,
                train_data_df,
                test_data_df,
                weights_dir,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Collect results
    for _ in range(N_OUTER_FOLDS):
        result = result_queue.get()
        if result is not None:
            all_fold_results[result["outer_idx"]] = result

    for p in processes:
        p.join()

    # Aggregate results
    all_predictions = {}  # elf_code -> list of predictions
    outer_fold_metrics = []
    inner_fold_records = []
    individual_model_test_metrics = []

    for fold_result in all_fold_results:
        if fold_result is None:
            continue

        outer_idx = fold_result["outer_idx"]
        ensemble_preds = np.array(fold_result["ensemble_preds"])
        mean_preds = np.mean(ensemble_preds, axis=0)
        targets = fold_result["test_targets"]
        elf_codes = fold_result["test_elf_codes"]

        # Calculate fold metrics
        fold_metrics = calculate_regression_metrics(mean_preds, targets)
        outer_fold_metrics.append(
            {
                "outer_fold": outer_idx + 1,
                "n_test": len(targets),
                "r2": fold_metrics["r2"],
                "rmse": fold_metrics["rmse"],
                "mae": fold_metrics["mae"],
            }
        )
        print(
            f"Outer Fold {outer_idx + 1} Ensemble: R²={fold_metrics['r2']:.3f}, RMSE={fold_metrics['rmse']:.3f}"
        )

        # Store predictions by elf_code
        for elf_code, pred, preds_list in zip(elf_codes, mean_preds, ensemble_preds.T):
            if elf_code not in all_predictions:
                all_predictions[elf_code] = {
                    "preds": [],
                    "target": targets[np.where(elf_codes == elf_code)[0][0]],
                }
            all_predictions[elf_code]["preds"].extend(preds_list.tolist())

        inner_fold_records.extend(fold_result["inner_fold_records"])

        # Extract individual model test metrics
        for record in fold_result["inner_fold_records"]:
            individual_model_test_metrics.append(
                {
                    "outer_fold": record["outer_fold"],
                    "model_idx_in_fold": record["model_idx"],
                    "test_r2": record["test_r2"],
                    "test_rmse": record["test_rmse"],
                }
            )

    # Aggregate final predictions
    final_elf_codes = []
    final_targets = []
    final_predictions = []
    n_preds_list = []

    for elf_code in sorted(all_predictions.keys()):
        item = all_predictions[elf_code]
        final_elf_codes.append(elf_code)
        final_targets.append(item["target"])
        final_predictions.append(np.mean(item["preds"]))
        n_preds_list.append(len(item["preds"]))

    final_targets = np.array(final_targets)
    final_predictions = np.array(final_predictions)

    # Calculate final metrics
    final_metrics = calculate_regression_metrics(final_predictions, final_targets)
    final_metrics["n_samples"] = len(final_targets)
    final_metrics["avg_preds_per_sample"] = np.mean(n_preds_list)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "elf_code": final_elf_codes,
            "actual_lce": final_targets,
            "predicted_lce": final_predictions,
            "n_predictions": n_preds_list,
            "error": final_targets - final_predictions,
            "abs_error": np.abs(final_targets - final_predictions),
        }
    )

    # Save results
    save_results(
        results_dir,
        predictions_df,
        final_metrics,
        outer_fold_metrics,
        inner_fold_records,
        individual_model_test_metrics,
    )

    # Generate parity plot
    plot_parity(
        final_targets,
        final_predictions,
        final_metrics,
        os.path.join(results_dir, "nested_cv_parity_plot.png"),
        model_name=GNN_TYPE,
    )

    # Print final results
    print_final_results(final_metrics, GNN_TYPE)

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {results_dir}")

    # Close logging
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
