"""
This module contains utility functions for GNN-based regression tasks.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from config.datasets import MAX_SOLVENTS, MAX_SALTS

from .GNN_modules import MoleculeEncoder, BaselineFormulationModel

# Import from proper locations
from tasks.training.train_parameters import (
    batch_size,
    num_epochs,
    molar_ratio_noise as concentration_noise_range,
)
from core.plotting import PlotLossesLR
from tasks.training.train_loss_function import get_regression_loss

# Alias for backwards compatibility - use MSELoss as the baseline loss
HelenaCustomLoss = get_regression_loss

# Define data_labels for backwards compatibility
data_labels = "LCE"

# Use dataset config values
max_num_solvent_molecules = MAX_SOLVENTS
max_num_salt_molecules = MAX_SALTS


GNN_TYPES = ["GCN", "GAT", "MPNN", "AttentiveFP", "Molformer"]
DEFAULT_GNN_TYPE = "GCN"
NODE_FEATURES_DIM = 6
EDGE_FEATURES_DIM = 3
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GAT_HEADS = 4


def get_device(show_device=True):
    """Check if GPU is available."""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if show_device:
            print(f"\nUsing device: {device}\n")
        return device
    except Exception as e:
        print(f"Error detecting device: {e}")
        sys.exit(1)


def get_running_time(start_time):
    """Calculate the running time of a script."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60

    print("/" * 30)
    print(f"PROCESS COMPLETED in {elapsed_minutes:.2f} minutes")
    print("/" * 30 + "\n")


def load_dataset(data_path):
    """Load dataset for GNN models.

    Dataset format: SMILES and molar_ratio are combined in Solvent_X/Salt_X columns
    as 'SMILES:molar_ratio' (e.g., 'C1COC(=O)O1:4.49683177761123').
    """
    dtypes = {}

    # Molecule columns contain 'SMILES:molar_ratio' format
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str

    dtypes["Method"] = str
    dtypes["Row_ID"] = int
    dtypes["CE"] = float
    dtypes["LCE"] = float

    dataset = pd.read_csv(data_path, header=0, dtype=dtypes)
    no_formulations = len(dataset.dropna(how="all"))
    print(f"Number of rows in the dataset: {no_formulations}")

    return dataset, no_formulations


class GNNFormulationDataset(Dataset):
    """Dataset for GNN-based formulation property prediction."""

    def __init__(
        self,
        data_df,
        molecule_encoder,
        label_field="LCE",
        max_num_solvent_molecules=MAX_SOLVENTS,
        max_num_salt_molecules=MAX_SALTS,
        concentration_noise=0.03,
    ):
        """
        Args:
            data_df: DataFrame containing formulation data
            molecule_encoder: MoleculeEncoder instance
            label_field: Target property column name
            max_num_solvent_molecules: Maximum number of solvent molecules
            max_num_salt_molecules: Maximum number of salt molecules
            concentration_noise: Amount of noise to add to concentrations during training
        """
        self.data_df = data_df.copy()
        self.encoder = molecule_encoder
        self.label_field = label_field
        self.max_num_solvent_molecules = max_num_solvent_molecules
        self.max_num_salt_molecules = max_num_salt_molecules
        self.max_num_molecules = max_num_solvent_molecules + max_num_salt_molecules
        self.concentration_noise = concentration_noise

        # Ensure method column is present and numeric
        self.data_df["Method"] = self.data_df["Method"].fillna("Aurbach")
        self.data_df["method_numeric"] = self.data_df["Method"].map(
            {"Aurbach": 0, "Cycle": 1}
        )

        # Ensure current column is present and log-transformed
        if "Current" in self.data_df.columns:
            self.data_df["log_current"] = np.log10(self.data_df["Current"] + 1e-8)
        else:
            self.data_df["log_current"] = 0.0

        # Process data
        self.processed_data = self._process_data()

    def _add_concentration_noise(self, concentration, concentration_noise_range=0.03):
        """Add noise to concentration value."""
        if concentration <= 0:
            return concentration

        min_allowed = concentration * (1 - concentration_noise_range)
        max_allowed = concentration * (1 + concentration_noise_range)
        return np.random.uniform(min_allowed, max_allowed)

    def _process_data(self):
        """Process and store data tuples with molecule SMILES and concentrations.

        Handles the HELENA dataset format where molecules are stored as 'SMILES:molar_ratio'.
        """
        processed_data = []

        for idx, row in self.data_df.iterrows():
            smiles_list = []
            concentrations = []

            # Process solvent molecules
            for i in range(1, self.max_num_solvent_molecules + 1):
                mol_col = f"Solvent_{i}"

                if mol_col in row and pd.notna(row[mol_col]):
                    # Parse SMILES:molar_ratio format
                    value = str(row[mol_col])
                    if ":" in value:
                        parts = value.split(":")
                        smiles = parts[0].strip()
                        molar_ratio = float(parts[1].strip())
                        smiles_list.append(smiles)
                        concentrations.append(molar_ratio)
                    else:
                        # Fallback: treat as SMILES only with default concentration
                        smiles_list.append(value)
                        concentrations.append(1.0)
                else:
                    smiles_list.append(None)
                    concentrations.append(0.0)

            # Process salt molecules
            for i in range(1, self.max_num_salt_molecules + 1):
                mol_col = f"Salt_{i}"

                if mol_col in row and pd.notna(row[mol_col]):
                    # Parse SMILES:molar_ratio format
                    value = str(row[mol_col])
                    if ":" in value:
                        parts = value.split(":")
                        smiles = parts[0].strip()
                        molar_ratio = float(parts[1].strip())
                        smiles_list.append(smiles)
                        concentrations.append(molar_ratio)
                    else:
                        # Fallback: treat as SMILES only with default concentration
                        smiles_list.append(value)
                        concentrations.append(1.0)
                else:
                    smiles_list.append(None)
                    concentrations.append(0.0)

            label = row[self.label_field]
            elf_code = row["Row_ID"]
            method = row["method_numeric"]
            log_current = row["log_current"]

            processed_data.append(
                {
                    "smiles": smiles_list,
                    "concentrations": concentrations,
                    "label": label,
                    "elf_code": elf_code,
                    "method": method,
                    "log_current": log_current,
                }
            )

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # Add noise to concentrations during training
        noisy_concentrations = [
            self._add_concentration_noise(c) for c in item["concentrations"]
        ]

        # Preprocess molecular graphs for valid molecules
        mol_graphs = []
        for i, smiles in enumerate(item["smiles"]):
            if smiles is not None:
                # Check if this is a Molformer encoder (uses SMILES strings directly)
                if hasattr(self.encoder, "encode_smiles"):
                    # For Molformer: keep SMILES as string
                    mol_graphs.append((i, smiles))
                else:
                    # For GNN encoders: convert to graph
                    graph = self.encoder.smiles_to_graph(smiles)
                    if graph is not None:
                        mol_graphs.append((i, graph))

        # Return processed data
        return {
            "smiles": item["smiles"],
            "concentrations": noisy_concentrations,
            "mol_graphs": mol_graphs,
            "label": item["label"],
            "elf_code": item["elf_code"],
            "method": item["method"],
            "log_current": item["log_current"],
        }


def collate_gnn_formulations(batch):
    """Collate function for batching GNN formulation data."""
    batch_smiles = [item["smiles"] for item in batch]
    batch_concentrations = [item["concentrations"] for item in batch]
    batch_mol_graphs = [item["mol_graphs"] for item in batch]
    batch_labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    batch_elf_codes = torch.tensor(
        [item["elf_code"] for item in batch], dtype=torch.long
    )
    batch_methods = torch.tensor(
        [item["method"] for item in batch], dtype=torch.float32
    )
    batch_currents = torch.tensor(
        [item["log_current"] for item in batch], dtype=torch.float32
    )

    # Reorganize mol_graphs to be grouped by formulation
    processed_mol_graphs = []
    for formulation_graphs in batch_mol_graphs:
        # Sort by original index
        sorted_graphs = sorted(formulation_graphs, key=lambda x: x[0])
        # Extract just the graph data
        graphs_only = [g[1] for g in sorted_graphs]
        processed_mol_graphs.append(graphs_only)

    return {
        "smiles": batch_smiles,
        "concentrations": torch.tensor(batch_concentrations, dtype=torch.float32),
        "mol_graphs": processed_mol_graphs,
        "labels": batch_labels,
        "elf_codes": batch_elf_codes,
        "methods": batch_methods,
        "currents": batch_currents,
    }


def dataset_partition(dataset, train_size, val_size, random_seed=42):
    """Partition dataset into train, validation, and test sets."""
    if random_seed:
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )
    else:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Split dataset
    train_end = int(train_size * len(dataset))
    val_end = train_end + int(val_size * len(dataset))

    train_df = dataset.iloc[:train_end]
    val_df = dataset.iloc[train_end:val_end]
    test_df = dataset.iloc[val_end:]

    return train_df, val_df, test_df


def dataset_partition_kfold(dataset, kfold_test_size, random_seed=42):
    """Partition dataset into k-folds with held-out test set."""
    if random_seed:
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )
    else:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    train_val_idx = int((1 - kfold_test_size) * len(dataset))

    train_val_data = dataset.iloc[:train_val_idx]
    test_data = dataset.iloc[train_val_idx:]

    print(f"\nTotal dataset size:\t{len(dataset)}")
    print(f"Train+Val size:\t\t{len(train_val_data)}")
    print(f"Test size:\t\t{len(test_data)}")

    return train_val_data, test_data


def create_gnn_datasets(*dataframes, molecule_encoder=None):
    """Create GNN datasets from dataframes."""
    if molecule_encoder is None:
        molecule_encoder = MoleculeEncoder()

    datasets = []
    for df in dataframes:
        dataset = GNNFormulationDataset(
            df,
            molecule_encoder,
            label_field=data_labels,
            max_num_solvent_molecules=max_num_solvent_molecules,
            max_num_salt_molecules=max_num_salt_molecules,
            concentration_noise=concentration_noise_range,
        )
        datasets.append(dataset)

    return datasets


def create_gnn_dataloaders(
    *datasets, batch_size=32, shuffle_default=False, sampler_default=None
):
    """Create data loaders for GNN datasets."""
    loaders = []

    for dataset in datasets:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_default,
            sampler=sampler_default,
            collate_fn=collate_gnn_formulations,
        )
        loaders.append(loader)

    return loaders


def initialize_gnn_model(
    gnn_type=DEFAULT_GNN_TYPE, device=None, molformer_encoder=None
):
    """Initialize a GNN-based formulation model."""
    if gnn_type not in GNN_TYPES:
        print(
            f"Warning: GNN type {gnn_type} not recognized. Using {DEFAULT_GNN_TYPE} instead."
        )
        gnn_type = DEFAULT_GNN_TYPE

    model = BaselineFormulationModel(
        gnn_type=gnn_type,
        node_features_dim=NODE_FEATURES_DIM,
        edge_features_dim=EDGE_FEATURES_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=1,
        num_layers=GNN_NUM_LAYERS,
        heads=GAT_HEADS,
        max_molecules=max_num_solvent_molecules + max_num_salt_molecules,
        molformer_encoder=molformer_encoder,
    )

    if device:
        model = model.to(device)

    return model


def create_baseline_model(
    gnn_type=DEFAULT_GNN_TYPE,
    hidden_dim=GNN_HIDDEN_DIM,
    num_layers=GNN_NUM_LAYERS,
    heads=GAT_HEADS,
    molformer_encoder=None,
):
    """Create a baseline formulation model with specified parameters."""
    model = BaselineFormulationModel(
        gnn_type=gnn_type,
        node_features_dim=NODE_FEATURES_DIM,
        edge_features_dim=EDGE_FEATURES_DIM,
        hidden_dim=hidden_dim,
        out_dim=1,
        num_layers=num_layers,
        heads=heads,
        max_molecules=max_num_solvent_molecules + max_num_salt_molecules,
        molformer_encoder=molformer_encoder,
    )
    return model


def setup_gnn_optimizer(model):
    """Set up optimizer for GNN model."""
    learning_rate = 1e-4  # Higher LR for Molformer on small dataset
    weight_decay = 0.001
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def setup_gnn_scheduler(optimizer, train_loader, num_epochs, max_lr=1e-3):
    """Set up learning rate scheduler for GNN model."""
    total_steps = len(train_loader) * num_epochs
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=100.0,
    )


def train_gnn_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    save_weight_file,
    plot_save_path=None,
    show_epoch_info=False,
    plot_losses=False,
    kfold_train=False,
    patience=10,
):
    """Train a GNN-based model with early stopping."""
    # Use the same loss function for consistency with HELENA
    criterion = HelenaCustomLoss()
    optimizer = setup_gnn_optimizer(model)
    scheduler = setup_gnn_scheduler(optimizer, train_loader, num_epochs)

    train_results = defaultdict(list)
    train_results["task"] = "Training"
    val_results = defaultdict(list)
    val_results["task"] = "Validation"
    best_val_loss = float("inf")

    # Early stopping parameters
    patience_counter = 0
    early_stop = False

    if plot_losses:
        train_losses = []
        val_losses = []
        learning_rates = []
        training_plot = PlotLossesLR()
        fig, plot_elements = training_plot.plot_training_loss_and_lr(
            train_losses, val_losses, learning_rates
        )

    if kfold_train:
        max_epochs = tqdm(range(num_epochs), desc="Training", leave=False)
    else:
        print("\nStarting GNN training...")
        max_epochs = range(num_epochs)

    for epoch in max_epochs:
        model.train()

        total_loss = 0
        epoch_train_predictions = []
        epoch_train_truths = []
        epoch_train_elf_codes = []

        if not kfold_train:
            print(f"\n---Epoch {epoch + 1}/{num_epochs}")
            max_batch = tqdm(train_loader, desc="Training", leave=False)
        else:
            max_batch = train_loader

        for batch in max_batch:
            # Move batch data to device
            methods = batch["methods"].to(device)
            labels = (
                batch["labels"].to(device).view(-1, 1)
            )  # Reshape to [batch_size, 1]
            elf_codes = batch["elf_codes"]
            concentrations = batch["concentrations"].to(device)
            mol_graphs = batch["mol_graphs"]

            optimizer.zero_grad()
            outputs = model(mol_graphs, concentrations)
            loss = criterion(outputs, labels, methods)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_train_predictions.extend(
                outputs.detach().cpu().numpy().flatten()
            )  # Flatten predictions
            epoch_train_truths.extend(
                labels.detach().cpu().numpy().flatten()
            )  # Flatten labels
            epoch_train_elf_codes.extend(elf_codes.detach().cpu().numpy())

        epoch_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        epoch_val_results = evaluate_gnn_model(
            model, val_loader, device, kfold_val=kfold_train
        )
        (
            epoch_val_predictions,
            epoch_val_targets,
            epoch_val_elf_codes,
            epoch_val_loss,
        ) = (
            epoch_val_results["predictions"],
            epoch_val_results["truths"],
            epoch_val_results["elf_codes"],
            epoch_val_results["loss"],
        )

        if show_epoch_info:
            print(
                f"Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, LR: {current_lr:.2e}"
            )

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0  # Reset counter

            # Save results when validation loss decreases
            train_results["loss"] = epoch_train_loss
            train_results["predictions"] = np.array(epoch_train_predictions)
            train_results["truths"] = np.array(epoch_train_truths)
            train_results["elf_codes"] = np.array(epoch_train_elf_codes)
            val_results["loss"] = best_val_loss
            val_results["predictions"] = epoch_val_predictions
            val_results["truths"] = epoch_val_targets
            val_results["elf_codes"] = epoch_val_elf_codes

            torch.save(model.state_dict(), save_weight_file)

            if show_epoch_info:
                print(f"Validation loss decreased, model weights saved")
        else:
            patience_counter += 1
            if show_epoch_info:
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                if show_epoch_info:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                early_stop = True
                break

        if plot_losses:
            # Update plots
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            learning_rates.append(current_lr)
            training_plot.update_loss_lr_plots(
                epoch, train_losses, val_losses, learning_rates, plot_elements
            )

    if plot_losses:
        training_plot.save_loss_lr_plot(fig, plot_save_path)
        print("\n")
        print("+" * 50)
        print(f"GNN Training {'stopped early' if early_stop else 'COMPLETE'}.")
        print(f"\nLoss plot saved to {plot_save_path}")
        print(f"Model weights saved to {save_weight_file}")
        print("+" * 50, "\n")

    return train_results, val_results


def evaluate_gnn_model(
    model, loader, device, task="Test", weights_path=None, kfold_val=False
):
    """Evaluate a GNN-based model on the given dataset."""
    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )

    criterion = HelenaCustomLoss()

    model.eval()
    total_loss = 0
    test_results = defaultdict(list)
    test_results["task"] = task

    with torch.no_grad():
        batch_test_predictions = []
        batch_test_truths = []
        batch_test_elf_codes = []

        if not kfold_val:
            max_batch = tqdm(loader, desc=f"Evaluating ({task})", leave=False)
        else:
            max_batch = loader

        for batch in max_batch:
            # Move batch data to device
            methods = batch["methods"].to(device)
            labels = (
                batch["labels"].to(device).view(-1, 1)
            )  # Reshape to [batch_size, 1]
            elf_codes = batch["elf_codes"]
            concentrations = batch["concentrations"].to(device)
            mol_graphs = batch["mol_graphs"]

            outputs = model(mol_graphs, concentrations)
            loss = criterion(outputs, labels, methods)

            total_loss += loss.item() * labels.size(0)
            batch_test_predictions.extend(
                outputs.detach().cpu().numpy().flatten()
            )  # Flatten predictions
            batch_test_truths.extend(
                labels.detach().cpu().numpy().flatten()
            )  # Flatten labels
            batch_test_elf_codes.extend(elf_codes.detach().cpu().numpy())

    test_loss = total_loss / len(loader.dataset)
    test_results["loss"] = test_loss
    test_results["predictions"] = np.array(batch_test_predictions)
    test_results["truths"] = np.array(batch_test_truths)
    test_results["elf_codes"] = np.array(batch_test_elf_codes)

    return test_results


def run_gnn_inference(model, loader, device, weights_path=None):
    """Run inference with a GNN-based model."""
    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )

    model.eval()
    inference_results = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            # Move batch data to device
            concentrations = batch["concentrations"].to(device)
            mol_graphs = batch["mol_graphs"]
            elf_codes = batch["elf_codes"]

            outputs = model(mol_graphs, concentrations)

            inference_results["predictions"].extend(
                outputs.detach().cpu().numpy().flatten()
            )  # Flatten predictions
            inference_results["elf_codes"].extend(elf_codes.cpu().numpy())

    # Convert to numpy arrays
    inference_results["predictions"] = np.array(inference_results["predictions"])
    inference_results["elf_codes"] = np.array(inference_results["elf_codes"])

    return inference_results


def calculate_metrics(*results, show_results=False, show_prediction_ranges=False):
    """Calculate RMSE and R² metrics for model evaluation."""
    metrics = {}
    print(f"METRICS RESULTS:") if show_results else None

    for result in results:
        task = result["task"]
        y_true = result["truths"]
        y_pred = result["predictions"]

        mse = mean_squared_error(y_true, y_pred)
        task_metrics = {
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_true, y_pred),
            "pearson_r": stats.pearsonr(y_true, y_pred)[0],
        }
        metrics[task] = task_metrics

        if show_results:
            print(f"\n{task}")
            print(
                f"\tLoss: {result['loss']:.4f}, RMSE: {task_metrics['rmse']:.4f}, "
                f"R2: {task_metrics['r2']:.4f}, Pearson r: {task_metrics['pearson_r']:.4f}"
            )

        if show_prediction_ranges:
            print(f"\tPrediction ranges:\t[{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
            print(f"\tTarget ranges:\t\t[{np.min(y_true):.4f}, {np.max(y_true):.4f}]")

    return metrics


def save_gnn_predictions_to_csv(*results, save_path=None, gnn_type=DEFAULT_GNN_TYPE):
    """Save GNN model predictions to CSV."""
    if save_path is None:
        save_path = os.getcwd()

    save_predictions_path = os.path.join(
        save_path, f"{gnn_type.lower()}_predictions.csv"
    )
    combined_df = pd.DataFrame()

    for result in results:
        task = result["task"]
        elf_codes = result["elf_codes"]
        truths = result["truths"]
        predictions = result["predictions"]

        df = pd.DataFrame(
            {
                "Task": task,
                "Elf_Code": elf_codes,
                "Truth": truths,
                "Prediction": predictions,
            }
        )
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Add error metrics
    combined_df["Absolute_Error"] = abs(
        combined_df["Truth"] - combined_df["Prediction"]
    )
    combined_df["Percent_Error"] = (
        combined_df["Absolute_Error"] / combined_df["Truth"].clip(lower=1e-5)
    ) * 100

    # Save to CSV
    combined_df.to_csv(save_predictions_path, index=False)
    print(f"GNN prediction data saved to {save_predictions_path}\n")

    return combined_df


def plot_gnn_results(
    train_results,
    val_results,
    test_results,
    metrics,
    save_path,
    gnn_type=DEFAULT_GNN_TYPE,
):
    """Create GNN evaluation plots with metrics."""
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
    metrics_text_pos = (0.85, 0.2)
    histogram_pos = [0.137, 0.74, 0.25, 0.2]
    axis_tick_size = 14
    axis_title_size = 18
    plot_title_size = 20
    histogram_tick_size = 12
    plt.figure(figsize=(12, 10))
    plt.scatter(train_true, train_pred, color="red", alpha=0.5, label="Training")
    plt.scatter(val_true, val_pred, color="blue", alpha=0.5, label="Validation")
    plt.scatter(test_true, test_pred, color="green", alpha=0.5, label="Test")

    # Add y=x line
    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    # Add metrics text
    metrics_text = (
        f"Training:\n RMSE: {train_rmse:.4f}\n R2: {train_r2:.4f}\n\n"
        f"Validation:\n RMSE: {val_rmse:.4f}\n R2: {val_r2:.4f}\n\n"
        f"Test:\n RMSE: {test_rmse:.4f}\n R2: {test_r2:.4f}"
    )
    plt.gcf().text(
        metrics_text_pos[0],
        metrics_text_pos[1],
        metrics_text,
        fontsize=axis_tick_size,
        va="center",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Add colored dots
    dot_positions = [(0.11, "red"), (0.023, "blue"), (-0.065, "green")]
    for y_offset, color in dot_positions:
        plt.gcf().text(
            metrics_text_pos[0] - 0.02,
            metrics_text_pos[1] + y_offset,
            "●",
            color=color,
            fontsize=axis_tick_size,
            va="center",
            ha="center",
        )

    # Set labels and title
    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)
    plt.title(
        f"{gnn_type} -- Actual vs Predicted Values with Metrics",
        fontsize=plot_title_size,
    )
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)

    # Add prediction distribution histogram
    inset_axes = plt.axes(histogram_pos)
    inset_axes.hist(all_pred, bins=50, alpha=0.75, color="gray")
    inset_axes.tick_params(axis="both", labelsize=histogram_tick_size)
    inset_axes.set_xlabel("Predictions", fontsize=histogram_tick_size)
    inset_axes.set_ylabel("Frequency", fontsize=histogram_tick_size)

    plt.tight_layout()
    gnn_plot_file_name = f"{gnn_type.lower()}_parity_plot.png"
    gnn_plot_full_save_path = os.path.join(save_path, gnn_plot_file_name)
    plt.savefig(gnn_plot_full_save_path)
    plt.close()

    print(f"GNN results plot saved to: {gnn_plot_full_save_path}")


def plot_kfold_cross_results(fold_metrics, save_path, gnn_type=DEFAULT_GNN_TYPE):
    """Plot cross-validation results focusing on RMSE and R2."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_to_plot = ["rmse", "r2"]
    data_to_plot = [
        [fold_metrics[fold]["metrics"][metric] for fold in fold_metrics.keys()]
        for metric in metrics_to_plot
    ]

    ax1.boxplot(data_to_plot, labels=["RMSE", "R²"])
    ax1.set_title(f"{gnn_type} - Distribution of Metrics Across Folds")
    ax1.grid(True)

    # Plot predictions vs actual for all folds
    all_true = []
    all_pred = []
    for fold in fold_metrics.keys():
        all_true.extend(fold_metrics[fold]["truths"])
        all_pred.extend(fold_metrics[fold]["predictions"])

    ax2.scatter(all_true, all_pred, alpha=0.5)
    ax2.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], "r--")
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title(f"{gnn_type} - Predicted vs Actual Values (All Folds)")
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    kfold_plot_filename = f"{gnn_type.lower()}_kfold_cross_val_results.png"
    plot_save_path = os.path.join(save_path, kfold_plot_filename)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"\nK-fold cross-validation plot saved to: {plot_save_path}")


def kfold_cross_validation(
    device,
    train_val_dataset,
    test_dataset,
    kfold_n_splits,
    gnn_type=DEFAULT_GNN_TYPE,
    kfold_weights_dir="kfold_weights",
    kfold_patience=10,
    molformer_encoder=None,
):
    """Perform k-fold cross-validation for GNN model."""
    os.makedirs(kfold_weights_dir, exist_ok=True)
    best_model_path = os.path.join(
        kfold_weights_dir, f"{gnn_type.lower()}_best_model.pth"
    )

    def calculate_average_kfold_metrics(all_kfold_metrics):
        """Calculate average metrics across all folds."""
        print("\n")
        print("+" * 50, f"\n{gnn_type} K-Fold Cross-Validation COMPLETE.")
        print("\nAverage Results Across Folds:")
        avg_kfold_metrics = {}
        avg_metrics = {}
        std_metrics = {}
        for metric in ["rmse", "r2"]:
            values = [
                all_kfold_metrics[fold]["metrics"][metric]
                for fold in all_kfold_metrics.keys()
            ]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
            print(
                f"\t{metric.upper()}:\t{avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"
            )
        print("+" * 50)
        avg_kfold_metrics["kfold_avg"] = avg_metrics
        avg_kfold_metrics["kfold_std"] = std_metrics

        return avg_kfold_metrics

    all_kfold_metrics = {}
    best_overall_rmse = float("inf")

    # Initialize cross-validation splitter
    kfold = KFold(n_splits=kfold_n_splits, shuffle=True, random_state=42)

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(range(len(train_val_dataset)))
    ):
        print(f"\nFold {fold + 1}/{kfold_n_splits}")

        fold_weights_path = os.path.join(
            kfold_weights_dir, f"{gnn_type.lower()}_fold_{fold + 1}_model.pth"
        )

        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = create_gnn_dataloaders(
            train_val_dataset, sampler_default=train_sampler
        )[0]
        val_loader = create_gnn_dataloaders(
            train_val_dataset, sampler_default=val_sampler
        )[0]

        model = initialize_gnn_model(gnn_type, device, molformer_encoder)

        _, fold_val_results = train_gnn_model(
            model,
            train_loader,
            val_loader,
            num_epochs,
            device,
            fold_weights_path,
            kfold_train=True,
            patience=kfold_patience,
        )

        # Calculate metrics for this fold
        fold_metrics_dict = calculate_metrics(fold_val_results)

        # Store both metrics and results for this fold
        all_kfold_metrics[fold] = {
            "metrics": fold_metrics_dict["Validation"],
            "predictions": fold_val_results["predictions"],
            "truths": fold_val_results["truths"],
        }
        current_rmse = fold_metrics_dict["Validation"]["rmse"]

        print(f"Fold {fold + 1} model saved to {fold_weights_path}")

        # Track best overall fold model
        if current_rmse < best_overall_rmse:
            best_overall_rmse = current_rmse
            best_train_loader = train_loader
            best_val_loader = val_loader
            torch.save(model.state_dict(), best_model_path)
            print(f"New best overall model saved to {best_model_path}")

        print(f"Fold {fold + 1} Results:")
        print(f"\tRMSE:\t{current_rmse:.4f}")
        print(f"\tR2:\t{fold_metrics_dict['Validation']['r2']:.4f}")

    avg_kfold_metrics = calculate_average_kfold_metrics(all_kfold_metrics)

    test_loader = create_gnn_dataloaders(test_dataset)[0]
    model = initialize_gnn_model(gnn_type, device, molformer_encoder)
    train_results = evaluate_gnn_model(
        model, best_train_loader, device, task="Training", weights_path=best_model_path
    )
    val_results = evaluate_gnn_model(
        model, best_val_loader, device, task="Validation", weights_path=best_model_path
    )
    test_results = evaluate_gnn_model(
        model, test_loader, device, task="Test", weights_path=best_model_path
    )

    return (
        train_results,
        val_results,
        test_results,
        all_kfold_metrics,
        avg_kfold_metrics,
    )
