"""Train HELENA-CE component ablations with the main regression splits and seeds."""

import sys
import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    get_combined_stratification_groups,
)
from modules.helena_ce.helena_ce_model import HELENA_CE
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE
from tasks.training.train_loss_function import get_regression_loss
from core.utils import (
    load_dataset,
    load_preprocessed_data,
    get_running_time,
    feature_engineering,
)
from modules.dataset_processing import CollateFn
from modules.tokenizer.tokenizer_helena import Tokenizer
from core.smiles_preprocessing import smiles_dict_preprocessing
from config.datasets import MAX_MOLECULES
from config.paths import DATA_DIR

MASTER_SEED = 142
VAL_SPLIT = 0.1

DATA_PATH = os.path.join(DATA_DIR, "HELENA-CE-26.csv")
base_results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)
os.makedirs(base_results_dir, exist_ok=True)




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
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_results_dir(base_dir):
    """Create a results directory with PID."""
    pid = os.getpid()
    run_dir = os.path.join(base_dir, f"Ablation_Ensemble_{pid}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir




class HELENA_CE_NoMoietyEncoder(HELENA_CE):
    """
    HELENA-CE variant without moiety attention.
    Replaces MoietyEncoder with raw bag-of-tokens counting (no learned projection).
    Tests: Is hierarchical moiety attention necessary?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        vocab_size = self.encoder_moiety.embedding.num_embeddings
        d_model = self.encoder_moiety.embedding.embedding_dim

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            batch_size = moiety_inputs.shape[0]
            counts = torch.zeros(batch_size, vocab_size, device=device)
            for b in range(batch_size):
                unique, cnts = torch.unique(moiety_inputs[b], return_counts=True)
                counts[b, unique.long()] = cnts.float()
            counts[:, 0] = 0
            counts = counts / (counts.sum(dim=1, keepdim=True).clamp(min=1.0))

            if vocab_size >= d_model:
                mol_embedding = counts[:, :d_model]
            else:
                mol_embedding = torch.zeros(batch_size, d_model, device=device)
                mol_embedding[:, :vocab_size] = counts
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1
        )

        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        formulation_features = self.formulation_norm(combined_output)

        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)

        method = batch["method"].to(device).float().unsqueeze(-1)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}


class HELENA_CE_NoMolarRatio(HELENA_CE):
    """
    HELENA-CE variant without molar ratio encoder.
    Uses identity (ones) instead of learned molar ratio weights.
    Tests: Does learned molar ratio encoding improve predictions?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        d_model = mol_embeddings.shape[-1]
        mr_weights = torch.ones(
            molar_ratios.shape[0], molar_ratios.shape[1], d_model, device=device
        )
        zero_mask = (molar_ratios == 0).unsqueeze(-1)
        mr_weights = mr_weights.masked_fill(zero_mask, 0.0)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1
        )

        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        formulation_features = self.formulation_norm(combined_output)

        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)

        method = batch["method"].to(device).float().unsqueeze(-1)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}


class HELENA_CE_NoSaltGroup(HELENA_CE):
    """
    HELENA-CE variant without salt information.
    Completely removes salt molecules from the formulation representation.
    Tests: How important is salt chemistry for CE prediction?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)

        salt_out = torch.zeros_like(solvent_out)

        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), torch.zeros_like(solvent_mask.any(dim=1))], dim=1
        )

        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        formulation_features = self.formulation_norm(combined_output)

        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)

        method = batch["method"].to(device).float().unsqueeze(-1)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}


class HELENA_CE_NoSolventGroupAttention(HELENA_CE):
    """
    HELENA-CE variant without solvent group attention.
    Replaces group attention for solvents with simple mean pooling.
    Tests: Is attention over multiple solvents beneficial?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_masked = scaled_mol_embeddings * solvent_mask.unsqueeze(-1).float()
        solvent_count = solvent_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        solvent_out = solvent_masked.sum(dim=1) / solvent_count
        solvent_out = torch.nan_to_num(solvent_out, nan=0.0)

        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1
        )

        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        formulation_features = self.formulation_norm(combined_output)

        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)

        method = batch["method"].to(device).float().unsqueeze(-1)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}


class HELENA_CE_NoFormulationAttention(HELENA_CE):
    """
    HELENA-CE variant without formulation (molecular) attention.
    Replaces attention with simple averaging of solvent and salt outputs.
    Tests: Is cross-attention between solvent/salt groups beneficial?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        combined_output = solvent_out
        combined_output = torch.nan_to_num(combined_output, nan=0.0)

        formulation_features = self.formulation_norm(combined_output)

        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)

        method = batch["method"].to(device).float().unsqueeze(-1)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}


class HELENA_CE_NoCurrent(HELENA_CE):
    """
    HELENA-CE variant without current density input.
    Replaces ln(current) with zeros.
    Tests: How important is current density for CE prediction?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1
        )

        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        formulation_features = self.formulation_norm(combined_output)

        batch_size = formulation_features.shape[0]
        ln_current = torch.zeros(batch_size, 1, device=device)

        method = batch["method"].to(device).float().unsqueeze(-1)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}


class HELENA_CE_NoMethod(HELENA_CE):
    """
    HELENA-CE variant without measurement method input.
    Replaces method with zeros.
    Tests: How important is measurement method for CE prediction?
    """

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1
        )

        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        formulation_features = self.formulation_norm(combined_output)

        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)

        batch_size = formulation_features.shape[0]
        method = torch.zeros(batch_size, 1, device=device)

        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}



ABLATION_CONFIGS = {
    "no_moiety_encoder": {
        "name": "No Moiety Encoder",
        "description": "Replace moiety attention with simple embedding + averaging",
        "model_class": HELENA_CE_NoMoietyEncoder,
    },
    "no_molar_ratio": {
        "name": "No Molar Ratio Encoder",
        "description": "Remove molar ratio weighting (use constant 1.0)",
        "model_class": HELENA_CE_NoMolarRatio,
    },
    "no_salt_group": {
        "name": "No Salt Group",
        "description": "Completely remove salt information from the model",
        "model_class": HELENA_CE_NoSaltGroup,
    },
    "no_solvent_group_attention": {
        "name": "No Solvent Group Attention",
        "description": "Replace solvent group attention with simple mean pooling",
        "model_class": HELENA_CE_NoSolventGroupAttention,
    },
    "no_formulation_attention": {
        "name": "No Formulation Attention",
        "description": "Replace formulation attention with simple averaging",
        "model_class": HELENA_CE_NoFormulationAttention,
    },
    "no_current": {
        "name": "No Current",
        "description": "Remove current density from input (use zeros)",
        "model_class": HELENA_CE_NoCurrent,
    },
    "no_method": {
        "name": "No Method",
        "description": "Remove measurement method from input (use zeros)",
        "model_class": HELENA_CE_NoMethod,
    },
}




def plot_parity(targets, predictions, metrics, path, title_prefix=""):
    """Create a parity plot for predictions vs actual values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5, s=50, edgecolors="none", c="green")
    min_val, max_val = (
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    )
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    plt.xlabel("Actual LCE", fontsize=14)
    plt.ylabel("Predicted LCE", fontsize=14)
    title = f"R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()




def train_single_ablation_model(
    train_data, map_dict, vocab_size, device, seed, model_class, ablation_name, desc=""
):
    """
    Train a single ablation model with train/val split.
    Same logic as train_single_model() in train_ensemble_regression.py.
    """
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

    model = model_class(
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


def run_inference_ablation(model, data_loader, device):
    """Run inference with ablation model."""
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


def train_outer_fold_worker_ablation(
    outer_idx,
    gpu_id,
    train_data,
    test_data,
    map_dict,
    vocab_size,
    moiety_key_map,
    key_moiety_map,
    weights_dir,
    model_class,
    ablation_name,
    result_queue,
):
    """Worker function to train all ensemble models for one outer fold."""
    try:
        os.makedirs(weights_dir, exist_ok=True)

        device = torch.device(f"cuda:{gpu_id}")
        print(f"[GPU {gpu_id}] Starting Outer Fold {outer_idx + 1} - {ablation_name}")

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
            "ensemble_preds": [],
            "test_targets": None,
            "row_ids": None,
            "all_loss_records": [],
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
            ) = train_single_ablation_model(
                train_data,
                map_dict,
                vocab_size,
                device,
                seed,
                model_class,
                ablation_name,
                desc,
            )

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

            checkpoint = {
                "model_state_dict": best_state,
                "moiety_key_map": moiety_key_map,
                "key_moiety_map": key_moiety_map,
                "vocab_size": vocab_size,
                "ablation": ablation_name,
                "outer_fold": outer_idx + 1,
                "model_idx": m + 1,
                "seed": seed,
            }
            torch.save(
                checkpoint,
                os.path.join(weights_dir, f"outer_{outer_idx + 1}_model_{m + 1}.pth"),
            )

            preds, targets, row_ids = run_inference_ablation(model, test_loader, device)
            fold_result["ensemble_preds"].append(preds)
            fold_result["test_targets"] = targets
            fold_result["row_ids"] = row_ids

            model_test_r2 = r2_score(targets, preds)
            model_test_rmse = np.sqrt(np.mean((targets - preds) ** 2))
            fold_result["individual_model_test_metrics"].append({
                "outer_fold": outer_idx + 1,
                "model_idx_in_fold": m + 1,
                "test_r2": model_test_r2,
                "test_rmse": model_test_rmse,
            })

            for epoch_idx, (train_loss, val_loss) in enumerate(
                zip(history["train_losses"], history["val_losses"])
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

            for pred, target, rid in zip(preds, targets, row_ids):
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


def run_ablation_single(
    ce_data,
    map_dict,
    vocab_size,
    moiety_key_map,
    key_moiety_map,
    model_class,
    ablation_name,
    ablation_dir,
    n_gpus,
):
    """
    Run nested CV + ensemble for a single ablation variant.
    Same structure as main() in train_ensemble_regression.py.
    """
    weights_dir = os.path.join(ablation_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {ablation_name}")
    print(f"{'=' * 60}")
    print(f"Dataset: {len(ce_data)} samples")

    ce_data = ce_data.sample(frac=1, random_state=MASTER_SEED).reset_index(drop=True)

    tokenizer = Tokenizer()
    strat_groups = get_combined_stratification_groups(ce_data, tokenizer)

    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=False)
    folds = list(outer_cv.split(ce_data, strat_groups))

    all_fold_results = [None] * N_OUTER_FOLDS

    print(f"\nProcessing {N_OUTER_FOLDS} Outer Folds in parallel across {n_gpus} GPUs")

    result_queue = mp.Queue()
    processes = []

    for outer_idx in range(N_OUTER_FOLDS):
        gpu_id = outer_idx % n_gpus
        train_idx, test_idx = folds[outer_idx]
        train_data = ce_data.iloc[train_idx].reset_index(drop=True)
        test_data = ce_data.iloc[test_idx].reset_index(drop=True)

        print(f"  Fold {outer_idx + 1} -> GPU {gpu_id}")

        p = mp.Process(
            target=train_outer_fold_worker_ablation,
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
                model_class,
                ablation_name,
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
    outer_fold_metrics = []
    inner_fold_records = []
    all_loss_records = []
    all_parity_records = []
    individual_model_test_metrics = []

    for fold_result in all_fold_results:
        if fold_result is None:
            continue

        outer_idx = fold_result["outer_idx"]

        ensemble_preds = np.array(fold_result["ensemble_preds"])
        mean_preds = np.mean(ensemble_preds, axis=0)
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
        all_targets.extend(targets)
        all_row_ids.extend(fold_result["row_ids"])

        inner_fold_records.extend(fold_result["inner_fold_records"])
        all_loss_records.extend(fold_result["all_loss_records"])
        all_parity_records.extend(fold_result["all_parity_records"])
        individual_model_test_metrics.extend(fold_result["individual_model_test_metrics"])

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    final_r2 = r2_score(all_targets, all_predictions)
    final_rmse = np.sqrt(np.mean((all_targets - all_predictions) ** 2))
    final_mae = np.mean(np.abs(all_targets - all_predictions))

    print(f"\n{'=' * 60}")
    print(f"ABLATION RESULTS: {ablation_name}")
    print(f"{'=' * 60}")
    print(f"R²:   {final_r2:.3f}")
    print(f"RMSE: {final_rmse:.3f}")
    print(f"MAE:  {final_mae:.3f}")

    pd.DataFrame(inner_fold_records).to_csv(
        os.path.join(ablation_dir, "inner_fold_results.csv"), index=False, float_format="%.3f"
    )
    pd.DataFrame(outer_fold_metrics).to_csv(
        os.path.join(ablation_dir, "outer_fold_metrics.csv"), index=False, float_format="%.3f"
    )
    pd.DataFrame(all_loss_records).to_csv(
        os.path.join(ablation_dir, "all_loss_data.csv"), index=False
    )
    pd.DataFrame(all_parity_records).to_csv(
        os.path.join(ablation_dir, "all_parity_data.csv"), index=False, float_format="%.3f"
    )
    pd.DataFrame(individual_model_test_metrics).to_csv(
        os.path.join(ablation_dir, "individual_model_test_metrics.csv"), index=False, float_format="%.3f"
    )

    predictions_df = pd.DataFrame(
        {
            "row_id": all_row_ids,
            "actual_lce": all_targets,
            "predicted_lce": all_predictions,
            "error": all_targets - all_predictions,
            "abs_error": np.abs(all_targets - all_predictions),
        }
    )
    predictions_df.to_csv(
        os.path.join(ablation_dir, "nested_cv_predictions.csv"), index=False, float_format="%.3f"
    )

    metrics = {
        "ablation": ablation_name,
        "r2": round(float(final_r2), 3),
        "rmse": round(float(final_rmse), 3),
        "mae": round(float(final_mae), 3),
        "n_samples": len(all_targets),
    }
    with open(os.path.join(ablation_dir, "nested_cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_parity(
        all_targets,
        all_predictions,
        {"r2": final_r2, "rmse": final_rmse},
        os.path.join(ablation_dir, "nested_cv_parity_plot.png"),
        title_prefix=ablation_name,
    )

    return metrics


def main(ablation_keys=None):
    """
    Run all ablation studies or specified ones.

    Args:
        ablation_keys: List of ablation keys to run, or None to run all
    """
    mp.set_start_method("spawn", force=True)

    set_seed(MASTER_SEED)
    start_time = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    if n_gpus == 0:
        raise RuntimeError("This training entry point requires at least one CUDA GPU.")

    results_dir = get_results_dir(base_results_dir)

    log_path = os.path.join(results_dir, "ablation.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"Results: {results_dir}")
    print(f"Dataset: {DATA_PATH}")

    process_id = os.getpid()
    preprocessed_path = smiles_dict_preprocessing(DATA_PATH, process_id=process_id)
    map_dict, vocab_size, moiety_key_map, key_moiety_map = load_preprocessed_data(
        preprocessed_path
    )

    ce_data = load_dataset(DATA_PATH)
    print(f"Dataset: {len(ce_data)} samples")

    if ablation_keys is None:
        ablation_keys = list(ABLATION_CONFIGS.keys())

    all_ablation_metrics = []

    for key in ablation_keys:
        if key not in ABLATION_CONFIGS:
            print(f"Warning: Unknown ablation key '{key}', skipping")
            continue

        config = ABLATION_CONFIGS[key]
        ablation_dir = os.path.join(results_dir, key)
        os.makedirs(ablation_dir, exist_ok=True)

        metrics = run_ablation_single(
            ce_data.copy(),
            map_dict,
            vocab_size,
            moiety_key_map,
            key_moiety_map,
            config["model_class"],
            config["name"],
            ablation_dir,
            n_gpus,
        )
        all_ablation_metrics.append(metrics)

    summary_df = pd.DataFrame(all_ablation_metrics)
    summary_df.to_csv(os.path.join(results_dir, "ablation_summary.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))

    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    get_running_time(start_time)
    print(f"\nResults saved to: {results_dir}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
