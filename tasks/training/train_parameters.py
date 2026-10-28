"""Training parameters for HELENA-CE model."""

import torch
import numpy as np
import pandas as pd
from collections import Counter

# Training hyperparameters
batch_size = 32
num_epochs = 600
learning_rate = 7e-5
learning_rate_classification = 5e-5
weight_decay = 0.08
molar_ratio_noise = 0.05
early_stopping_patience = 80

# Nested ensemble parameters
N_OUTER_FOLDS = 5
N_ENSEMBLE = 8

# Classification threshold
LCE_THRESHOLD_BINARY = 2.0

# Stratification
RARE_MOIETY_THRESHOLD = 0.05

# Learning rate scheduler
warmup_epochs = 60
plateau_patience = 15
plateau_factor = 0.5
plateau_min_lr = 5e-8


def setup_optimizer(model, lr=learning_rate):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def setup_scheduler(optimizer, warmup=warmup_epochs, max_lr=learning_rate):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / 80, end_factor=1.0, total_iters=warmup
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=plateau_factor,
        patience=plateau_patience,
        min_lr=plateau_min_lr,
    )
    return warmup_scheduler, plateau_scheduler, warmup


def convert_lce_to_binary_class(lce_values, threshold=LCE_THRESHOLD_BINARY):
    if isinstance(lce_values, torch.Tensor):
        return (lce_values >= threshold).long()
    return (np.array(lce_values) >= threshold).astype(np.int64)


def get_combined_stratification_groups(df, tokenizer):
    """Stratify by current density bins and rare moiety presence."""
    from config.datasets import SOLVENT_COLUMNS, SALT_COLUMNS

    # Extract moieties per sample
    sample_moieties = {}
    for idx, row in df.iterrows():
        moieties = set()
        for col in SOLVENT_COLUMNS + SALT_COLUMNS:
            if col in df.columns and pd.notna(row[col]):
                value = str(row[col])
                if ":" in value:
                    smiles = value.split(":")[0].strip()
                    if smiles:
                        try:
                            result = tokenizer.get_moieties(smiles)
                            if result and len(result) >= 4:
                                for mid in result[2]:
                                    name = tokenizer.key_moiety_map.get(
                                        mid, f"Unk_{mid}"
                                    )
                                    moieties.add(name)
                        except Exception:
                            pass
        sample_moieties[idx] = moieties

    # Find rare moieties
    moiety_counts = Counter()
    for moieties in sample_moieties.values():
        moiety_counts.update(moieties)
    rare = {m for m, c in moiety_counts.items() if c / len(df) < RARE_MOIETY_THRESHOLD}

    print(
        f"Stratification: {len(rare)} rare moieties (<{RARE_MOIETY_THRESHOLD * 100:.0f}%)"
    )

    # Build groups: current_bin * 2 + has_rare
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
