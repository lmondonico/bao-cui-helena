"""CE dataset paths, columns, and molecule-count limits."""

import os
from config.paths import DATA_DIR

INFERENCE_DATA_DIR = os.path.join(DATA_DIR, "inference")

# CE Dataset - fixed path
CE_DATASET_PATH = os.path.join(DATA_DIR, "HELENA-CE-26.csv")

# CE Dataset Configuration Constants
MAX_SOLVENTS = 5
MAX_SALTS = 3
MAX_MOLECULES = MAX_SOLVENTS + MAX_SALTS
TARGET_COLUMN = "LCE"

REQUIRED_COLUMNS = ["Row_ID", "CE", "LCE"]
OPTIONAL_COLUMNS = ["Method", "Current", "Capacity", "Cycle", "Reference"]

SOLVENT_COLUMNS = [f"Solvent_{i}" for i in range(1, MAX_SOLVENTS + 1)]
SALT_COLUMNS = [f"Salt_{i}" for i in range(1, MAX_SALTS + 1)]


def get_ce_dataset():
    """Get the CE dataset path."""
    if not os.path.exists(CE_DATASET_PATH):
        raise FileNotFoundError(f"CE dataset not found: {CE_DATASET_PATH}")
    print(f"Dataset: {os.path.basename(CE_DATASET_PATH)}")
    return CE_DATASET_PATH


def get_inference_dataset(filename):
    """Get an inference dataset path by filename.

    Args:
        filename: Name of the file in the inference directory (e.g., 'my_data.csv')
    """
    filepath = os.path.join(INFERENCE_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Inference dataset not found: {filepath}")
    print(f"Inference dataset: {filename}")
    return filepath
