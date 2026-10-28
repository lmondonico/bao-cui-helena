"""
Dataset validation utilities for HELENA-CE.
"""

import pandas as pd
from rdkit import Chem
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.datasets import (
    SOLVENT_COLUMNS,
    SALT_COLUMNS,
    MAX_SOLVENTS,
    MAX_SALTS,
)


def validate_inference_dataset(file_path: str) -> pd.DataFrame:
    """
    Validate an inference dataset and return it with any necessary defaults.

    For inference datasets, we don't require CE/LCE columns since we're predicting them.
    We do require Row_ID and valid SMILES strings.

    Args:
        file_path: Path to the CSV/Parquet file

    Returns:
        DataFrame with validated data and default values for missing optional columns
    """
    # Load file
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        import csv

        df = pd.read_csv(
            file_path,
            header=0,
            low_memory=False,
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
            on_bad_lines="skip",
        )
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    print(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")

    # Check for Row_ID column
    if "Row_ID" not in df.columns:
        print("  Creating Row_ID column (sequential integers)")
        df["Row_ID"] = range(len(df))

    # Check for Method column (default: Aurbach = 0)
    if "Method" not in df.columns:
        print("  Adding default Method column (Aurbach = 0)")
        df["Method"] = 0
    else:
        # Convert string methods to numeric
        df["Method"] = df["Method"].fillna("Aurbach")
        df["Method"] = df["Method"].map(lambda x: 0 if str(x).lower() == "aurbach" else 1)

    # Check for Current column (default: 0.5 mA/cm^2)
    if "Current" not in df.columns:
        print("  Adding default Current column (0.5 mA/cm^2)")
        df["Current"] = 0.5
    else:
        df["Current"] = df["Current"].fillna(0.5)

    # Validate SMILES strings
    all_molecule_cols = SOLVENT_COLUMNS + SALT_COLUMNS
    invalid_smiles = []

    for col_name in all_molecule_cols:
        if col_name in df.columns:
            for idx, value in df[col_name].items():
                if pd.notna(value):
                    smiles = str(value).split(":")[0].strip()
                    if smiles and Chem.MolFromSmiles(smiles) is None:
                        invalid_smiles.append((col_name, idx, smiles))

    if invalid_smiles:
        print(f"\n  WARNING: Found {len(invalid_smiles)} invalid SMILES strings:")
        for col, idx, smiles in invalid_smiles[:10]:  # Show first 10
            print(f"    - Row {idx}, {col}: '{smiles}'")
        if len(invalid_smiles) > 10:
            print(f"    ... and {len(invalid_smiles) - 10} more")
    else:
        print("  All SMILES strings validated successfully")

    # Add placeholder LCE/CE columns if not present (for compatibility)
    if "LCE" not in df.columns:
        df["LCE"] = 0.0
    if "CE" not in df.columns:
        df["CE"] = 0.0

    print(f"  Validation complete: {len(df)} rows ready for inference")
    return df
