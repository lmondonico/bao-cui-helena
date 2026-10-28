"""
Checking the integrity of a CE electrolyte dataset CSV file.
"""

import pandas as pd
from rdkit import Chem
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.datasets import (
    get_ce_dataset,
    TARGET_COLUMN,
    REQUIRED_COLUMNS,
    SOLVENT_COLUMNS,
    SALT_COLUMNS,
)


def validate_dataset(file_path: str):
    if not file_path.endswith(".csv"):
        print(f"Error: Provided file '{file_path}' is not a CSV file.")
        sys.exit(1)

    try:
        dataset = pd.read_csv(file_path, header=0, low_memory=False)
        print("File loaded successfully.\n")
        print(f"Target column: {TARGET_COLUMN}")
    except FileNotFoundError:
        print(f"Fatal Error: The file was not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal Error: Failed to read CSV file: {e}")
        sys.exit(1)

    row_id_ok = check_row_ids(dataset)
    smiles_ok = check_smiles_strings(dataset)
    columns_ok = check_required_columns(dataset)
    target_values_ok = check_target_values(dataset)

    print("\n--- Validation Summary ---")
    if row_id_ok and smiles_ok and columns_ok and target_values_ok:
        print("All checks passed. The dataset appears to be valid.")
        return True
    else:
        print("Validation failed. Please review the errors listed above.")
        return False


def check_row_ids(df: pd.DataFrame) -> bool:
    print("\n[1/4] Checking 'Row_ID' column...")
    is_ok = True

    if df["Row_ID"].isnull().any():
        is_ok = False
        missing_indices = df[df["Row_ID"].isnull()].index.tolist()
        line_numbers = [i + 2 for i in missing_indices]
        print("\n" + "-" * 50)
        print(f"ERROR: Found {len(line_numbers)} row(s) with a missing 'Row_ID'.")
        print(f"Problematic line number(s): {line_numbers}")
        print("-" * 50)
    else:
        print("  > No missing 'Row_ID' values found.")

    valid_row_ids_df = df.dropna(subset=["Row_ID"])
    duplicates = valid_row_ids_df[valid_row_ids_df["Row_ID"].duplicated(keep=False)]

    if not duplicates.empty:
        is_ok = False
        print("\n" + "-" * 50)
        print("ERROR: Found duplicate 'Row_ID' values.")
        for code, group in duplicates.groupby("Row_ID"):
            line_numbers = [i + 2 for i in group.index.tolist()]
            print(f"  - Row_ID '{int(code)}' is duplicated on lines: {line_numbers}")
        print("-" * 50)
    else:
        print("  > No duplicate 'Row_ID' values found.")

    return is_ok


def check_smiles_strings(df: pd.DataFrame) -> bool:
    print("\n[2/4] Checking SMILES strings for chemical validity...")

    all_molecule_cols = SOLVENT_COLUMNS + SALT_COLUMNS

    smiles_to_check = set()

    for col_name in all_molecule_cols:
        if col_name in df.columns:
            for value in df[col_name].dropna():
                smiles = str(value).split(":")[0].strip()
                if smiles:
                    smiles_to_check.add(smiles)

    if not smiles_to_check:
        print("  > Warning: No SMILES strings were found to validate.")
        return True

    print(f"  > Found {len(smiles_to_check)} unique SMILES strings to validate...")

    invalid_smiles = []
    for smiles in smiles_to_check:
        if Chem.MolFromSmiles(smiles) is None:
            invalid_smiles.append(smiles)

    if not invalid_smiles:
        print("  > All SMILES strings are chemically valid.")
        return True
    else:
        print("\n" + "-" * 50)
        print(f"ERROR: Found {len(invalid_smiles)} invalid SMILES string(s):")
        for smiles in invalid_smiles:
            print(f"  - '{smiles}'")
        print("-" * 50)
        return False


def check_required_columns(df: pd.DataFrame) -> bool:
    print("\n[3/4] Checking required columns...")

    missing_columns = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        print(f"  > ERROR: Missing required columns: {missing_columns}")
        return False
    else:
        print(f"  > All required columns present: {REQUIRED_COLUMNS}")
        return True


def check_target_values(df: pd.DataFrame) -> bool:
    print("\n[4/4] Checking target values...")

    is_ok = True
    missing_values = []

    for idx, row in df.iterrows():
        line_num = idx + 2  # +2 because CSV index starts at 0 and we have a header
        row_id = row.get("Row_ID")
        missing_cols = []

        if pd.isna(row.get("CE")) or row.get("CE") == "":
            missing_cols.append("CE")
        if pd.isna(row.get("LCE")) or row.get("LCE") == "":
            missing_cols.append("LCE")

        if missing_cols:
            missing_values.append((line_num, row_id, missing_cols))
            is_ok = False

    if missing_values:
        print("\n" + "-" * 50)
        print(f"ERROR: Found {len(missing_values)} row(s) with missing target values.")
        for line_num, row_id, missing_cols in missing_values:
            print(f"  - Line {line_num} (Row_ID {row_id}): missing {', '.join(missing_cols)}")
        print("-" * 50)
    else:
        print("  > All rows have valid 'CE' and 'LCE' values.")

    return is_ok


if __name__ == "__main__":
    data_path = get_ce_dataset()
    sys.exit(0 if validate_dataset(data_path) else 1)
