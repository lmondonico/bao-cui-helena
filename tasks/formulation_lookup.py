"""
Information retrieval for CE electrolyte dataset based on Row IDs.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.datasets import get_ce_dataset


def load_dataset_simple(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return None

    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def lookup_reference(df, row_id_number):
    try:
        row_id_number = int(row_id_number)
    except ValueError:
        return None, "Invalid Row_ID. Please enter a valid integer."

    matching_rows = df[df["Row_ID"] == row_id_number]

    if matching_rows.empty:
        return None, f"No entry found for Row_ID: {row_id_number}"

    row = matching_rows.iloc[0]

    info = {
        "Row_ID": row["Row_ID"],
        "Electrolyte": row["Electrolyte"] if "Electrolyte" in row else "N/A",
        "Reference": row["Reference"] if "Reference" in row else "N/A",
        "CE": row["CE"] if "CE" in row else None,
        "LCE": row["LCE"] if "LCE" in row else None,
        "Method": row["Method"] if "Method" in row else None,
        "Current": row["Current"] if "Current" in row else None,
        "Capacity": row["Capacity"] if "Capacity" in row else None,
        "Cycle": row["Cycle"] if "Cycle" in row else None,
    }

    solvents = {}
    salts = {}

    for i in range(1, 6):
        col_name = f"Solvent_{i}"
        if col_name in row and pd.notna(row[col_name]) and row[col_name].strip():
            solvents[i] = row[col_name]

    for i in range(1, 4):
        col_name = f"Salt_{i}"
        if col_name in row and pd.notna(row[col_name]) and row[col_name].strip():
            salts[i] = row[col_name]

    info["Solvents"] = solvents
    info["Salts"] = salts

    return info, None


def parse_smiles_molar_ratio(entry):
    if ":" in entry:
        smiles, molar_ratio = entry.split(":", 1)
        return f"{smiles.strip()} - {molar_ratio.strip()}"
    else:
        return entry.strip()


def display_info(info):
    print("\n" + "=" * 80)
    print(f"ROW ID: {info['Row_ID']}")
    print("=" * 80)
    print(f"Electrolyte: {info['Electrolyte']}")

    print(f"\nEfficiency Data:")
    if info.get("CE") is not None:
        print(f"  CE: {info['CE']}%")
    if info.get("LCE") is not None:
        print(f"  LCE: {info['LCE']}")

    if info["Solvents"]:
        print(f"\nSolvents:")
        for solvent_num, entry in info["Solvents"].items():
            formatted_entry = parse_smiles_molar_ratio(entry)
            print(f"  Solvent {solvent_num}: {formatted_entry}")

    if info["Salts"]:
        print(f"\nSalts:")
        for salt_num, entry in info["Salts"].items():
            formatted_entry = parse_smiles_molar_ratio(entry)
            print(f"  Salt {salt_num}: {formatted_entry}")

    print(f"\nExperimental Conditions:")
    if info.get("Method"):
        print(f"  Method: {info['Method']}")
    if info.get("Current"):
        print(f"  Current: {info['Current']} mA/cm²")
    if info.get("Capacity"):
        print(f"  Capacity: {info['Capacity']} mAh/cm²")
    if info.get("Cycle"):
        print(f"  Cycle: {info['Cycle']}")

    print(f"\nReference: {info['Reference']}")
    print("=" * 80)


def main():
    print("-" * 50)
    print("CE Electrolyte Database - Row_ID Lookup")
    print("Type 'quit', 'exit', or 'q' to stop the program")
    print("-" * 50)

    dataset_path = get_ce_dataset()
    df = load_dataset_simple(dataset_path)
    if df is None:
        return

    print("Dataset type: Coulombic Efficiency (CE)")
    print(f"Dataset loaded successfully! Contains {len(df)} entries.")
    print(f"Row_ID range: {df['Row_ID'].min()} - {df['Row_ID'].max()}")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nEnter Row_ID number (or 'quit' to exit): ").strip()

            if user_input.lower() in ["quit", "exit", "q", ""]:
                break

            info, error = lookup_reference(df, user_input)

            if error:
                print(f"Error: {error}")
            else:
                display_info(info)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
