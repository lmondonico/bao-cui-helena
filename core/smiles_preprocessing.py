"""
Preprocessing of every molecule in the dataset before training. This saved memory and time during training.
"""

import os
import sys
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.moiety_features import (
    map_atom_to_moiety,
    build_moiety_edges,
    get_adj_matrix,
    get_dist_matrix,
)
from modules.tokenizer.tokenizer_helena import Tokenizer
from config.datasets import MAX_SOLVENTS, MAX_SALTS

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def smiles_dict_preprocessing(data_path, process_id=None):
    print("Preprocessing molecules SMILES strings...")
    print("Using tokenizer: Helena")
    print("Dataset type: Coulombic Efficiency (CE)")
    if process_id:
        print(f"Process ID: {process_id}")

    max_num_solvent_molecules = MAX_SOLVENTS
    max_num_salt_molecules = MAX_SALTS

    # Load dataset with flexible dtypes
    dtypes = {}
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str
    dtypes["Row_ID"] = int

    # CE dataset columns
    dtypes["Method"] = str
    dtypes["CE"] = float
    dtypes["LCE"] = float

    # Use robust CSV reading settings to handle formatting issues
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

    # Extract SMILES from SMILES:molar-ratio format
    for mol_col in solvent_columns + salt_columns:
        if mol_col in dataset.columns:
            for value in dataset[mol_col].dropna():
                if pd.notna(value) and value != "":
                    # Parse SMILES:molar-ratio format
                    try:
                        parts = str(value).split(":")
                        if len(parts) == 2:
                            smiles = parts[0].strip()
                            if smiles:
                                unique_SMILES_in_dataset.append(smiles)
                    except (ValueError, AttributeError):
                        pass  # Skip invalid entries

    unique_SMILES_in_toy_dataset = sorted(set(unique_SMILES_in_dataset))
    print(f"Total unique SMILES to preprocess: {len(unique_SMILES_in_toy_dataset)}")

    tokenizer = Tokenizer()

    # Create a dictionary to store all preprocessed information for each unique SMILES
    all_mols_dict = {
        smi: {
            "adj_matrix": None,
            "dist_matrix": None,
            "clique_atom_ids": None,
            "edges": None,
            "moiety_ids": None,
            "is_in_ring": None,
        }
        for smi in unique_SMILES_in_toy_dataset
    }

    # Iterate over each unique SMILES and preprocess
    for smiles in tqdm.tqdm(unique_SMILES_in_toy_dataset, desc="Preprocessing"):
        try:
            # Tokenize the SMILES string
            result = tokenizer.get_moieties(smiles)
            clique_atom_ids, edges, moiety_ids, matches, is_in_ring = (
                result[0], result[1], result[2], result[3], result[4]
            )

            atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
            moiety_edges = build_moiety_edges(edges, atom_to_moiety)

            # Node list of moiety indices
            node_list = list(range(len(clique_atom_ids)))  # moiety indices
            moiety_edges = list(moiety_edges)  # Edges between moieties

            # Generate distance and adjacency matrices
            dist_matrix = get_dist_matrix(node_list, moiety_edges)
            adj_matrix = get_adj_matrix(node_list, moiety_edges)

            # Store the preprocessed information in the dictionary
            all_mols_dict[smiles]["adj_matrix"] = adj_matrix
            all_mols_dict[smiles]["dist_matrix"] = dist_matrix
            all_mols_dict[smiles]["clique_atom_ids"] = clique_atom_ids
            all_mols_dict[smiles]["edges"] = edges
            all_mols_dict[smiles]["moiety_ids"] = moiety_ids
            all_mols_dict[smiles]["is_in_ring"] = is_in_ring

        except Exception as e:
            print(f"Error processing mol {smiles}: {e}")
            # Handle errors by assigning default or placeholder values
            all_mols_dict[smiles]["adj_matrix"] = np.array([[]], dtype=np.float32)
            all_mols_dict[smiles]["dist_matrix"] = np.array([[]], dtype=np.float32)
            all_mols_dict[smiles]["clique_atom_ids"] = []
            all_mols_dict[smiles]["edges"] = []
            all_mols_dict[smiles]["moiety_ids"] = []
            all_mols_dict[smiles]["is_in_ring"] = []

    # Add moiety dictionaries to the saved data
    preprocessed_data = {
        "molecules": all_mols_dict,
        "moiety_key_map": tokenizer.moiety_key_map,
        "key_moiety_map": tokenizer.key_moiety_map,
        "vocab_size": max(tokenizer.moiety_key_map.values()) + 1
        if tokenizer.moiety_key_map
        else 1,
    }

    # Define the output path for the preprocessed data
    from config.paths import DATA_DIR

    if process_id:
        output_path = os.path.join(
            DATA_DIR, f"preprocessed_mol_info_{process_id}.npy"
        )
    else:
        output_path = os.path.join(
            DATA_DIR, f"preprocessed_mol_info.npy"
        )
    np.save(output_path, preprocessed_data)
    print("+" * 60)
    print("Molecules preprocessing COMPLETE!")
    print("+" * 60, "\n")

    return output_path


if __name__ == "__main__":
    from config.datasets import get_ce_dataset

    data_path = get_ce_dataset()
    print("Using Helena tokenizer")
    smiles_dict_preprocessing(data_path)
