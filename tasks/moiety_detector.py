"""
Moiety detection for electrolyte dataset based on SMILES representations.
"""

SMILES_TO_CHECK = [
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
]

from rdkit import Chem
from rdkit import RDLogger
import pandas as pd
import os
import sys
import glob
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.imports import *
from core.utils import (
    setup_logging,
    stop_logging,
)

from modules.dataset_processing import *
from modules.moiety_features import *
from modules.tokenizer.tokenizer_helena import Tokenizer
from config.datasets import get_ce_dataset, SOLVENT_COLUMNS, SALT_COLUMNS

RDLogger.DisableLog("rdApp.*")
import warnings

warnings.filterwarnings("ignore")

from config.paths import RESULTS_DIR

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_smiles_from_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    all_smiles = set()

    for col in SOLVENT_COLUMNS + SALT_COLUMNS:
        if col in df.columns:
            for value in df[col].dropna():
                if isinstance(value, str) and ":" in value:
                    smiles = value.split(":")[0]
                    if smiles.strip():
                        all_smiles.add(smiles.strip())

    return list(all_smiles)


def analyze_dataset_moieties(smiles_list, dataset_path=None):
    detector_results_dir = os.path.join(
        RESULTS_DIR, f"DatasetMoietyAnalysis_helena_{os.getpid()}"
    )
    os.makedirs(detector_results_dir, exist_ok=True)

    logger, file_handler = setup_logging(
        detector_results_dir, "dataset_moiety_analysis.log"
    )

    tokenizer = Tokenizer()
    all_moieties = []
    moiety_to_smiles = defaultdict(set)

    print(f"\nAnalyzing {len(smiles_list)} unique SMILES from dataset...")

    valid_count = 0
    for i, smiles in enumerate(smiles_list):
        try:
            result = tokenizer.get_moieties(smiles)
            if len(result) >= 4:
                _, _, moiety_ids, _ = result[:4]

                functional_group_names = [
                    tokenizer.key_moiety_map.get(key, f"Unknown_{key}")
                    for key in moiety_ids
                ]
                all_moieties.extend(functional_group_names)

                for moiety_name in functional_group_names:
                    moiety_to_smiles[moiety_name].add(smiles)

                valid_count += 1

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(smiles_list)} molecules...")
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")

    moiety_counts = Counter(all_moieties)

    print(f"\n{'=' * 60}")
    print(f"DATASET MOIETY ANALYSIS RESULTS")
    print(f"{'=' * 60}")
    dataset_name = os.path.basename(dataset_path) if dataset_path else "Unknown"
    print(f"Dataset: {dataset_name} (CE)")
    print(f"Total unique SMILES processed: {valid_count}")
    print(f"Total moieties found: {len(all_moieties)}")
    print(f"Unique moiety types: {len(moiety_counts)}")
    print(f"\nMoiety counts:")
    print(f"{'Moiety Name':<40} {'Count':<8} {'SMILES (for rare moieties)'}")
    print(f"{'-' * 80}")

    for moiety, count in moiety_counts.most_common():
        if count <= 2:
            smiles_list = list(moiety_to_smiles[moiety])
            smiles_str = ", ".join(smiles_list)
            print(f"{moiety:<40} {count:<8} {smiles_str}")
        else:
            print(f"{moiety:<40} {count:<8}")

    print(f"{'=' * 60}")

    stop_logging(logger, file_handler)


def detect_moieties(smiles_to_check):
    detector_results_dir = os.path.join(
        RESULTS_DIR, f"MoietyDetect_helena_{os.getpid()}"
    )
    os.makedirs(detector_results_dir, exist_ok=True)

    # Set up logging to capture terminal output
    logger, file_handler = setup_logging(
        detector_results_dir, "moiety_detector_output.log"
    )

    tokenizer = Tokenizer()
    for smiles in smiles_to_check:
        result = tokenizer.get_moieties(smiles)
        if result is None:
            print(f"\n[ERROR] Could not process SMILES: {smiles}")
            print("Skipping...")
            continue
        clique_atom_ids, edges, moiety_ids, matches = result[:4]

        functional_group_names = [
            tokenizer.key_moiety_map.get(key, "Unknown") for key in moiety_ids
        ]

        image_filename = f"{smiles}_helena.png"
        image_save_path = os.path.join(detector_results_dir, image_filename)

        tokenizer.highlight_and_display_molecule(
            Chem.MolFromSmiles(smiles), matches, save_path=image_save_path
        )

        print(f"\nMolecule SMILES: {smiles}")
        print("\nMoieties and Ring Systems (moieties/Cliques):")

        for fg_name, atom_indices in matches:
            print(f"- {fg_name}: atoms {sorted(atom_indices)}")

        print("\nAtoms per moiety:")
        print(clique_atom_ids)
        print("\nEdges:")
        print(edges)
        print("\nMoieties list:")
        print(moiety_ids)
        print("\nMoieties Names:")
        print(functional_group_names)
        print("\n" + "-" * 50 + "\n")

    stop_logging(logger, file_handler)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print("Using tokenizer: Helena")

    while True:
        test_dataset = (
            input("Do you want to test the entire dataset? (y/n): ").lower().strip()
        )
        if test_dataset in ["y", "yes"]:
            dataset_path = get_ce_dataset()
            smiles_list = extract_smiles_from_dataset(dataset_path)
            if smiles_list:
                analyze_dataset_moieties(smiles_list, dataset_path)
            else:
                print("No valid SMILES found in the dataset.")
            break
        elif test_dataset in ["n", "no"]:
            detect_moieties(SMILES_TO_CHECK)
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
