"""
Process high-CE predictions to generate moiety intermediate pkl files.
Extracts moiety patterns from binary and ternary solvent formulations.
Uses multiprocessing for speed.
"""

########################
## Parameters
HIGH_CE_FILENAME = "high_ce_predictions.csv"
PROCESS_BINARY = True
PROCESS_TERNARY = True
NUM_WORKERS = None  # None = use all available cores
########################

import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from config.paths import DATA_DIR
from config.datasets import MAX_SOLVENTS, MAX_SALTS
from modules.tokenizer.tokenizer_helena import Tokenizer
from tasks.postprocess.postprocess_utils.excluded_moieties import EXCLUDED_MOIETIES

# Directory paths
INFERENCE_POSTPROCESS_DIR = os.path.join(DATA_DIR, "inference_postprocess")
HIGH_CE_PATH = os.path.join(INFERENCE_POSTPROCESS_DIR, HIGH_CE_FILENAME)
BINARY_OUTPUT_DIR = os.path.join(INFERENCE_POSTPROCESS_DIR, "binary")
TERNARY_OUTPUT_DIR = os.path.join(INFERENCE_POSTPROCESS_DIR, "ternary")

# Salt name mapping for display
SALT_NAMES = {
    "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F": "FSI",
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F": "TFSI",
    "[Li+].F[P-](F)(F)(F)(F)F": "PF6",
    "[Li+].[B-]1(OC(=O)C(=O)O1)(F)F": "DFOB",
}


def get_salt_name(salt_smiles):
    """Convert salt SMILES to a short name if available."""
    return SALT_NAMES.get(salt_smiles, salt_smiles[:15] + "..." if len(salt_smiles) > 15 else salt_smiles)


def parse_smiles_ratio(value):
    """Parse SMILES:molar_ratio format and return (smiles, ratio)."""
    if pd.isna(value) or value is None:
        return None, 0.0
    try:
        parts = str(value).split(":")
        if len(parts) == 2:
            smiles = parts[0].strip()
            ratio = float(parts[1].strip())
            return smiles, ratio
        else:
            return str(value).strip(), 1.0
    except Exception:
        return None, 0.0


def extract_moieties_from_smiles(smiles, tokenizer, key_moiety_map):
    """Extract moiety names from a SMILES string."""
    try:
        result = tokenizer.get_moieties(smiles)
        if result is None:
            return []
        _, _, moiety_ids, *_ = result
        moiety_names = [key_moiety_map.get(mid, f"m_{mid}") for mid in moiety_ids]
        moiety_names = [m for m in moiety_names if m not in EXCLUDED_MOIETIES]
        return moiety_names
    except Exception:
        return []


def load_high_ce_data():
    """Load high-CE predictions CSV."""
    if not os.path.exists(HIGH_CE_PATH):
        print(f"ERROR: High CE predictions file not found: {HIGH_CE_PATH}")
        print("Please run run_inference_postprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(HIGH_CE_PATH)
    print(f"Loaded {len(df)} high-CE formulations")
    return df


def count_solvents_row(row_dict):
    """Count the number of solvents with non-zero molar ratio."""
    count = 0
    for i in range(1, MAX_SOLVENTS + 1):
        col = f"Solvent_{i}"
        if col in row_dict:
            smiles, ratio = parse_smiles_ratio(row_dict[col])
            if smiles and ratio > 0:
                count += 1
    return count


def get_salts_info_dict(row_dict):
    """Get list of (smiles, ratio) for all salts."""
    salts = []
    for i in range(1, MAX_SALTS + 1):
        col = f"Salt_{i}"
        if col in row_dict:
            smiles, ratio = parse_smiles_ratio(row_dict[col])
            if smiles and ratio > 0:
                salts.append((smiles, ratio))
    return salts


def get_solvents_info_dict(row_dict, max_solvents):
    """Get list of (smiles, ratio) for solvents up to max_solvents."""
    solvents = []
    for i in range(1, max_solvents + 1):
        col = f"Solvent_{i}"
        if col in row_dict:
            smiles, ratio = parse_smiles_ratio(row_dict[col])
            if smiles and ratio > 0:
                solvents.append((smiles, ratio))
    return solvents


def get_salt_combo_key(salts):
    """Generate salt combination key and display name."""
    if len(salts) == 0:
        return ("none", "no_salt", "No Salt")
    elif len(salts) == 1:
        salt_smiles = salts[0][0]
        display_name = get_salt_name(salt_smiles)
        return ("single", salt_smiles, display_name)
    elif len(salts) == 2:
        salt_names = [(get_salt_name(s[0]), s[0]) for s in salts]
        salt_names.sort(key=lambda x: x[0])
        salt_key = f"{salt_names[0][1]}_{salt_names[1][1]}"
        display_name = f"{salt_names[0][0]} + {salt_names[1][0]}"
        return ("binary", salt_key, display_name)
    else:
        sorted_salts = sorted([s[0] for s in salts])
        salt_key = "_".join(sorted_salts)
        display_name = " + ".join([get_salt_name(s) for s in sorted_salts])
        return ("complex", salt_key, display_name)


# ============================================================
# Worker functions for multiprocessing
# ============================================================

def init_worker():
    """Initialize worker with tokenizer."""
    global worker_tokenizer, worker_key_moiety_map
    worker_tokenizer = Tokenizer()
    worker_key_moiety_map = worker_tokenizer.key_moiety_map


def process_binary_row(row_tuple):
    """Process a single binary formulation row. Returns processed data or None."""
    idx, row_dict = row_tuple
    global worker_tokenizer, worker_key_moiety_map

    solvents = get_solvents_info_dict(row_dict, MAX_SOLVENTS)
    salts = get_salts_info_dict(row_dict)

    if len(solvents) != 2:
        return None

    sol1_smiles, sol1_ratio = solvents[0]
    sol2_smiles, sol2_ratio = solvents[1]

    moieties1 = extract_moieties_from_smiles(sol1_smiles, worker_tokenizer, worker_key_moiety_map)
    moieties2 = extract_moieties_from_smiles(sol2_smiles, worker_tokenizer, worker_key_moiety_map)

    if not moieties1 or not moieties2:
        return None

    salt_combo, salt_key, display_key = get_salt_combo_key(salts)
    full_key = (salt_combo, salt_key, display_key)

    moiety_pairs = []
    for m1 in moieties1:
        for m2 in moieties2:
            moiety_pairs.append({"moiety1": m1, "moiety2": m2, "idx": idx})

    return {
        "idx": idx,
        "full_key": full_key,
        "salt_combo_key": (salt_combo, salt_key),
        "salts": [s[0] for s in salts],
        "moiety_pairs": moiety_pairs,
        "molecules": {sol1_smiles: moieties1, sol2_smiles: moieties2},
    }


def process_ternary_row(row_tuple):
    """Process a single ternary formulation row. Returns processed data or None."""
    idx, row_dict = row_tuple
    global worker_tokenizer, worker_key_moiety_map

    solvents = get_solvents_info_dict(row_dict, MAX_SOLVENTS)
    salts = get_salts_info_dict(row_dict)

    if len(solvents) != 3:
        return None

    sol1_smiles, sol1_ratio = solvents[0]
    sol2_smiles, sol2_ratio = solvents[1]
    sol3_smiles, sol3_ratio = solvents[2]

    moieties1 = extract_moieties_from_smiles(sol1_smiles, worker_tokenizer, worker_key_moiety_map)
    moieties2 = extract_moieties_from_smiles(sol2_smiles, worker_tokenizer, worker_key_moiety_map)
    moieties3 = extract_moieties_from_smiles(sol3_smiles, worker_tokenizer, worker_key_moiety_map)

    if not moieties1 or not moieties2 or not moieties3:
        return None

    salt_combo, salt_key, display_key = get_salt_combo_key(salts)
    full_key = (salt_combo, salt_key, display_key)

    triplet_record = {
        "moieties1": moieties1,
        "moieties2": moieties2,
        "moieties3": moieties3,
        "solvent1": sol1_smiles,
        "solvent2": sol2_smiles,
        "solvent3": sol3_smiles,
        "conc1": sol1_ratio,
        "conc2": sol2_ratio,
        "conc3": sol3_ratio,
        "idx": idx,
    }

    return {
        "idx": idx,
        "full_key": full_key,
        "salt_combo_key": (salt_combo, salt_key),
        "salts": [s[0] for s in salts],
        "triplet_record": triplet_record,
        "molecules": {
            sol1_smiles: moieties1,
            sol2_smiles: moieties2,
            sol3_smiles: moieties3,
        },
    }


def count_solvents_wrapper(row_tuple):
    """Wrapper for counting solvents in parallel."""
    idx, row_dict = row_tuple
    return idx, count_solvents_row(row_dict)


# ============================================================
# Main processing functions
# ============================================================

def process_binary_formulations(df, num_workers):
    """Process binary solvent formulations using multiprocessing."""
    print("\n" + "=" * 50)
    print("PROCESSING BINARY FORMULATIONS")
    print("=" * 50)

    # Convert DataFrame to list of (idx, dict) tuples for multiprocessing
    print("Preparing data for parallel processing...")
    rows_data = [(idx, row.to_dict()) for idx, row in df.iterrows()]

    # Count solvents in parallel
    print("Counting solvents in parallel...")
    with mp.Pool(processes=num_workers) as pool:
        solvent_counts = list(tqdm(
            pool.imap(count_solvents_wrapper, rows_data, chunksize=1000),
            total=len(rows_data),
            desc="Counting solvents"
        ))

    # Filter for binary (exactly 2 solvents)
    binary_indices = {idx for idx, count in solvent_counts if count == 2}
    binary_rows = [(idx, row_dict) for idx, row_dict in rows_data if idx in binary_indices]
    print(f"Found {len(binary_rows)} binary solvent formulations")

    if len(binary_rows) == 0:
        print("No binary formulations to process.")
        return

    # Process binary formulations in parallel
    print("Processing binary formulations in parallel...")
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_binary_row, binary_rows, chunksize=500),
            total=len(binary_rows),
            desc="Processing binary"
        ))

    # Aggregate results
    print("Aggregating results...")
    salt_combinations = defaultdict(list)
    all_salts = []
    moiety_pair_data = {}
    processed_molecules = {}

    for result in results:
        if result is None:
            continue

        idx = result["idx"]
        full_key = result["full_key"]
        salt_combo_key = result["salt_combo_key"]

        salt_combinations[salt_combo_key].append(idx)
        all_salts.extend(result["salts"])

        if full_key not in moiety_pair_data:
            moiety_pair_data[full_key] = []
        moiety_pair_data[full_key].extend(result["moiety_pairs"])

        for smiles, moieties in result["molecules"].items():
            if smiles not in processed_molecules:
                processed_molecules[smiles] = moieties

    # Print statistics
    print(f"\nSalt combination statistics:")
    for (combo, key), indices in salt_combinations.items():
        if combo == "single":
            display = get_salt_name(key)
        elif combo == "binary":
            parts = key.split("_")
            display = f"{get_salt_name(parts[0])} + {get_salt_name(parts[1])}"
        else:
            display = key
        print(f"  - {combo.capitalize()} '{display}': {len(indices)} formulations")

    # Save outputs
    os.makedirs(BINARY_OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(BINARY_OUTPUT_DIR, "salt_combinations.pkl"), "wb") as f:
        pickle.dump(dict(salt_combinations), f)
    with open(os.path.join(BINARY_OUTPUT_DIR, "all_salts.pkl"), "wb") as f:
        pickle.dump(all_salts, f)
    with open(os.path.join(BINARY_OUTPUT_DIR, "moiety_pair_data.pkl"), "wb") as f:
        pickle.dump(moiety_pair_data, f)
    with open(os.path.join(BINARY_OUTPUT_DIR, "processed_molecules.pkl"), "wb") as f:
        pickle.dump(processed_molecules, f)

    print(f"\nBinary intermediates saved to: {BINARY_OUTPUT_DIR}")


def process_ternary_formulations(df, num_workers, solvent_counts=None):
    """Process ternary solvent formulations using multiprocessing."""
    print("\n" + "=" * 50)
    print("PROCESSING TERNARY FORMULATIONS")
    print("=" * 50)

    # Convert DataFrame to list of (idx, dict) tuples
    print("Preparing data for parallel processing...")
    rows_data = [(idx, row.to_dict()) for idx, row in df.iterrows()]

    # Count solvents in parallel if not already done
    if solvent_counts is None:
        print("Counting solvents in parallel...")
        with mp.Pool(processes=num_workers) as pool:
            solvent_counts = list(tqdm(
                pool.imap(count_solvents_wrapper, rows_data, chunksize=1000),
                total=len(rows_data),
                desc="Counting solvents"
            ))

    # Filter for ternary (exactly 3 solvents)
    ternary_indices = {idx for idx, count in solvent_counts if count == 3}
    ternary_rows = [(idx, row_dict) for idx, row_dict in rows_data if idx in ternary_indices]
    print(f"Found {len(ternary_rows)} ternary solvent formulations")

    if len(ternary_rows) == 0:
        print("No ternary formulations to process.")
        return

    # Process ternary formulations in parallel
    print("Processing ternary formulations in parallel...")
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_ternary_row, ternary_rows, chunksize=500),
            total=len(ternary_rows),
            desc="Processing ternary"
        ))

    # Aggregate results
    print("Aggregating results...")
    salt_combinations = defaultdict(list)
    all_salts = []
    moiety_triplet_data = {}
    processed_molecules = {}

    for result in results:
        if result is None:
            continue

        idx = result["idx"]
        full_key = result["full_key"]
        salt_combo_key = result["salt_combo_key"]

        salt_combinations[salt_combo_key].append(idx)
        all_salts.extend(result["salts"])

        if full_key not in moiety_triplet_data:
            moiety_triplet_data[full_key] = []
        moiety_triplet_data[full_key].append(result["triplet_record"])

        for smiles, moieties in result["molecules"].items():
            if smiles not in processed_molecules:
                processed_molecules[smiles] = moieties

    # Print statistics
    print(f"\nSalt combination statistics:")
    for (combo, key), indices in salt_combinations.items():
        if combo == "single":
            display = get_salt_name(key)
        elif combo == "binary":
            parts = key.split("_")
            display = f"{get_salt_name(parts[0])} + {get_salt_name(parts[1])}"
        else:
            display = key
        print(f"  - {combo.capitalize()} '{display}': {len(indices)} formulations")

    # Save outputs
    os.makedirs(TERNARY_OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(TERNARY_OUTPUT_DIR, "salt_combinations.pkl"), "wb") as f:
        pickle.dump(dict(salt_combinations), f)
    with open(os.path.join(TERNARY_OUTPUT_DIR, "all_salts.pkl"), "wb") as f:
        pickle.dump(all_salts, f)
    with open(os.path.join(TERNARY_OUTPUT_DIR, "moiety_triplet_data.pkl"), "wb") as f:
        pickle.dump(moiety_triplet_data, f)
    with open(os.path.join(TERNARY_OUTPUT_DIR, "processed_molecules.pkl"), "wb") as f:
        pickle.dump(processed_molecules, f)

    print(f"\nTernary intermediates saved to: {TERNARY_OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("GENERATE MOIETY INTERMEDIATES (MULTICORE)")
    print("=" * 60)

    # Determine number of workers
    num_workers = NUM_WORKERS if NUM_WORKERS else mp.cpu_count()
    print(f"Using {num_workers} CPU cores")

    # Load data
    df = load_high_ce_data()

    # Process formulations
    if PROCESS_BINARY:
        process_binary_formulations(df, num_workers)

    if PROCESS_TERNARY:
        process_ternary_formulations(df, num_workers)

    print("\n" + "=" * 60)
    print("INTERMEDIATE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Binary outputs: {BINARY_OUTPUT_DIR}")
    print(f"Ternary outputs: {TERNARY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
