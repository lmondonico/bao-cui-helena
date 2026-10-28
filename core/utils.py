"""
This module contains utility functions for the regression task.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.imports import *
from modules.dataset_processing import *
from modules.moiety_features import *
from modules.tokenizer.tokenizer_helena import Tokenizer
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU selection will use PyTorch memory only.")


def get_running_time(start_time):
    """Calculate the running time of a script."""

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60

    print("\n" + "/" * 30)
    print(f"PROCESS COMPLETED in {elapsed_minutes:.2f} minutes")
    print("/" * 30 + "\n")

    return


def get_unique_results_dir(base_name):
    """
    Create a unique results directory using PID to prevent overlapping results.
    """
    from config.paths import RESULTS_DIR

    pid = os.getpid()
    unique_dir = os.path.join(RESULTS_DIR, f"{base_name}_{pid}")
    return unique_dir


def copy_parameter_files(results_dir):
    """
    Copy the contents of current parameter files into a single txt file.
    """
    from config.paths import PROJECT_ROOT

    param_files = [
        os.path.join(PROJECT_ROOT, "tasks/training/train_parameters.py"),
        os.path.join(PROJECT_ROOT, "modules/helena_ce/helena_ce_parameters.py"),
        os.path.join(PROJECT_ROOT, "tasks/training/train_loss_function.py"),
    ]
    output_path = os.path.join(results_dir, "training_config.txt")

    try:
        with open(output_path, "w") as output_file:
            for file_path in param_files:
                if not os.path.exists(file_path):
                    print(f"Warning: Parameter file not found: {file_path}")
                    continue

                output_file.write(f"{'=' * 80}\n")
                output_file.write(f"FILE: {file_path}\n")
                output_file.write(f"{'=' * 80}\n\n")

                with open(file_path, "r") as input_file:
                    output_file.write(input_file.read())
                output_file.write("\n\n")

        print(f"Parameter files copied to: {output_path}")
    except Exception as e:
        print(f"Error copying parameter files: {e}")


def setup_logging(results_dir, filename="task_terminal_output.log"):
    """
    Set up logging to capture terminal output to a file in the results directory.
    """
    import logging
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(results_dir, filename)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    class StreamToLogger:
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ""
            self.terminal = sys.stdout

        def write(self, buf):
            self.terminal.write(buf)
            self.linebuf += buf
            if buf.endswith("\n"):
                self.logger.log(self.log_level, self.linebuf.rstrip())
                self.linebuf = ""

        def flush(self):
            if self.linebuf:
                self.logger.log(self.log_level, self.linebuf)
                self.linebuf = ""
            self.terminal.flush()

    sys.stdout = StreamToLogger(logger)

    print(f"Logging experiment output to: {log_file}")

    return logger, file_handler


def stop_logging(logger, file_handler):
    """
    Stop logging and restore normal stdout.
    """
    import sys

    sys.stdout = sys.__stdout__

    logger.removeHandler(file_handler)
    file_handler.close()

    print(f"Logging stopped and file closed")


def save_preprocessed_file(source_path, results_dir):
    """Move preprocessed file to results directory with original filename."""
    import shutil

    original_filename = os.path.basename(source_path)
    dest_path = os.path.join(results_dir, original_filename)
    shutil.move(source_path, dest_path)
    print(f"Saved preprocessed file to: {dest_path}")
    return dest_path


def load_file_auto(file_path):
    """
    Load file automatically based on extension (CSV or Parquet).

    Args:
        file_path (str): Path to file (.csv or .parquet)

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        import csv

        return pd.read_csv(
            file_path,
            header=0,
            low_memory=False,
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
            on_bad_lines="skip",
        )
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. Only .csv and .parquet are supported."
        )


def load_dataset(data_path):
    """Load CE dataset with proper column handling."""
    from config.datasets import MAX_SOLVENTS, MAX_SALTS

    dtypes = {}

    # Molecule columns
    for i in range(1, MAX_SOLVENTS + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, MAX_SALTS + 1):
        dtypes[f"Salt_{i}"] = str

    # CE dataset columns
    dtypes["Method"] = str
    dtypes["CE"] = float
    dtypes["LCE"] = float

    # Optional common columns
    dtypes["Reference"] = str
    dtypes["Electrolyte"] = str

    # Load file based on extension
    if data_path.endswith(".parquet"):
        dataset = pd.read_parquet(data_path)
        # Convert dtypes for parquet
        for col, dtype in dtypes.items():
            if col in dataset.columns:
                dataset[col] = dataset[col].astype(dtype)
    else:
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

    return dataset


def load_preprocessed_data(custom_path=None):
    """Load dataset and preprocessed data."""

    try:
        if custom_path:
            preprocess_path = custom_path
        else:
            preprocess_path = "data/preprocessed_mol_info.npy"

        preprocessed_data = np.load(preprocess_path, allow_pickle=True).item()

        if "molecules" in preprocessed_data:
            map_dict = preprocessed_data["molecules"]
            input_vocab_size = preprocessed_data.get("vocab_size", 1)
            moiety_key_map = preprocessed_data.get("moiety_key_map", {})
            key_moiety_map = preprocessed_data.get("key_moiety_map", {})
        else:
            map_dict = preprocessed_data
            input_vocab_size = 1 + max(
                max(data["moiety_ids"])
                for data in map_dict.values()
                if "moiety_ids" in data
            )
            moiety_key_map = {}
            key_moiety_map = {}

        return map_dict, input_vocab_size, moiety_key_map, key_moiety_map
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        sys.exit(1)


def feature_engineering(
    map_dict,
    molar_ratio_noise,
    *database,
    label_field="LCE",
):
    """Perform feature engineering on CE datasets.

    Args:
        map_dict: Mapping dictionary for molecular features
        molar_ratio_noise: Noise level for molar ratios
        *database: Variable number of datasets to process
        label_field: Label field name (default: "LCE"). Pass None for inference mode (no labels needed).
    """
    tokenizer = Tokenizer()
    processed_datasets = []

    for data in database:
        processed_dataset = FeatureEngineering(
            data,
            tokenizer,
            map_dict,
            label_field=label_field,
            molar_ratio_noise=molar_ratio_noise,
        )
        processed_datasets.append(processed_dataset)

    return processed_datasets


def create_data_loaders(
    *datasets,
    max_num_molecules,
    batch_size,
    shuffle_default=False,
    sampler_default=None,
    drop_last=True,
):
    """Create data loaders for training, validation, and test sets."""

    collate_function = CollateFn(max_num_mols=max_num_molecules)
    processed_dataloaders = []

    for dataset in datasets:
        processed_dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_default,
            sampler=sampler_default,
            collate_fn=collate_function,
            num_workers=0,
            pin_memory=False,
            drop_last=drop_last,
        )
        processed_dataloaders.append(processed_dataset)

    return processed_dataloaders


def calculate_metrics(*results, show_results=False, show_prediction_ranges=False):
    """Calculate RMSE and R² metrics for a fold."""

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
            "pearson_r": pearsonr(y_true, y_pred)[0],
        }
        metrics[task] = task_metrics

        if show_results:
            print(f"\n{task}")
            print(
                f"\tLoss: {result['loss']:.4f}, RMSE: {task_metrics['rmse']:.3f}, R2: {task_metrics['r2']:.3f}, Pearson r: {task_metrics['pearson_r']:.3f}"
            )

        if show_prediction_ranges:
            print(f"\tPrediction ranges:\t[{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
            print(f"\tTarget ranges:\t\t[{np.min(y_true):.2f}, {np.max(y_true):.2f}]")

    return metrics


def extract_training_mappings_from_checkpoint(checkpoint_path):
    """Extract moiety mappings from enhanced model checkpoint"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    moiety_key_map = checkpoint["moiety_key_map"]
    key_moiety_map = checkpoint["key_moiety_map"]
    vocab_size = checkpoint["vocab_size"]
    return moiety_key_map, key_moiety_map, vocab_size, checkpoint


def print_model_parameters(model, model_name="Model", results_dir=None):
    """
    Print model parameter count summary including total, trainable, and breakdown by component.
    Optionally saves to a txt file in the results directory.

    Args:
        model: PyTorch model
        model_name: Name to display in the header
        results_dir: If provided, saves the summary to model_parameters.txt in this directory
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"{model_name} PARAMETER SUMMARY")
    lines.append(f"{'=' * 60}")
    lines.append(f"Total parameters:     {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append(f"")
    lines.append(f"Breakdown by component:")
    lines.append(f"{'-' * 60}")

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        pct = 100.0 * params / total_params if total_params > 0 else 0
        lines.append(f"  {name:30s} {params:>12,}  ({pct:5.1f}%)")

    lines.append(f"{'=' * 60}")

    # Save to file if results_dir provided
    if results_dir is not None:
        output_path = os.path.join(results_dir, "model_parameters.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))


def process_single_smiles_worker(args):
    """
    Worker function for parallel SMILES processing.

    Args:
        args: Tuple of (smiles, moiety_key_map, key_moiety_map)

    Returns:
        Tuple of (smiles, processed_data_dict) or (smiles, None) if error
    """
    smiles, moiety_key_map, key_moiety_map = args

    try:
        # Import inside worker to avoid serialization issues
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from modules.tokenizer.tokenizer_helena import Tokenizer
        from modules.moiety_features import (
            map_atom_to_moiety,
            build_moiety_edges,
            get_dist_matrix,
            get_adj_matrix,
        )

        # Reconstruct tokenizer in worker process
        tokenizer = Tokenizer()
        tokenizer.moiety_key_map = moiety_key_map
        tokenizer.key_moiety_map = key_moiety_map

        # Process the SMILES molecule
        result = tokenizer.get_moieties(smiles)
        clique_atom_ids, edges, moiety_ids = result[0], result[1], result[2]

        atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
        moiety_edges = build_moiety_edges(edges, atom_to_moiety)

        node_list = list(range(len(clique_atom_ids)))
        moiety_edges = list(moiety_edges)

        dist_matrix = get_dist_matrix(node_list, moiety_edges)
        adj_matrix = get_adj_matrix(node_list, moiety_edges)

        processed_data = {
            "adj_matrix": adj_matrix,
            "dist_matrix": dist_matrix,
            "clique_atom_ids": clique_atom_ids,
            "edges": edges,
            "moiety_ids": moiety_ids,
        }

        return smiles, processed_data

    except Exception as e:
        # Return error case
        import numpy as np

        error_data = {
            "adj_matrix": np.array([[]], dtype=np.float32),
            "dist_matrix": np.array([[]], dtype=np.float32),
            "clique_atom_ids": [],
            "edges": [],
            "moiety_ids": [],
        }
        print(f"Error processing mol {smiles}: {e}")
        return smiles, error_data


def create_inference_preprocessing(
    inference_data_path,
    moiety_key_map,
    key_moiety_map,
    vocab_size,
    results_dir,
    process_id,
):
    """Create inference NPY using training moiety mappings"""
    print("Creating inference preprocessing with training mappings...")

    from config.datasets import MAX_SOLVENTS, MAX_SALTS

    max_num_solvent_molecules = MAX_SOLVENTS
    max_num_salt_molecules = MAX_SALTS

    dtypes = {}
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str
    dtypes["Row_ID"] = int

    df = load_file_auto(inference_data_path)
    # Apply dtype conversions after loading
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    unique_SMILES_in_dataset = set()
    for col in [f"Solvent_{i}" for i in range(1, max_num_solvent_molecules + 1)] + [
        f"Salt_{i}" for i in range(1, max_num_salt_molecules + 1)
    ]:
        smiles_values = df[col].dropna().astype(str)
        smiles_values = smiles_values[smiles_values != "nan"]
        for smiles_string in smiles_values:
            if ":" in smiles_string:
                smiles_part = smiles_string.split(":")[0]
                unique_SMILES_in_dataset.add(smiles_part)

    # Initialize all_mols_dict
    all_mols_dict = {
        smi: {
            "adj_matrix": None,
            "dist_matrix": None,
            "clique_atom_ids": None,
            "edges": None,
            "moiety_ids": None,
        }
        for smi in unique_SMILES_in_dataset
    }

    # Prepare arguments for multiprocessing
    unique_smiles_list = list(unique_SMILES_in_dataset)
    args_list = [
        (smiles, moiety_key_map, key_moiety_map) for smiles in unique_smiles_list
    ]

    # Determine optimal number of cores (leave some for system)
    num_cores = min(90, mp.cpu_count() - 6, len(unique_smiles_list))
    print(f"Using {num_cores} cores for parallel SMILES processing...")

    # Process SMILES molecules in parallel
    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_smiles_worker, args_list),
                    total=len(args_list),
                    desc=f"Preprocessing {len(args_list)} molecules (parallel)",
                )
            )

        # Collect results back into all_mols_dict
        for smiles, processed_data in results:
            if processed_data is not None:
                all_mols_dict[smiles] = processed_data

        print(
            f"Successfully processed {len(results)} molecules using {num_cores} cores"
        )

    except Exception as e:
        print(f"Multiprocessing failed, falling back to sequential processing: {e}")
        # Fallback to sequential processing
        tokenizer = Tokenizer()
        tokenizer.moiety_key_map = moiety_key_map
        tokenizer.key_moiety_map = key_moiety_map

        for smiles in tqdm(
            unique_SMILES_in_dataset,
            desc="Preprocessing inference molecules (sequential fallback)",
        ):
            try:
                result = tokenizer.get_moieties(smiles)
                clique_atom_ids, edges, moiety_ids = result[0], result[1], result[2]

                atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
                moiety_edges = build_moiety_edges(edges, atom_to_moiety)

                node_list = list(range(len(clique_atom_ids)))
                moiety_edges = list(moiety_edges)

                dist_matrix = get_dist_matrix(node_list, moiety_edges)
                adj_matrix = get_adj_matrix(node_list, moiety_edges)

                all_mols_dict[smiles]["adj_matrix"] = adj_matrix
                all_mols_dict[smiles]["dist_matrix"] = dist_matrix
                all_mols_dict[smiles]["clique_atom_ids"] = clique_atom_ids
                all_mols_dict[smiles]["edges"] = edges
                all_mols_dict[smiles]["moiety_ids"] = moiety_ids

            except Exception as e:
                print(f"Error processing mol {smiles}: {e}")
                all_mols_dict[smiles]["adj_matrix"] = np.array([[]], dtype=np.float32)
                all_mols_dict[smiles]["dist_matrix"] = np.array([[]], dtype=np.float32)
                all_mols_dict[smiles]["clique_atom_ids"] = []
                all_mols_dict[smiles]["edges"] = []
                all_mols_dict[smiles]["moiety_ids"] = []

    preprocessed_data = {
        "molecules": all_mols_dict,
        "moiety_key_map": moiety_key_map,
        "key_moiety_map": key_moiety_map,
        "vocab_size": vocab_size,
    }

    output_path = os.path.join(
        results_dir, f"preprocessed_mol_info_inference_{process_id}.npy"
    )
    np.save(output_path, preprocessed_data)
    print(f"Inference preprocessing saved to: {output_path}")
    return output_path
