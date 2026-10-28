"""
INFERENCE MODULE - HELENA-CE Model

Performs inference using trained HELENA-CE model to predict LCE (log Coulombic Efficiency)
for battery electrolyte formulations.
"""

#################################
CHUNK_BATCH_SIZE = 1024
GPU_ID = None  # Set to specific GPU ID or None for auto-selection
#################################

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import glob
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.paths import (
    INFERENCE_DIR,
    get_inference_chunks_dir,
    get_inference_preprocessing_path,
    get_reference_model_path,
)
from config.datasets import MAX_MOLECULES
from core.utils import (
    extract_training_mappings_from_checkpoint,
    load_preprocessed_data,
    get_unique_results_dir,
    setup_logging,
    stop_logging,
    load_dataset,
    feature_engineering,
)
from core.runtime_utils import get_least_used_gpu
from modules.helena_ce.helena_ce_model import HELENA_CE
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE
from modules.dataset_processing import CollateFn


def process_chunk(chunk_path, map_dict, model, device, batch_size):
    """Process a single chunk and return predictions."""
    print(f"Processing: {chunk_path.name}")

    chunk_dataset = load_dataset(data_path=str(chunk_path))

    [chunk_inference_dataset] = feature_engineering(
        map_dict,
        0,  # No augmentation during inference
        chunk_dataset,
        label_field=None,
    )

    collate_fn = CollateFn(MAX_MOLECULES)
    chunk_loader = torch.utils.data.DataLoader(
        chunk_inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    predictions = []
    row_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(chunk_loader, desc=f"Inference - {chunk_path.name}"):
            row_id_batch = batch["row_id"]
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch)
            preds = outputs["ce"].cpu().numpy().flatten()
            predictions.extend(preds)
            row_ids.extend(row_id_batch.numpy())

    del chunk_dataset, chunk_inference_dataset, chunk_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(predictions), np.array(row_ids)


def main(reference_model_path=None, checkpoint=None):
    """Run inference on chunked dataset."""
    pid = os.getpid()

    # Set seeds for deterministic inference
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    results_dir = get_unique_results_dir("HELENA_CE_Inference")
    os.makedirs(results_dir, exist_ok=True)

    logger, file_handler = setup_logging(results_dir, "inference.log")

    print("=" * 60)
    print("HELENA-CE INFERENCE")
    print("=" * 60)
    print(f"Process ID: {pid}")
    print("=" * 60 + "\n")

    start_time = time.time()

    # GPU selection
    if GPU_ID is not None:
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
        elif GPU_ID >= torch.cuda.device_count():
            print(f"GPU {GPU_ID} not available, using auto-selection")
            device = torch.device(f"cuda:{get_least_used_gpu()}")
        else:
            device = torch.device(f"cuda:{GPU_ID}")
            print(f"Using GPU: {device}")
    else:
        device = (torch.device(f"cuda:{get_least_used_gpu()}")
                  if torch.cuda.is_available() else torch.device("cpu"))
        print(f"Using device: {device}")

    # Reserve GPU memory early to distribute inference across GPUs
    if device != torch.device("cpu"):
        print(f"Reserving GPU memory on {device}...")
        gpu_reserve_tensor = torch.zeros(100_000_000, dtype=torch.float32).to(device)
    else:
        gpu_reserve_tensor = None

    # Load checkpoint
    if checkpoint is None:
        if reference_model_path is None:
            reference_model_path = get_reference_model_path()

        if not os.path.exists(reference_model_path):
            print(f"Model not found: {reference_model_path}")
            sys.exit(1)

        print(f"Loading model: {os.path.basename(reference_model_path)}")
        _, _, input_vocab_size, checkpoint = extract_training_mappings_from_checkpoint(
            reference_model_path
        )
    else:
        print("Using provided checkpoint")
        input_vocab_size = checkpoint["vocab_size"]

    # Load preprocessing data
    dataset_npy_path = get_inference_preprocessing_path(pid)
    if os.path.exists(dataset_npy_path):
        print(f"Preprocessing file: {os.path.basename(dataset_npy_path)}")
        map_dict, _, _, _ = load_preprocessed_data(custom_path=dataset_npy_path)
    else:
        print(f"ERROR: Preprocessing file not found: {dataset_npy_path}")
        sys.exit(1)

    # Load chunks
    chunks_dir = get_inference_chunks_dir(pid)
    if not os.path.exists(chunks_dir):
        print(f"Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.csv")))
    if not chunk_files:
        print(f"No chunk files found in: {chunks_dir}")
        sys.exit(1)

    # Read dataset info
    chunk_info_path = os.path.join(chunks_dir, "chunk_info.txt")
    dataset_name = "Unknown"
    if os.path.exists(chunk_info_path):
        with open(chunk_info_path, "r") as f:
            for line in f:
                if line.startswith("source_dataset:"):
                    dataset_name = os.path.basename(line.split(":", 1)[1].strip())
                    break

    print(f"\nDataset: {dataset_name}")
    print(f"Chunks: {len(chunk_files)}")

    # Model configuration
    task_type = checkpoint.get("task_type", "regression")
    num_classes = checkpoint.get("num_classes", None)

    print(f"\nModel config:")
    print(f"  Task: {task_type}")
    print(f"  Vocab size: {input_vocab_size}")
    print(f"  Max molecules: {MAX_MOLECULES}")

    # Initialize model
    model = HELENA_CE(
        **ARCHITECTURE,
        input_vocab_size=input_vocab_size,
        max_num_molecules=MAX_MOLECULES,
        task_type=task_type,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Release GPU reservation now that model is loaded
    if gpu_reserve_tensor is not None:
        del gpu_reserve_tensor
        torch.cuda.empty_cache()

    # Process chunks
    temp_dir = os.path.join(results_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    temp_files = []

    for chunk_file in chunk_files:
        chunk_path = Path(chunk_file)
        preds, row_ids = process_chunk(
            chunk_path, map_dict, model, device, CHUNK_BATCH_SIZE
        )

        if task_type == "regression":
            chunk_results = pd.DataFrame({
                "Row_ID": row_ids,
                "LCE_Predicted": preds,
                "CE_Predicted": 100 * (1 - 10 ** (-preds)),
            })
        else:
            chunk_results = pd.DataFrame({
                "Row_ID": row_ids,
                "CE_Class": preds.astype(int),
            })

        temp_file = os.path.join(temp_dir, f"temp_{chunk_path.stem}.csv")
        chunk_results.to_csv(temp_file, index=False)
        temp_files.append(temp_file)
        print(f"Completed {chunk_path.name}: {len(preds)} predictions")

    # Combine results
    all_results = [pd.read_csv(f) for f in temp_files]
    output_df = pd.concat(all_results, ignore_index=True)

    # Save outputs
    output_df.to_csv(os.path.join(results_dir, "inference_output.csv"), index=False)
    try:
        output_df.to_parquet(os.path.join(results_dir, "inference_output.parquet"), index=False)
    except ImportError:
        print("Note: Parquet output skipped (pyarrow/fastparquet not installed). CSV saved.")

    # Cleanup
    for temp_file in temp_files:
        os.remove(temp_file)
    os.rmdir(temp_dir)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"\nDataset: {dataset_name}")
    print(f"Results: {results_dir}")
    print(f"Predictions: {len(output_df)}")
    print(f"Time: {total_time/60:.2f} min")
    print("=" * 60)

    stop_logging(logger, file_handler)


if __name__ == "__main__":
    main()
