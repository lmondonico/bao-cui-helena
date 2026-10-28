"""
Pipeline script to run all HELENA-CE inference steps in sequence.

This script runs preprocessing, chunking, and inference using the trained
HELENA-CE model weights (helena_ce_trained_weights.pth).
"""

#################################
# CONFIGURATION
DATASET_FILENAME = "perturbation_study.csv"
#################################

import sys
import os
import time
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.paths import (
    INFERENCE_DIR,
    get_reference_model_path,
    get_inference_preprocessing_path,
    get_inference_chunks_dir,
)
from config.datasets import get_inference_dataset, INFERENCE_DATA_DIR
from core.utils import (
    extract_training_mappings_from_checkpoint,
    create_inference_preprocessing,
    load_file_auto,
)
from core.check_dataset import validate_inference_dataset
import importlib.util


def main(dataset_filename):
    """Run the complete inference pipeline.

    Args:
        dataset_filename: Name of the dataset file in data/inference/ directory
    """
    pid = os.getpid()

    print("=" * 80)
    print("HELENA-CE INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Process ID: {pid}")
    print("\nSteps:")
    print("  1. Dataset preprocessing")
    print("  2. Chunk creation")
    print("  3. Inference execution")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 1: Load dataset and model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATASET AND MODEL")
    print("=" * 80 + "\n")

    data_path = get_inference_dataset(dataset_filename)
    dataset_name = os.path.basename(data_path)
    print(f"Dataset: {dataset_name}")
    print(f"Path: {data_path}\n")

    reference_model_path = get_reference_model_path()
    print(f"Model: {os.path.basename(reference_model_path)}")

    # Extract mappings from model checkpoint
    moiety_key_map, key_moiety_map, input_vocab_size, checkpoint = (
        extract_training_mappings_from_checkpoint(reference_model_path)
    )
    print(f"Vocab size: {input_vocab_size}")

    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING")
    print("=" * 80 + "\n")

    start_time = time.time()

    output_path = create_inference_preprocessing(
        data_path,
        moiety_key_map,
        key_moiety_map,
        input_vocab_size,
        str(INFERENCE_DIR),
        "dataset",
    )

    if output_path and os.path.exists(output_path):
        final_path = Path(get_inference_preprocessing_path(pid))
        if output_path != str(final_path):
            os.rename(output_path, final_path)

        preprocessing_time = time.time() - start_time
        print(f"Preprocessing complete ({preprocessing_time:.2f}s)")
        print(f"File: {final_path.name}")
    else:
        raise RuntimeError("Failed to create preprocessing file")

    # =========================================================================
    # STEP 3: Chunking
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: CHUNK CREATION")
    print("=" * 80 + "\n")

    chunk_start_time = time.time()

    # Validate dataset
    print("Validating dataset...")
    try:
        validate_inference_dataset(data_path)
    except Exception as e:
        raise RuntimeError(f"Dataset validation failed: {e}") from e

    # Create chunks
    chunks_dir = Path(get_inference_chunks_dir(pid))
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = 100000
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Output: {chunks_dir}")

    try:
        df = load_file_auto(data_path)
        total_rows = len(df)
        print(f"Dataset rows: {total_rows:,}")

        chunk_counter = 1
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_filename = chunks_dir / f"chunk_{chunk_counter:04d}.csv"
            chunk_df.to_csv(chunk_filename, index=False)
            print(f"  Chunk {chunk_counter}: rows {start_idx:,}-{end_idx:,}")
            chunk_counter += 1

        # Save chunk info
        chunk_info = {
            "source_dataset": data_path,
            "total_chunks": chunk_counter - 1,
            "chunk_size": chunk_size,
            "total_rows": total_rows,
            "chunks_directory": str(chunks_dir),
            "process_id": pid,
        }
        with open(chunks_dir / "chunk_info.txt", "w") as f:
            for key, value in chunk_info.items():
                f.write(f"{key}: {value}\n")

        chunk_time = time.time() - chunk_start_time
        print(f"\nChunking complete ({chunk_time:.2f}s)")
        print(f"Created {chunk_counter - 1} chunks")

    except Exception as e:
        raise RuntimeError(f"Chunk creation failed: {e}") from e

    # =========================================================================
    # STEP 4: Inference
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: INFERENCE")
    print("=" * 80 + "\n")

    inference_start_time = time.time()

    # Import and run inference
    inference_module_path = os.path.join(INFERENCE_DIR, "inference_module.py")
    spec = importlib.util.spec_from_file_location(
        "inference_module", inference_module_path
    )
    inference_module = importlib.util.module_from_spec(spec)
    sys.modules["inference_module"] = inference_module
    spec.loader.exec_module(inference_module)

    inference_module.main(
        reference_model_path=reference_model_path, checkpoint=checkpoint
    )

    inference_time = time.time() - inference_start_time

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nDataset: {dataset_name}")
    print(f"\nTiming:")
    print(f"  Preprocessing: {preprocessing_time:.2f}s")
    print(f"  Chunking:      {chunk_time:.2f}s")
    print(f"  Inference:     {inference_time:.2f}s")
    print(f"  Total:         {total_time:.2f}s ({total_time / 60:.2f} min)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HELENA inference pipeline")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DATASET_FILENAME,
        help=f"Dataset filename in {INFERENCE_DATA_DIR}/ (default: {DATASET_FILENAME})",
    )
    args = parser.parse_args()
    main(args.dataset)
