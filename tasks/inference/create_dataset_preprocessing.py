"""
Create preprocessing file for HELENA-CE inference.

Creates a preprocessed molecule information file (.npy) that can be reused
across multiple inference runs with the same molecules.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.paths import (
    INFERENCE_DIR,
    get_reference_model_path,
    get_inference_preprocessing_path,
)
from config.datasets import get_inference_dataset, INFERENCE_DATA_DIR
from core.utils import (
    extract_training_mappings_from_checkpoint,
    create_inference_preprocessing,
)
from core.check_dataset import validate_inference_dataset


def main(dataset_filename):
    """Create dataset preprocessing NPY file.

    Args:
        dataset_filename: Name of the dataset file in data/inference/ directory
    """
    pid = os.getpid()

    print("=" * 60)
    print("DATASET PREPROCESSING FOR HELENA-CE INFERENCE")
    print("=" * 60)
    print(f"Process ID: {pid}\n")

    # Load model and dataset
    reference_model_path = Path(get_reference_model_path())
    print(f"Model: {reference_model_path.name}")

    data_path = get_inference_dataset(dataset_filename)
    dataset_name = os.path.basename(data_path)
    print(f"Dataset: {dataset_name}")
    print(f"Path: {data_path}\n")

    start_time = time.time()

    # Extract mappings from model
    print("Extracting mappings from model...")
    moiety_key_map, key_moiety_map, input_vocab_size, _ = (
        extract_training_mappings_from_checkpoint(str(reference_model_path))
    )
    print(f"Vocabulary size: {input_vocab_size}")

    # Validate dataset
    print("\nValidating dataset...")
    validate_inference_dataset(data_path)

    # Create preprocessing file
    print("\nCreating preprocessing file...")
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

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Output: {final_path.name}")
        print(f"Time: {total_time:.2f}s")
    else:
        print("ERROR: Failed to create preprocessing file")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create preprocessing for inference")
    parser.add_argument(
        "dataset",
        help=f"Dataset filename in {INFERENCE_DATA_DIR}",
    )
    args = parser.parse_args()
    main(args.dataset)
