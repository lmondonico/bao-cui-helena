"""
Create chunks from large inference CSV file for memory-efficient processing.

Splits large datasets into smaller chunks for parallel/batched inference.
"""

import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.datasets import get_inference_dataset, INFERENCE_DATA_DIR
from config.paths import get_inference_chunks_dir
from core.utils import load_file_auto
from core.check_dataset import validate_inference_dataset


def create_inference_chunks(input_file_path, chunk_size=100000, pid=None):
    """Split large inference file into smaller chunks.

    Args:
        input_file_path: Path to the inference file (.csv or .parquet)
        chunk_size: Number of rows per chunk (default: 100000)
        pid: Process ID for session isolation (optional)

    Returns:
        bool: True if successful
    """
    chunks_dir = Path(get_inference_chunks_dir(pid))
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_file_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Output: {chunks_dir}")

    try:
        print("\nLoading dataset...")
        full_df = load_file_auto(input_file_path)
        total_rows = len(full_df)
        print(f"Total rows: {total_rows:,}")

        print("\nCreating chunks...")
        chunk_counter = 1
        rows_processed = 0

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = full_df.iloc[start_idx:end_idx]

            chunk_filename = f"chunk_{chunk_counter:03d}.csv"
            chunk_df.to_csv(chunks_dir / chunk_filename, index=False)

            rows_processed += len(chunk_df)
            print(f"  {chunk_filename}: {len(chunk_df):,} rows")
            chunk_counter += 1

        # Save chunk info
        chunk_info = {
            "source_dataset": input_file_path,
            "total_chunks": chunk_counter - 1,
            "chunk_size": chunk_size,
            "total_rows": rows_processed,
            "chunks_directory": str(chunks_dir),
            "process_id": pid if pid is not None else os.getpid(),
        }

        with open(chunks_dir / "chunk_info.txt", "w") as f:
            for key, value in chunk_info.items():
                f.write(f"{key}: {value}\n")

        print(f"\nCreated {chunk_counter - 1} chunks")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main(dataset_filename):
    """Create chunks from inference dataset.

    Args:
        dataset_filename: Name of the dataset file in data/inference/ directory
    """
    pid = os.getpid()

    print("=" * 60)
    print("CHUNK CREATION FOR HELENA-CE INFERENCE")
    print("=" * 60)
    print(f"Process ID: {pid}\n")

    data_path = get_inference_dataset(dataset_filename)
    dataset_name = os.path.basename(data_path)
    print(f"Dataset: {dataset_name}")
    print(f"Path: {data_path}\n")

    # Validate dataset
    print("Validating dataset...")
    try:
        validate_inference_dataset(data_path)
    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        return

    # Create chunks
    success = create_inference_chunks(data_path, chunk_size=100000, pid=pid)

    if success:
        print("\n" + "=" * 60)
        print("CHUNKING COMPLETE")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Chunks directory: chunks_{pid}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create chunks for inference")
    parser.add_argument(
        "dataset",
        help=f"Dataset filename in {INFERENCE_DATA_DIR}",
    )
    args = parser.parse_args()
    main(args.dataset)
