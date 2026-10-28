"""Repository paths for datasets, checkpoints, and generated outputs."""

import os
from pathlib import Path

# Automatically calculate PROJECT_ROOT based on this file's location
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

# Core directories
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Inference directories
INFERENCE_DIR = os.path.join(TASKS_DIR, "inference")
INFERENCE_CHUNKS_DIR = os.path.join(INFERENCE_DIR, "chunks")


def get_reference_model_path():
    """
    Returns the path to the trained HELENA-CE model weights.

    Returns:
        str: Path to the model file
    """
    model_path = os.path.join(DATA_DIR, "helena_ce_trained_weights.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Please place 'helena_ce_trained_weights.pth' in {DATA_DIR}"
        )

    return model_path


def get_inference_chunks_dir(pid=None):
    """
    Get PID-based chunks directory for parallel inference sessions.

    Args:
        pid: Process ID (optional). If None, uses current process ID.

    Returns:
        Path to PID-isolated chunks directory
    """
    if pid is None:
        pid = os.getpid()
    return os.path.join(INFERENCE_DIR, f"chunks_{pid}")


def get_inference_preprocessing_path(pid=None):
    """
    Get PID-based preprocessing file path for parallel inference sessions.

    Args:
        pid: Process ID (optional). If None, uses current process ID.

    Returns:
        Path to PID-isolated preprocessing .npy file
    """
    if pid is None:
        pid = os.getpid()
    return os.path.join(INFERENCE_DIR, f"dataset_preprocessing_{pid}.npy")
