"""
Runtime profiling utilities for the HELENA-CE model.

Provides simple functions to measure:
- Model parameter count
- FLOPs (floating point operations)
- GPU memory usage
- Training/inference timing
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class RuntimeTracker:
    """Context manager for timing code sections."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_time = time.time() - self.start_time

    def get_elapsed_seconds(self) -> float:
        """Return elapsed time in seconds."""
        return self.elapsed_time if self.elapsed_time is not None else 0.0


def get_gpu_utilization(device_id):
    """Get GPU compute utilization percentage using pynvml."""
    if not PYNVML_AVAILABLE:
        return 0

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except Exception:
        return 0


def get_gpu_memory_usage(device_id):
    """Get GPU memory utilization percentage and details."""
    if not PYNVML_AVAILABLE:
        try:
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory
            usage_percent = (allocated / total) * 100 if total > 0 else 0
            return usage_percent, allocated, total
        except Exception:
            return 0, 0, 0

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_usage_percent = (memory_info.used / memory_info.total) * 100
        return memory_usage_percent, memory_info.used, memory_info.total
    except Exception:
        return 0, 0, 0


def calculate_gpu_score(device_id, memory_weight=0.4, utilization_weight=0.6):
    """Calculate combined GPU usage score (lower is better)."""
    memory_usage, _, _ = get_gpu_memory_usage(device_id)
    gpu_utilization = get_gpu_utilization(device_id)

    score = (memory_weight * memory_usage) + (utilization_weight * gpu_utilization)
    return score


def print_gpu_status(show_details=True):
    """Print current GPU status for debugging."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    num_gpus = torch.cuda.device_count()

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
        except Exception:
            print("NVML initialization failed")
            return

    print(f"\nGPU Status ({num_gpus} GPUs):")
    if show_details:
        print("GPU | Util% | Mem% | MemUsed/Total(GB) | Score")
        print("-" * 55)

    for i in range(num_gpus):
        try:
            memory_usage, mem_used, mem_total = get_gpu_memory_usage(i)
            gpu_util = get_gpu_utilization(i)
            score = calculate_gpu_score(i)

            if show_details:
                print(
                    f" {i}  | {gpu_util:4.1f} | {memory_usage:4.1f} | {mem_used / 1e9:5.2f}/{mem_total / 1e9:5.2f} | {score:5.1f}"
                )
            else:
                print(
                    f"GPU {i}: {gpu_util:4.1f}% util, {memory_usage:4.1f}% mem (score: {score:4.1f})"
                )

        except Exception as e:
            print(f"GPU {i}: Error - {e}")

    if show_details:
        print("-" * 55)


def get_least_used_gpu():
    """Find GPU with lowest combined memory + utilization score."""
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return 0

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
        except Exception:
            print("NVML initialization failed, falling back to memory-only selection")
            return get_least_used_gpu_fallback()

    best_gpu = 0
    best_score = float("inf")

    for i in range(num_gpus):
        try:
            if PYNVML_AVAILABLE:
                score = calculate_gpu_score(i)
            else:
                memory_used = torch.cuda.memory_allocated(i)
                score = memory_used

            if score < best_score:
                best_score = score
                best_gpu = i

        except Exception:
            continue

    return best_gpu


def get_least_used_gpu_fallback():
    """Fallback method using only PyTorch memory allocation."""
    num_gpus = torch.cuda.device_count()
    min_memory_used = float("inf")
    best_gpu = 0

    for i in range(num_gpus):
        try:
            memory_used = torch.cuda.memory_allocated(i)
            if memory_used < min_memory_used:
                min_memory_used = memory_used
                best_gpu = i
        except Exception:
            continue

    return best_gpu


def get_device(show_device=True, show_gpu_details=False):
    """Check if GPU is available and select least used one."""

    print("\n")
    print("+" * 60)
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if show_gpu_details:
                    print_gpu_status(show_details=True)

                device_id = get_least_used_gpu()
                device = torch.device(f"cuda:{device_id}")

                if show_device:
                    memory_usage, _, _ = get_gpu_memory_usage(device_id)
                    gpu_util = get_gpu_utilization(device_id)
                    score = calculate_gpu_score(device_id)
                    print(
                        f"Selected GPU {device_id}: {gpu_util:.1f}% util, {memory_usage:.1f}% mem (score: {score:.1f})"
                    )
                    print(f"Using device: {device}")
            else:
                device = torch.device("cuda")
                if show_device:
                    print(f"\nUsing device: {device} (single GPU)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            if show_device:
                print(f"\nUsing device: {device}")
        else:
            device = torch.device("cpu")
            if show_device:
                print(f"\nUsing device: {device}")

        print("+" * 60, "\n")
        return device
    except Exception as e:
        print(f"Error detecting device: {e}")
        sys.exit(1)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def count_parameters_by_component(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Count parameters by model component with percentage breakdown.

    Designed for HELENA_CE model architectures.

    Args:
        model: HELENA_CE PyTorch model

    Returns:
        Dictionary mapping component names to their parameter counts and percentages
    """

    def count_params(module):
        """Count trainable parameters in a module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    breakdown = {}
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check if model has the expected HELENA_CE components
    if hasattr(model, "encoder_moiety"):
        moiety_params = count_params(model.encoder_moiety)
        breakdown["moiety_encoder"] = {
            "params": moiety_params,
            "percent": round(100 * moiety_params / total_params, 1),
        }

    # Molecule Layer: transf_layer + transf_norm (8 linear + 8 layernorm)
    if hasattr(model, "transf_layer") and hasattr(model, "transf_norm"):
        molecule_layer_params = count_params(model.transf_layer) + count_params(
            model.transf_norm
        )
        breakdown["molecule_layer"] = {
            "params": molecule_layer_params,
            "percent": round(100 * molecule_layer_params / total_params, 1),
        }

    if hasattr(model, "molecular_attention"):
        formulation_params = count_params(model.molecular_attention)
        breakdown["formulation_attention"] = {
            "params": formulation_params,
            "percent": round(100 * formulation_params / total_params, 1),
        }

    if hasattr(model, "molar_ratio_encoder"):
        mr_params = count_params(model.molar_ratio_encoder)
        breakdown["molar_ratio_encoder"] = {
            "params": mr_params,
            "percent": round(100 * mr_params / total_params, 1),
        }

    if hasattr(model, "temperature_encoder"):
        temp_params = count_params(model.temperature_encoder)
        breakdown["temperature_encoder"] = {
            "params": temp_params,
            "percent": round(100 * temp_params / total_params, 1),
        }

    # Current encoder (for CE head in joint models)
    if hasattr(model, "current_encoder"):
        current_params = count_params(model.current_encoder)
        breakdown["current_encoder"] = {
            "params": current_params,
            "percent": round(100 * current_params / total_params, 1),
        }

    # Method encoder (for CE head in joint models)
    if hasattr(model, "method_encoder"):
        method_params = count_params(model.method_encoder)
        breakdown["method_encoder"] = {
            "params": method_params,
            "percent": round(100 * method_params / total_params, 1),
        }

    # Single prediction head (for HELENA_CE)
    if hasattr(model, "prediction_head"):
        head_params = count_params(model.prediction_head)
        breakdown["prediction_head"] = {
            "params": head_params,
            "percent": round(100 * head_params / total_params, 1),
        }

    # IC head (for joint models)
    if hasattr(model, "ic_head"):
        ic_head_params = count_params(model.ic_head)
        breakdown["ic_head"] = {
            "params": ic_head_params,
            "percent": round(100 * ic_head_params / total_params, 1),
        }

    # CE head (for joint models)
    if hasattr(model, "ce_head"):
        ce_head_params = count_params(model.ce_head)
        breakdown["ce_head"] = {
            "params": ce_head_params,
            "percent": round(100 * ce_head_params / total_params, 1),
        }

    # Add global pooling if it has learnable parameters
    if hasattr(model, "global_pooling"):
        pooling_params = count_params(model.global_pooling)
        if pooling_params > 0:
            breakdown["global_pooling"] = {
                "params": pooling_params,
                "percent": round(100 * pooling_params / total_params, 1),
            }

    return breakdown


def measure_flops(model: nn.Module, sample_input: tuple) -> Optional[int]:
    """
    Measure FLOPs (floating point operations) for a single forward pass.

    Args:
        model: PyTorch model
        sample_input: Tuple of input tensors (same format as model forward pass)

    Returns:
        Total FLOPs, or None if fvcore is not available
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        # Move model and inputs to CPU for FLOPs counting (more stable)
        model_cpu = model.cpu()
        sample_input_cpu = tuple(
            x.cpu() if isinstance(x, torch.Tensor) else x for x in sample_input
        )

        flops = FlopCountAnalysis(model_cpu, sample_input_cpu)
        total_flops = flops.total()

        # Move model back to original device
        if torch.cuda.is_available():
            model.cuda()

        return int(total_flops)
    except ImportError:
        print("Warning: fvcore not installed. FLOPs measurement skipped.")
        print("Install with: pip install fvcore")
        return None


def measure_memory(reset: bool = True) -> Dict[str, float]:
    """
    Measure peak GPU memory usage.

    Args:
        reset: If True, reset peak memory statistics after measurement

    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"peak_memory_gb": 0.0, "allocated_gb": 0.0}

    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    current_memory = torch.cuda.memory_allocated() / 1024**3

    if reset:
        torch.cuda.reset_peak_memory_stats()

    return {
        "peak_memory_gb": round(peak_memory, 3),
        "allocated_gb": round(current_memory, 3),
    }


def save_runtime_stats(
    stats: Dict[str, Any], output_path: Path, filename: str = "runtime_stats.json"
):
    """
    Save runtime statistics to JSON file.

    Args:
        stats: Dictionary of runtime statistics
        output_path: Directory to save the file
        filename: Name of the output file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nRuntime statistics saved to: {filepath}")


def format_runtime_summary(stats: Dict[str, Any]) -> str:
    """
    Format runtime statistics as a readable string.

    Args:
        stats: Dictionary of runtime statistics

    Returns:
        Formatted string summary
    """
    lines = ["\n" + "=" * 60]
    lines.append("RUNTIME STATISTICS SUMMARY")
    lines.append("=" * 60)

    if "model" in stats:
        lines.append("\nModel Complexity:")
        lines.append(
            f"  Parameters (total):     {stats['model'].get('parameters', 0):,}"
        )
        lines.append(
            f"  Parameters (trainable): {stats['model'].get('trainable_parameters', 0):,}"
        )
        if stats["model"].get("flops") is not None:
            flops_m = stats["model"]["flops"] / 1e6
            flops_g = stats["model"]["flops"] / 1e9
            lines.append(f"  FLOPs per forward pass: {flops_m:.2f}M ({flops_g:.3f}G)")

    if "training" in stats:
        lines.append("\nTraining Performance:")
        lines.append(
            f"  Time per epoch:   {stats['training'].get('time_per_epoch_sec', 0):.2f} sec"
        )
        lines.append(
            f"  Total time:       {stats['training'].get('total_time_min', 0):.2f} min"
        )
        lines.append(
            f"  Peak GPU memory:  {stats['training'].get('peak_memory_gb', 0):.2f} GB"
        )

    if "inference" in stats:
        lines.append("\nInference Performance:")
        lines.append(
            f"  Throughput (batch=1024): {stats['inference'].get('throughput_pred_per_sec', 0):.1f} pred/sec"
        )
        lines.append(
            f"  Latency (single pred):   {stats['inference'].get('latency_ms', 0):.2f} ms"
        )
        lines.append(
            f"  Peak GPU memory:         {stats['inference'].get('peak_memory_gb', 0):.2f} GB"
        )

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def print_runtime_summary(stats: Dict[str, Any]):
    """Print formatted runtime statistics."""
    print(format_runtime_summary(stats))
