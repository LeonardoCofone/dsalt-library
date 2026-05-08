import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def resolve_device(device: str = "cpu", num_gpus: int = 1) -> tuple[torch.device, list[int]]:
    """
    Resolve the primary device and list of GPU IDs based on the requested device and number of GPUs.

    Args:
        device (str): The device type, either "cpu" or "cuda".
        num_gpus (int): The number of GPUs to use.

    Returns:
        tuple: (primary_device, gpu_ids) where gpu_ids is an empty list for CPU or a list of GPU indices.
    """
    if device == "cpu":
        return torch.device("cpu"), []

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU.")
        return torch.device("cpu"), []

    available = torch.cuda.device_count()
    n = min(num_gpus, available)

    if n <= 0:
        return torch.device("cpu"), []

    gpu_ids = list(range(n))
    primary = torch.device(f"cuda:{gpu_ids[0]}")  # Primary device is the first GPU
    return primary, gpu_ids


def print_gpu_info() -> str:
    """
    Return formatted information about available GPUs as a string.

    Returns:
        str: Formatted GPU information.
    """
    if not torch.cuda.is_available():
        return "CUDA not available."
    
    lines = [f"\n{'='*60}"]
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"GPU {i}: {p.name}  {p.total_memory/1e9:.1f}GB")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)