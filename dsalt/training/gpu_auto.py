import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: nn.Module, rank: int) -> DDP:
    return DDP(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=False)


def get_device(device: str, num_gpus: int) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if num_gpus == 1:
        return torch.device("cuda:0")
    return torch.device("cuda")


def model_to_device(model: nn.Module, device: torch.device, num_gpus: int) -> nn.Module:
    model = model.to(device)
    if device.type == "cuda" and num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    return model


def count_gpus() -> int:
    return torch.cuda.device_count()