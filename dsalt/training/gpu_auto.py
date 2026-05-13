import os
import torch
import torch.distributed as dist


def get_device(rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def get_gpu_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    dev = device or torch.device("cuda")
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
        "utilization_pct": round(100.0 * allocated / max(total, 1e-6), 1),
    }


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(10000 + (os.getpid() % 50000)))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    return rank == 0


def barrier(rank: int, world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()