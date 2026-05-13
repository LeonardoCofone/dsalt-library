import os
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_gpus() -> int:
    return torch.cuda.device_count()


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


def init_accelerator(
    mixed_precision: str = "bf16",
    grad_accum: int = 1,
    log_dir: str = "./logs",
    seed: int = 42,
) -> Accelerator:
    set_seed(seed)

    project_cfg = ProjectConfiguration(project_dir=log_dir, logging_dir=log_dir)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=grad_accum,
        device_placement=True,
        split_batches=False,
        step_scheduler_with_optimizer=False,
        project_config=project_cfg,
        dynamo_backend="no",
    )

    if accelerator.is_main_process:
        n = count_gpus()
        print(
            f"[accelerate] processes={accelerator.num_processes} | "
            f"gpus_visible={n} | "
            f"mixed_precision={mixed_precision} | "
            f"grad_accum={grad_accum}"
        )

    return accelerator


def prepare_model_training(accelerator, model, optimizer, train_loader, val_loader, scheduler):
    return accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)