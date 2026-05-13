import torch
import torch.nn as nn
from accelerate import Accelerator


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_gpus():
    return torch.cuda.device_count()


def init_accelerator(
    mixed_precision: str = "fp16",
    grad_accum: int = 1,
):
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=grad_accum,
        device_placement=True,
        split_batches=True,
        step_scheduler_with_optimizer=False,
    )
    return accelerator


def prepare_model_training(accelerator, model, optimizer, train_loader, val_loader):
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    return model, optimizer, train_loader, val_loader