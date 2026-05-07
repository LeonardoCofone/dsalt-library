import os
import sys
import subprocess
from typing import Tuple
import torch
import torch.nn as nn
import torch.distributed as dist


class GPUAutoConfig:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cuda_available = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.cuda_available else 0
        self.is_distributed = (
            dist.is_available()
            and dist.is_initialized()
        )
        self.rank = dist.get_rank() if self.is_distributed else int(os.environ.get("RANK", 0))
        self.world_size = dist.get_world_size() if self.is_distributed else int(os.environ.get("WORLD_SIZE", 1))

        if self.verbose and self.rank == 0:
            self._print_info()

    def _print_info(self):
        print(f"\n{'='*60}")
        print(f"GPU Auto-Config Report")
        print(f"{'='*60}")
        print(f"CUDA Available:      {self.cuda_available}")
        print(f"Number of GPUs:      {self.num_gpus}")
        print(f"Is Distributed:      {self.is_distributed}")
        if self.num_gpus > 0:
            for i in range(min(self.num_gpus, 8)):
                try:
                    prop = torch.cuda.get_device_properties(i)
                    print(f"  GPU{i}: {prop.name} ({prop.total_memory / 1e9:.1f}GB)")
                except Exception:
                    pass
        if self.is_distributed:
            print(f"Rank:                {self.rank}/{self.world_size}")
        print(f"{'='*60}\n")

    @property
    def device(self) -> torch.device:
        if self.cuda_available:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def in_torchrun(self) -> bool:
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ

    def recommended_strategy(self) -> str:
        if not self.cuda_available:
            return "cpu"
        if self.in_torchrun and self.world_size > 1:
            return "distributed_ddp"
        if self.num_gpus == 1:
            return "single_gpu"
        return "cpu"


def auto_detect_gpus_simple() -> Tuple[int, torch.device]:
    if not torch.cuda.is_available():
        return 0, torch.device("cpu")
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if num_gpus > 0 else "cpu")
    return num_gpus, device


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    print(f"\n{'='*70}")
    print(f"GPU Information")
    print(f"{'='*70}")
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name}")
        print(f"  Memory: {prop.total_memory / 1e9:.2f} GB")
        print(f"  Multiprocessors: {prop.multi_processor_count}")
    print(f"{'='*70}\n")