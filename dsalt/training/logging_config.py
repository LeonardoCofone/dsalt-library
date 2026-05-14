import logging
import sys
import time
import torch
from pathlib import Path


class _DSALTFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ["it_s", "tok_s", "mem_gb", "rank_eff"]:
            if not hasattr(record, field):
                setattr(record, field, 0.0)
        return True


_FMT_CONSOLE = (
    "%(asctime)s | %(levelname).1s | %(it_s)5.2f it/s | %(mem_gb)4.1f GB | %(message)s"
)
_FMT_FILE = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str,
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    console_fmt = logging.Formatter(fmt=_FMT_CONSOLE, datefmt=_DATEFMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(console_fmt)
    sh.addFilter(_DSALTFilter())
    logger.addHandler(sh)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_fmt = logging.Formatter(fmt=_FMT_FILE, datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
        fh.setFormatter(file_fmt)
        fh.addFilter(_DSALTFilter())
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class StepTimer:
    def __init__(self, window: int = 50, device: torch.device = None):
        self._window = window
        self._times: list[float] = []
        self._t0: float | None = None
        self._device = device

    def start(self) -> None:
        if self._device and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._t0 = time.perf_counter()

    def stop(self, total_tokens: int = 0) -> dict:
        if self._t0 is None:
            return {}
        
        if self._device and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        
        if len(self._times) > self._window:
            self._times.pop(0)
        
        avg_time = sum(self._times) / len(self._times)
        it_s = 1.0 / avg_time if avg_time > 0 else 0
        
        tok_s = (total_tokens / elapsed) if elapsed > 0 else 0
        
        self._t0 = None
        return {"it_s": it_s, "tok_s": tok_s, "step_time": elapsed}