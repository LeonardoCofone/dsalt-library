import logging
import sys
import time
import torch
from pathlib import Path

BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            'INFO': GREEN,
            'WARNING': YELLOW,
            'ERROR': RED,
            'CRITICAL': RED + BOLD,
            'DEBUG': BLUE
        }.get(record.levelname, RESET)

        stats = ""
        if hasattr(record, "it_s") and record.it_s > 0:
            stats += f"{BLUE}{record.it_s:5.2f} it/s{RESET} | "
        if hasattr(record, "mem_gb") and record.mem_gb > 0:
            color_mem = RED if record.mem_gb > 20 else GREEN
            stats += f"{color_mem}{record.mem_gb:4.1f} GB{RESET} | "
        if hasattr(record, "rank_eff") and record.rank_eff > 0:
            stats += f"{YELLOW}Rank: {record.rank_eff:.2f}{RESET} | "

        record.levelname = f"{level_color}{record.levelname:.1s}{RESET}"
        time_str = f"{BOLD}{self.formatTime(record, self.datefmt)}{RESET}"
        
        return f"{time_str} | {record.levelname} | {stats}{record.getMessage()}"

class _DSALTFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ["it_s", "tok_s", "mem_gb", "rank_eff"]:
            if not hasattr(record, field):
                setattr(record, field, 0.0)
        return True

def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    logger.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
    sh.addFilter(_DSALTFilter())
    logger.addHandler(sh)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
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
        if self._t0 is None: return {}
        if self._device and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        if len(self._times) > self._window: self._times.pop(0)
        
        avg_time = sum(self._times) / len(self._times)
        return {
            "it_s": 1.0 / avg_time if avg_time > 0 else 0,
            "tok_s": (total_tokens / elapsed) if elapsed > 0 else 0,
            "step_time": elapsed
        }