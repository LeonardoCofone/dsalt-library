import logging
import sys
import time
from pathlib import Path


class _ThroughputFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "it_s"):
            record.it_s = 0.0
        if not hasattr(record, "tok_s"):
            record.tok_s = 0
        if not hasattr(record, "mem_gb"):
            record.mem_gb = 0.0
        return True


_FMT_CONSOLE = (
    "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
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
    sh.addFilter(_ThroughputFilter())
    logger.addHandler(sh)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_fmt = logging.Formatter(fmt=_FMT_FILE, datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
        fh.setFormatter(file_fmt)
        fh.addFilter(_ThroughputFilter())
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class StepTimer:
    def __init__(self, window: int = 50):
        self._window = window
        self._times: list[float] = []
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        if len(self._times) > self._window:
            self._times.pop(0)
        self._t0 = None
        return elapsed

    @property
    def avg_it_s(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    def reset(self) -> None:
        self._times.clear()
        self._t0 = None