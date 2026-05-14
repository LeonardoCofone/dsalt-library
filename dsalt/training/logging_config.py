import logging
import math
import sys
import time
from pathlib import Path

import torch

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"

BG_GREEN = "\033[48;5;22m"


def _bar(value: float, max_val: float, width: int = 8, color: str = GREEN) -> str:
    filled = int(round(value / max(max_val, 1e-9) * width))
    filled = max(0, min(filled, width))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def _mem_color(gb: float) -> str:
    if gb > 12: return RED
    if gb > 8:  return YELLOW
    return GREEN


def _loss_color(v: float) -> str:
    if v > 10: return RED
    if v > 5:  return YELLOW
    return GREEN


def _fmt_ppl(loss: float) -> str:
    try:
        ppl = math.exp(min(loss, 20.0))
        c = RED if ppl > 1000 else (YELLOW if ppl > 100 else GREEN)
        return f"{c}{ppl:>12.4f}{RESET}"
    except Exception:
        return f"{DIM}{'N/A':>12}{RESET}"


def _fmt_float(s: str, width: int = 10, color: str = WHITE) -> str:
    try:
        return f"{color}{float(s):>{width}.4f}{RESET}"
    except Exception:
        return f"{DIM}{'nan':>{width}}{RESET}"


def _parse_step_record(msg: str) -> dict:
    parts = {}
    for chunk in msg.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, _, v = chunk.partition("=")
        else:
            space = chunk.find(" ")
            if space == -1:
                continue
            k = chunk[:space]
            v = chunk[space:]
        parts[k.strip()] = v.strip()
    return parts


SEP  = f"{DIM}{'─' * 72}{RESET}"
SEP2 = f"{DIM}{'━' * 72}{RESET}"


class _StepFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        if "training start" in msg:
            return self._banner(msg)
        if "val_ppl" in msg or "val ppl" in msg.lower():
            return self._val(msg)
        if "checkpoint" in msg or "done |" in msg:
            return f"  {MAGENTA}✓ {msg}{RESET}"

        return self._step(record, msg)

    def _banner(self, msg: str) -> str:
        lines = [
            f"\n{BOLD}{CYAN}{SEP2[len(DIM):-len(RESET)]}{RESET}",
            f"{BOLD}{CYAN}  ██████╗  ███████╗  █████╗  ██╗   ████████╗{RESET}",
            f"{BOLD}{CYAN}  ██╔══██╗ ██╔════╝ ██╔══██╗ ██║   ╚══██╔══╝{RESET}",
            f"{BOLD}{CYAN}  ██║  ██║ ███████╗ ███████║ ██║      ██║   {RESET}",
            f"{BOLD}{CYAN}  ██║  ██║ ╚════██║ ██╔══██║ ██║      ██║   {RESET}",
            f"{BOLD}{CYAN}  ██████╔╝ ███████║ ██║  ██║ ███████╗ ██║   {RESET}",
            f"{BOLD}{CYAN}  ╚═════╝  ╚══════╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝   {RESET}",
            f"{BOLD}{CYAN}{'━' * 72}{RESET}",
        ]
        for part in msg.split("|"):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition("=")
            lines.append(f"  {YELLOW}{k.strip():>22}{RESET} {DIM}={RESET} {WHITE}{v.strip()}{RESET}")
        lines.append(f"{BOLD}{CYAN}{'━' * 72}{RESET}\n")
        return "\n".join(lines)

    def _val(self, msg: str) -> str:
        is_best = "← best" in msg
        c = GREEN if is_best else YELLOW
        badge = f"  {BOLD}{BG_GREEN} ★ NEW BEST {RESET}" if is_best else ""
        return (
            f"\n{c}{'─' * 60}{RESET}\n"
            f"  {BOLD}{c}VAL{RESET}  {WHITE}{msg}{RESET}{badge}\n"
            f"{c}{'─' * 60}{RESET}\n"
        )

    def _step(self, record: logging.LogRecord, msg: str) -> str:
        parts   = _parse_step_record(msg)
        ts      = f"{DIM}{self.formatTime(record, self.datefmt)}{RESET}"
        it_s    = getattr(record, "it_s",   0.0)
        mem_gb  = getattr(record, "mem_gb", 0.0)
        mem_c   = _mem_color(mem_gb)
        mem_bar = _bar(mem_gb, 16.0, width=6, color=mem_c)

        step  = parts.get("step",  "?")
        lr    = parts.get("lr",    "?")
        noise = parts.get("noise", "?")

        try:
            loss_v = float(parts.get("loss", "nan"))
            lc     = _loss_color(loss_v)
            loss_s = f"{lc}{loss_v:>10.4f}{RESET}"
            ppl_s  = _fmt_ppl(loss_v)
        except Exception:
            loss_s = f"{DIM}{'nan':>10}{RESET}"
            ppl_s  = f"{DIM}{'nan':>12}{RESET}"

        sigma2 = _fmt_float(parts.get("σ²",      parts.get("sigma2", "nan")), width=10, color=YELLOW)
        rank   = _fmt_float(parts.get("rank",     "nan"),                      width=8,  color=CYAN)
        res    = _fmt_float(parts.get("res",      "nan"),                      width=10, color=WHITE)
        H      = _fmt_float(parts.get("H",        "nan"),                      width=8,  color=GREEN)
        sink   = _fmt_float(parts.get("sink",     "nan"),                      width=8,  color=YELLOW)
        hstd   = _fmt_float(parts.get("head_std", parts.get("hstd", "nan")),   width=8,  color=MAGENTA)
        noise_s = _fmt_float(noise,                                             width=8,  color=RED)

        speed_s = (
            f"{CYAN}{it_s:5.2f} it/s{RESET}"
            if it_s > 0 else f"{DIM} ?.?? it/s{RESET}"
        )
        mem_s = (
            f"{mem_c}{mem_gb:4.1f} GB{RESET}"
            if mem_gb > 0 else f"{DIM} ?.? GB{RESET}"
        )

        bar = "─" * 72

        return (
            f"\n{DIM}{bar}{RESET}\n"
            f"  {ts}   {BOLD}step {CYAN}{step:>5}{RESET}   "
            f"lr {MAGENTA}{lr}{RESET}   "
            f"{speed_s}   {mem_s} {mem_bar}\n"
            f"{DIM}{bar}{RESET}\n"
            f"  {'loss':>8}  {loss_s}    {'ppl':>8}  {ppl_s}\n"
            f"  {'σ²':>8}  {sigma2}    {'rank':>8}  {rank}\n"
            f"  {'residual':>8}  {res}    {'entropy H':>8}  {H}\n"
            f"  {'noise':>8}  {noise_s}    {'sink':>8}  {sink}    {'head_std':>8}  {hstd}\n"
            f"{DIM}{bar}{RESET}"
        )


class _DSALTFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ["it_s", "tok_s", "mem_gb", "rank_eff"]:
            if not hasattr(record, field):
                setattr(record, field, 0.0)
        return True


def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_StepFormatter(datefmt="%H:%M:%S"))
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
        if self._t0 is None:
            return {}
        if self._device and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        if len(self._times) > self._window:
            self._times.pop(0)

        avg_time = sum(self._times) / len(self._times)
        return {
            "it_s":      1.0 / avg_time if avg_time > 0 else 0.0,
            "tok_s":     total_tokens / elapsed if elapsed > 0 else 0.0,
            "step_time": elapsed,
        }