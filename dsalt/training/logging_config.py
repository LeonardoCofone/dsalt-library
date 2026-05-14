import logging
import sys
import time
import torch
from pathlib import Path

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLACK   = "\033[30m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"

BG_DARK  = "\033[48;5;235m"
BG_BLUE  = "\033[48;5;17m"
BG_GREEN = "\033[48;5;22m"
BG_RED   = "\033[48;5;52m"

def _bar(value: float, max_val: float, width: int = 8, color: str = GREEN) -> str:
    filled = int(round(value / max(max_val, 1e-9) * width))
    filled = max(0, min(filled, width))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"

def _mem_color(gb: float) -> str:
    if gb > 12:  return RED
    if gb > 8:   return YELLOW
    return GREEN

def _loss_color(loss: float) -> str:
    if loss > 10: return RED
    if loss > 5:  return YELLOW
    return GREEN

def _ppl_str(loss: float) -> str:
    import math
    try:
        ppl = math.exp(min(loss, 20.0))
        color = RED if ppl > 1000 else (YELLOW if ppl > 100 else GREEN)
        return f"{color}{ppl:>10.2f}{RESET}"
    except Exception:
        return f"{DIM}{'N/A':>10}{RESET}"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()

        if "training start" in msg:
            return self._format_banner(msg)

        if "val_ppl" in msg or "val ppl" in msg.lower():
            return self._format_val(msg)

        if "checkpoint" in msg or "done |" in msg:
            return self._format_checkpoint(msg)

        it_s   = getattr(record, "it_s",   0.0)
        mem_gb = getattr(record, "mem_gb", 0.0)

        time_str = f"{DIM}{self.formatTime(record, self.datefmt)}{RESET}"
        level_str = f"{GREEN}I{RESET}"

        speed_str = (
            f"{CYAN}{it_s:5.2f}{RESET} {DIM}it/s{RESET}"
            if it_s > 0 else f"{DIM} ?.?? it/s{RESET}"
        )

        mem_c = _mem_color(mem_gb)
        mem_str = (
            f"{mem_c}{mem_gb:4.1f}{RESET} {DIM}GB{RESET}"
            if mem_gb > 0 else f"{DIM} ?.? GB{RESET}"
        )

        return f"{time_str} {DIM}│{RESET} {level_str} {DIM}│{RESET} {speed_str} {DIM}│{RESET} {mem_str} {DIM}│{RESET} {msg}"

    def _format_banner(self, msg: str) -> str:
        lines = []
        lines.append(f"\n{BOLD}{CYAN}{'━' * 72}{RESET}")
        lines.append(f"{BOLD}{CYAN}  ██████╗ ███████╗ █████╗ ██╗  ████████╗{RESET}")
        lines.append(f"{BOLD}{CYAN}  ██╔══██╗██╔════╝██╔══██╗██║  ╚══██╔══╝{RESET}")
        lines.append(f"{BOLD}{CYAN}  ██║  ██║███████╗███████║██║     ██║   {RESET}")
        lines.append(f"{BOLD}{CYAN}  ██║  ██║╚════██║██╔══██║██║     ██║   {RESET}")
        lines.append(f"{BOLD}{CYAN}  ██████╔╝███████║██║  ██║███████╗██║   {RESET}")
        lines.append(f"{BOLD}{CYAN}  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝   {RESET}")
        lines.append(f"{BOLD}{CYAN}{'━' * 72}{RESET}")
        for part in msg.split(" | "):
            key, _, val = part.partition("=")
            lines.append(f"  {YELLOW}{key.strip():>22}{RESET} {DIM}={RESET} {WHITE}{val.strip()}{RESET}")
        lines.append(f"{BOLD}{CYAN}{'━' * 72}{RESET}\n")
        return "\n".join(lines)

    def _format_val(self, msg: str) -> str:
        is_best = "← best" in msg
        color   = GREEN if is_best else YELLOW
        badge   = f" {BOLD}{BG_GREEN} ★ NEW BEST {RESET}" if is_best else ""
        return f"\n{color}{'─' * 60}{RESET}\n  {BOLD}{color}VAL{RESET}  {WHITE}{msg}{RESET}{badge}\n{color}{'─' * 60}{RESET}\n"

    def _format_checkpoint(self, msg: str) -> str:
        return f"  {MAGENTA} {msg}{RESET}"


class _StepFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()

        if any(k in msg for k in ["training start", "val_ppl", "checkpoint", "done |"]):
            return ColoredFormatter(datefmt="%H:%M:%S").format(record)

        parts = {}
        for chunk in msg.split(" | "):
            if " " in chunk.strip():
                first_space = chunk.strip().index(" ")
                k = chunk.strip()[:first_space]
                v = chunk.strip()[first_space+1:]
                parts[k] = v.strip()

        try:
            step     = parts.get("step", "?").strip()
            loss_raw = float(parts.get("loss", "nan"))
            ppl_raw  = float(parts.get("ppl",  "nan"))
            lr_raw   = parts.get("lr", "?")
            sigma2   = parts.get("σ₂", "?")
            rank     = parts.get("rank", "?")
            res      = parts.get("res", "?")
            H        = parts.get("H", "?")
            noise    = parts.get("noise", "?")
            sink     = parts.get("sink", "?")
            hstd     = parts.get("head_std", "?")

            lc = _loss_color(loss_raw)
            it_s   = getattr(record, "it_s",   0.0)
            mem_gb = getattr(record, "mem_gb", 0.0)
            mem_c  = _mem_color(mem_gb)
            mem_bar = _bar(mem_gb, 16.0, width=6, color=mem_c)

            time_str = f"{DIM}{self.formatTime(record, self.datefmt)}{RESET}"

            line1 = (
                f"{time_str} {DIM}│{RESET} "
                f"{BOLD}step {CYAN}{step:>4}{RESET}  "
                f"loss {lc}{loss_raw:7.4f}{RESET}  "
                f"ppl {_ppl_str(loss_raw)}  "
                f"lr {MAGENTA}{lr_raw}{RESET}"
            )
            line2 = (
                f"{'':13}{DIM}│{RESET} "
                f"σ² {YELLOW}{sigma2:>10}{RESET}  "
                f"rank {CYAN}{rank:>7}{RESET}  "
                f"res {WHITE}{res:>8}{RESET}  "
                f"H {GREEN}{H:>8}{RESET}"
            )
            line3 = (
                f"{'':13}{DIM}│{RESET} "
                f"noise {RED}{noise:>8}{RESET}  "
                f"sink {YELLOW}{sink:>8}{RESET}  "
                f"hstd {MAGENTA}{hstd:>8}{RESET}  "
                f"{CYAN}{it_s:5.2f} it/s{RESET}  "
                f"{mem_c}{mem_gb:4.1f}GB{RESET} {mem_bar}"
            )
            return f"{line1}\n{line2}\n{line3}"
        except Exception:
            return msg


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
            "tok_s":     (total_tokens / elapsed) if elapsed > 0 else 0.0,
            "step_time": elapsed,
        }