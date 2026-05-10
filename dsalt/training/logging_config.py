import logging
import sys


_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_CYAN           = "\033[36m"
_BRIGHT_GREEN   = "\033[92m"
_BRIGHT_YELLOW  = "\033[93m"
_BRIGHT_RED     = "\033[91m"
_BRIGHT_CYAN    = "\033[96m"

_LEVEL_COLORS = {
    logging.DEBUG:    _DIM + _CYAN,
    logging.INFO:     _BRIGHT_GREEN,
    logging.WARNING:  _BRIGHT_YELLOW,
    logging.ERROR:    _BRIGHT_RED,
    logging.CRITICAL: _BOLD + _BRIGHT_RED,
}

_LEVEL_LABELS = {
    logging.DEBUG:    "DBG",
    logging.INFO:     "INF",
    logging.WARNING:  "WRN",
    logging.ERROR:    "ERR",
    logging.CRITICAL: "CRT",
}


class _ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        time_str  = self.formatTime(record, datefmt="%H:%M:%S")
        level_str = _LEVEL_LABELS.get(record.levelno, record.levelname)
        msg       = record.getMessage()
        name      = record.name.replace("dsalt.", "").replace("dsalt", "root")

        if self.use_color:
            lvl_color = _LEVEL_COLORS.get(record.levelno, _RESET)
            return (
                f"{_DIM}{time_str}{_RESET} "
                f"{lvl_color}{_BOLD}[{level_str}]{_RESET} "
                f"{_BRIGHT_CYAN}{name}{_RESET} "
                f"{msg}"
            )
        else:
            return f"{time_str} [{level_str}] {name} {msg}"


def _supports_color(stream) -> bool:
    try:
        return hasattr(stream, "isatty") and stream.isatty()
    except Exception:
        return False


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    logger = logging.getLogger("dsalt")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    use_color = _supports_color(sys.stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(use_color=use_color))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(_ColorFormatter(use_color=False))
        logger.addHandler(file_handler)