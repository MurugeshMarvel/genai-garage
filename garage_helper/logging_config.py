import logging
import sys

LOGGER_NAME = "garage_helper"

_FMT = logging.Formatter(
    "[%(asctime)s] %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)


class _FlushHandler(logging.StreamHandler):
    """Flush after every record so Jupyter cell output stays in order."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def get_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if verbose else logging.WARNING

    if not logger.handlers:
        handler = _FlushHandler(sys.stdout)
        handler.setFormatter(_FMT)
        logger.addHandler(handler)
        logger.propagate = False  # don't bubble up to the root logger

    # Always sync level so calling get_logger(verbose=True) in a later cell works
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)

    return logger


def set_verbose(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
