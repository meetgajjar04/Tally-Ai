"""Central logging setup."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from .config import LOG_DIR

LOG_FILE = LOG_DIR / "tally_ai_desk.log"

_logger = logging.getLogger("tally_ai_desk")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

if not _logger.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    _logger.addHandler(_fh)

    _sh = logging.StreamHandler()
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    _logger.addHandler(_sh)


def get_logger(name: str = "") -> logging.Logger:
    return _logger.getChild(name) if name else _logger
