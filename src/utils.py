from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


DEFAULT_RANDOM_STATE: int = 42


def get_project_root() -> Path:
    """Return the project root directory (the directory containing this file's parent)."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for scripts/notebooks."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
