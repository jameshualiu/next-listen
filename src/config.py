"""Shared paths, env loading, and small helpers used across src/*.py.

Every script in this package is run directly (`python src/whatever.py`),
so cross-module imports here use bare module names (e.g. `import config`)
rather than package-relative imports -- see CLAUDE.md's operational
commands, which all invoke scripts this way.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Raw Spotify API response cache (CLAUDE.md's caching rule).
SPOTIFY_RAW_DIR = DATA_RAW / "spotify"
# Manually-downloaded Million Playlist Dataset slices (not fetched by code).
MPD_DIR = DATA_RAW / "mpd"

# Read-only scopes -- this project never writes to a user's Spotify account.
SPOTIFY_SCOPES = "user-top-read user-library-read user-read-recently-played"

RANDOM_SEED = 42
DEFAULT_WEIGHT_EMBEDDING = 0.6


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Copy .env.example to .env and fill it in."
        )
    return value
