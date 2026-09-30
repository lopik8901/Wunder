"""Keep MIOpen's writable databases inside the competition workspace."""
from __future__ import annotations

import os
from pathlib import Path


def configure_miopen_cache() -> None:
    base = Path(__file__).resolve().parent / "environment"
    for name, child in (("MIOPEN_USER_DB_PATH", "miopen_db"),
                        ("MIOPEN_CUSTOM_CACHE_DIR", "miopen_cache")):
        path = base / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
