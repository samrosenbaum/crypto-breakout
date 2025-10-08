"""Utilities for loading and working with strategy configuration files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Load YAML based configuration profiles.

    Parameters
    ----------
    path:
        Path to a YAML configuration file. The loader caches parsed
        configurations to avoid re-reading the same file multiple times.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")

    @lru_cache(maxsize=8)
    def load(self) -> Dict[str, Any]:
        """Parse the YAML configuration file and return a dictionary."""

        with self.path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(
                "Configuration root must be a mapping (dict); "
                f"received type {type(data)!r}"
            )

        return data
