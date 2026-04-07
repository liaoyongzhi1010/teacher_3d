from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dictionary with attribute-style access for nested config values."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(data)
