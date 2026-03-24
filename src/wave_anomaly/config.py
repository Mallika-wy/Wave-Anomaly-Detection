from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return (base / path).resolve()
