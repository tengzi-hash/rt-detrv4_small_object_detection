"""Lightweight YAML loading helpers for explicit project builders."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


INCLUDE_KEY = "__include__"
REPLACE_KEY = "__replace__"


def merge_dict(dct: dict[str, Any], another_dct: dict[str, Any], inplace: bool = True) -> dict[str, Any]:
    """Merge ``another_dct`` into ``dct`` recursively."""

    def _merge(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        for key, value in incoming.items():
            if isinstance(value, dict):
                replace_value = bool(value.get(REPLACE_KEY))
                normalized_value = {
                    nested_key: nested_value
                    for nested_key, nested_value in value.items()
                    if nested_key != REPLACE_KEY
                }
                if replace_value or key not in current or not isinstance(current[key], dict):
                    current[key] = copy.deepcopy(normalized_value)
                else:
                    _merge(current[key], normalized_value)
            else:
                current[key] = value
        return current

    target = dct if inplace else copy.deepcopy(dct)
    return _merge(target, another_dct)


def _load_config_recursive(file_path: Path, merged_cfg: dict[str, Any]) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file_handle:
        file_cfg = yaml.safe_load(file_handle) or {}

    include_entries = file_cfg.pop(INCLUDE_KEY, []) or []
    for include_entry in include_entries:
        include_path = Path(include_entry).expanduser()
        if not include_path.is_absolute():
            include_path = file_path.parent / include_path
        _load_config_recursive(include_path.resolve(), merged_cfg)

    merge_dict(merged_cfg, file_cfg, inplace=True)
    return merged_cfg


def load_config(file_path: str | Path) -> dict[str, Any]:
    """Load one YAML config file, resolving recursive includes."""
    resolved_path = Path(file_path).expanduser().resolve()
    if resolved_path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Only YAML configs are supported, got: {resolved_path}")
    return _load_config_recursive(resolved_path, {})


__all__ = ["INCLUDE_KEY", "REPLACE_KEY", "load_config", "merge_dict"]
