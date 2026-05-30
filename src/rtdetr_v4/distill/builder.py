"""Builders for distillation-only components."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from torch import nn

from .teachers import DINOv2TeacherModel, DINOv3TeacherModel
from ..config import load_config, merge_dict


TEACHER_CLASSES = {
    "DINOv2TeacherModel": DINOv2TeacherModel,
    "DINOv3TeacherModel": DINOv3TeacherModel,
}


def _prepare_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    working_cfg = merge_dict({}, config, inplace=True)
    if overrides:
        merge_dict(working_cfg, overrides, inplace=True)
    return working_cfg


def _resolve_teacher_entry(
    config: dict[str, Any],
    teacher_entry: str | dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if teacher_entry is None:
        raise ValueError("Teacher config must define the top-level key 'teacher_model'.")

    if isinstance(teacher_entry, str):
        teacher_name = teacher_entry
        overrides: dict[str, Any] = {}
    elif isinstance(teacher_entry, dict):
        if "type" not in teacher_entry:
            raise ValueError("Teacher config must be either a string name or a dict with 'type'.")
        teacher_name = str(teacher_entry["type"])
        overrides = {key: value for key, value in teacher_entry.items() if key != "type"}
    else:
        raise TypeError(f"Unsupported teacher config entry: {type(teacher_entry).__name__}")

    base_cfg = {}
    if teacher_name in config:
        teacher_cfg = config[teacher_name]
        if teacher_cfg is None:
            teacher_cfg = {}
        if not isinstance(teacher_cfg, dict):
            raise TypeError(f"Expected teacher section '{teacher_name}' to be a dict, got: {type(teacher_cfg).__name__}")
        base_cfg = merge_dict({}, teacher_cfg, inplace=True)
    if overrides:
        merge_dict(base_cfg, overrides, inplace=True)
    return teacher_name, base_cfg


def _instantiate_teacher(teacher_name: str, teacher_cfg: dict[str, Any]) -> nn.Module:
    teacher_cls = TEACHER_CLASSES.get(teacher_name)
    if teacher_cls is None:
        supported = ", ".join(sorted(TEACHER_CLASSES))
        raise ValueError(f"Unsupported teacher type: {teacher_name}. Supported: {supported}")

    constructor = inspect.signature(teacher_cls.__init__)
    accepted_names = {
        name
        for name, parameter in constructor.parameters.items()
        if name != "self" and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    kwargs = {key: value for key, value in teacher_cfg.items() if key in accepted_names}
    return teacher_cls(**kwargs)


def build_teacher_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Build the teacher network defined by one config dictionary."""
    working_cfg = _prepare_config(config, overrides=overrides)
    teacher_name, teacher_cfg = _resolve_teacher_entry(working_cfg, working_cfg.get("teacher_model"))
    return _instantiate_teacher(teacher_name, teacher_cfg)


def build_teacher_from_yaml(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Load one YAML file and build its teacher network."""
    config = load_config(str(config_path))
    return build_teacher_from_config(config, overrides=overrides)


__all__ = ["build_teacher_from_config", "build_teacher_from_yaml"]
