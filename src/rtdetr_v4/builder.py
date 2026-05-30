"""Top-level builders for the migrated RT-DETR v4 project.

The project exposes project-local RTv4 builders under ``rtdetr_v4.models``.
This module provides the stable entrypoints the rest of the project should call:

- ``build_model(...)``
- ``build_postprocessor(...)``
- ``build_criterion(...)``
- ``build_teacher(...)``
"""

from __future__ import annotations

from torch import nn

from .distill import build_teacher_from_yaml
from .models import (
    build_criterion_from_yaml,
    build_model_from_yaml,
    build_postprocessor_from_yaml,
)


SUPPORTED_MODEL_TYPES = {"rtdetr_v4"}
SUPPORTED_CRITERION_TYPES = {"rtdetr_v4"}
SUPPORTED_POSTPROCESSOR_TYPES = {"rtdetr_v4"}
SUPPORTED_TEACHER_TYPES = {"rtdetr_v4_teacher"}


def _resolve_model_config_request(component_config: dict) -> tuple[str, dict]:
    """Extract the project-local YAML path plus overrides from one component config."""
    config_path = component_config.get("config_path")
    if not config_path:
        raise ValueError(
            "Model component config must define 'config_path', for example "
            "'configs/project.yml'."
        )

    overrides = component_config.get("config_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError(f"'config_overrides' must be a dict, got: {type(overrides).__name__}")
    return str(config_path), dict(overrides)


def build_model(model_config: dict) -> nn.Module:
    """Build the configured model.

    Supported model type right now:

    - ``rtdetr_v4``: build the project-local RTv4 model stack
    """

    model_type = model_config.get("type", "rtdetr_v4")
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported}")

    config_path, overrides = _resolve_model_config_request(model_config)
    return build_model_from_yaml(config_path, overrides=overrides)


def build_criterion(criterion_config: dict) -> nn.Module:
    """Build the configured training criterion.

    Supported criterion type right now:

    - ``rtdetr_v4``: build ``RTv4Criterion`` from the project-local RTv4 YAML stack
    """

    criterion_type = criterion_config.get("type", "rtdetr_v4")
    if criterion_type not in SUPPORTED_CRITERION_TYPES:
        supported = ", ".join(sorted(SUPPORTED_CRITERION_TYPES))
        raise ValueError(
            f"Unsupported criterion type: {criterion_type}. Supported types: {supported}"
        )

    config_path, overrides = _resolve_model_config_request(criterion_config)
    return build_criterion_from_yaml(config_path, overrides=overrides)


def build_postprocessor(postprocessor_config: dict) -> nn.Module:
    """Build the configured output postprocessor.

    Supported postprocessor type right now:

    - ``rtdetr_v4``: build the project-local ``PostProcessor`` from the RTv4 YAML stack
    """

    postprocessor_type = postprocessor_config.get("type", "rtdetr_v4")
    if postprocessor_type not in SUPPORTED_POSTPROCESSOR_TYPES:
        supported = ", ".join(sorted(SUPPORTED_POSTPROCESSOR_TYPES))
        raise ValueError(
            f"Unsupported postprocessor type: {postprocessor_type}. Supported types: {supported}"
        )

    config_path, overrides = _resolve_model_config_request(postprocessor_config)
    return build_postprocessor_from_yaml(config_path, overrides=overrides)


def build_teacher(teacher_config: dict) -> nn.Module:
    """Build the configured distillation teacher.

    Supported teacher type right now:

    - ``rtdetr_v4_teacher``: build ``teacher_model`` from one project-local
      RTv4 YAML config
    """

    teacher_type = teacher_config.get("type", "rtdetr_v4_teacher")
    if teacher_type not in SUPPORTED_TEACHER_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TEACHER_TYPES))
        raise ValueError(
            f"Unsupported teacher type: {teacher_type}. Supported types: {supported}"
        )

    config_path, overrides = _resolve_model_config_request(teacher_config)
    return build_teacher_from_yaml(config_path, overrides=overrides)


__all__ = [
    "SUPPORTED_CRITERION_TYPES",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_POSTPROCESSOR_TYPES",
    "SUPPORTED_TEACHER_TYPES",
    "build_criterion",
    "build_model",
    "build_postprocessor",
    "build_teacher",
]
