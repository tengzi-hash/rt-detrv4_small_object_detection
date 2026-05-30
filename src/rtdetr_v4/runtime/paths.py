"""Runtime path normalization shared by local entrypoints."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..utils.misc import resolve_project_path


def resolve_config_paths(config: dict[str, Any], *, output_dir_override: str | None) -> dict[str, Any]:
    """Resolve relative paths in one loaded config against the project root."""
    resolved = copy.deepcopy(config)

    if output_dir_override:
        resolved["output_dir"] = str(resolve_project_path(output_dir_override))
        for index, stage in enumerate(resolved.get("stages", [])):
            stage_name = stage.get("name", f"stage{index + 1}")
            stage["output_dir"] = str(Path(resolved["output_dir"]) / stage_name)
    elif resolved.get("output_dir"):
        resolved["output_dir"] = str(resolve_project_path(resolved["output_dir"]))

    if resolved.get("student_checkpoint"):
        resolved["student_checkpoint"] = str(resolve_project_path(resolved["student_checkpoint"]))

    dataset_cfg = resolved.get("dataset", {})
    dataset_root = dataset_cfg.get("root")
    resolved_dataset_root = None
    if dataset_root:
        resolved_dataset_root = resolve_project_path(dataset_root)
        dataset_cfg["root"] = str(resolved_dataset_root)

    dataset_path_keys = (
        "class_summary_csv",
        "train_images",
        "train_annotations",
        "train_manifest",
        "train_split_csv",
        "val_images",
        "val_annotations",
        "val_manifest",
        "val_split_csv",
    )
    for key in dataset_path_keys:
        if dataset_cfg.get(key):
            path_value = Path(dataset_cfg[key])
            if resolved_dataset_root is not None and not path_value.is_absolute():
                dataset_cfg[key] = str((resolved_dataset_root / path_value).resolve())
            else:
                dataset_cfg[key] = str(resolve_project_path(path_value))

    teacher_cfg = resolved.get("teacher_model")
    if isinstance(teacher_cfg, dict):
        for key in ("dinov3_repo_path", "dinov3_weights_path", "dinov2_repo_path", "weights_path"):
            if teacher_cfg.get(key):
                teacher_cfg[key] = str(resolve_project_path(teacher_cfg[key]))

    for key in ("HGNetv2", "PResNet"):
        component_cfg = resolved.get(key)
        if isinstance(component_cfg, dict) and component_cfg.get("local_model_dir"):
            component_cfg["local_model_dir"] = str(resolve_project_path(component_cfg["local_model_dir"]))

    for stage in resolved.get("stages", []):
        if stage.get("output_dir"):
            stage["output_dir"] = str(resolve_project_path(stage["output_dir"]))

    train_cfg = resolved.get("train")
    if isinstance(train_cfg, dict) and train_cfg.get("output_dir"):
        train_cfg["output_dir"] = str(resolve_project_path(train_cfg["output_dir"]))

    return resolved


__all__ = ["resolve_config_paths"]
