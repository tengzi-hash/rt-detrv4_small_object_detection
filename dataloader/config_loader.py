from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dataloader.entities import BatchConfig


GENERATED_DIRS = {"images", "labels", "unlabel_images", "unlabel_std", "manifests", "doublecheck"}
NON_RAW_DIRS = GENERATED_DIRS | {"double_check", "onhold"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path) -> Path:
    return Path(value).resolve()


def sample_fixes_to_overrides(sample_fixes: dict[str, Any]) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    list_fields = {
        "force_merge": "force_merge_classes",
        "drop": "drop_classes",
        "box_fixes": "box_fixes",
        "add_boxes": "add_boxes",
    }
    for label_stem, fix in sample_fixes.items():
        if not isinstance(fix, dict):
            continue
        key = str(label_stem)
        override: dict[str, Any] = {"label_path_endswith" if "/" in key or "\\" in key else "label_stem": key}
        if "remap" in fix:
            override["remap_classes"] = dict(fix["remap"] or {})
        for source, target in list_fields.items():
            if source in fix:
                override[target] = list(fix[source] or [])
        overrides.append(override)
    return overrides


def normalized_batch_rules(raw: dict[str, Any], global_rules: dict[str, Any]) -> dict[str, Any]:
    rules = dict(global_rules)
    rules.update(raw.get("rules") or {})
    for key in ("merge_overlapping_clamp", "drop_covered_by", "drop_clamp_covered_by_clamp_2"):
        if key in raw:
            rules[key] = raw[key]
    if "drop_clamp_covered_by_clamp_2" in rules and not isinstance(rules["drop_clamp_covered_by_clamp_2"], bool):
        rules["clamp_2_cover_threshold"] = float(rules["drop_clamp_covered_by_clamp_2"])
        rules["drop_clamp_covered_by_clamp_2"] = True

    overrides = list(rules.get("sample_overrides") or [])
    overrides.extend(sample_fixes_to_overrides(raw.get("sample_fixes") or {}))
    if overrides:
        rules["sample_overrides"] = overrides
    return rules


def discover_batches(config: dict[str, Any], output_dir: Path, source_overrides: list[str]) -> list[BatchConfig]:
    global_remap = dict(config.get("class_remap") or {})
    global_drop = set(config.get("drop_classes") or [])
    global_rules = dict(config.get("rules") or {})
    raw_batches = [{"name": Path(path).name, "path": path} for path in source_overrides] if source_overrides else list(config.get("batches") or [])
    if not raw_batches:
        exclude_names = set(config.get("exclude_dirs") or []) | NON_RAW_DIRS
        raw_batches = [
            {"name": child.name, "path": str(child)}
            for child in sorted(output_dir.iterdir())
            if child.is_dir() and child.name not in exclude_names
        ]

    batches: list[BatchConfig] = []
    for index, raw in enumerate(raw_batches, start=1):
        path = resolve_path(raw["path"])
        if not path.exists():
            raise FileNotFoundError(f"Batch path does not exist: {path}")
        remap = dict(global_remap)
        remap.update(raw.get("class_remap") or {})
        remap.update(raw.get("remap") or {})
        drop = set(global_drop)
        drop.update(raw.get("drop_classes") or [])
        batches.append(
            BatchConfig(
                name=str(raw.get("name") or f"batch_{index:02d}"),
                path=path,
                class_remap=remap,
                drop_classes=drop,
                rules=normalized_batch_rules(raw, global_rules),
            )
        )
    if not batches:
        raise RuntimeError("No raw batches found. Add batches in config or pass --source.")
    return batches


def duplicate_rules(config: dict[str, Any]) -> dict[str, Any]:
    legacy_policy = ((config.get("rules") or {}).get("duplicate_image_policy") or {}).get("same_classes_same_count") or {}
    return {
        "duplicate_min_iou": float(config.get("duplicate_min_iou", legacy_policy.get("min_pair_iou", 0.50))),
        "duplicate_merge_conflicts": bool(config.get("duplicate_merge_conflicts", False)),
        "duplicate_merge_iou": float(config.get("duplicate_merge_iou", config.get("duplicate_min_iou", legacy_policy.get("min_pair_iou", 0.50)))),
        "duplicate_prefer_label_path_contains": list(config.get("duplicate_prefer_label_path_contains") or []),
        "duplicate_keep": list(config.get("duplicate_keep") or []),
        "duplicate_same_image_scope": str(config.get("duplicate_same_image_scope", "same_image_name")),
    }
