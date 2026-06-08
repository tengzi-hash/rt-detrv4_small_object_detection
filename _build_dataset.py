"""Build the locomotive detection dataset from raw batches.

Pipeline:
1. scan images/XML;
2. separate image-without-label and label-without-image;
3. normalize class names;
4. drop configured classes/boxes;
5. resolve duplicate names/images;
6. write images, labels, unlabel_images, double_check, manifests;
7. create train/val split.

Default mode writes the cleaned dataset. Use ``--dry-run`` to preview only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GENERATED_DIRS = {"images", "labels", "unlabel_images", "manifests"}
NON_RAW_DIRS = GENERATED_DIRS | {"double_check", "onhold"}


@dataclass
class Batch:
    name: str
    path: Path
    class_remap: dict[str, str]
    drop_classes: set[str]
    rules: dict[str, Any]


@dataclass(frozen=True)
class Box:
    cls: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def signature(self) -> tuple[str, int, int, int, int]:
        return (self.cls, self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass(frozen=True)
class RawObject:
    cls: str
    box: Box | None


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    label_path: Path
    image_hash: str
    boxes: list[Box]
    width: int
    height: int
    batch_name: str

    @property
    def label_signature(self) -> tuple[tuple[str, int, int, int, int], ...]:
        return tuple(sorted(box.signature() for box in self.boxes))

    @property
    def classes(self) -> list[str]:
        return sorted({box.cls for box in self.boxes})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned RT-DETRv4 dataset.")
    parser.add_argument("--config", default="configs/dataset_build.yml", help="Dataset build config.")
    parser.add_argument("--source", action="append", default=[], help="Override/add raw source directory.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; do not write generated dataset files.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of hard-linking where possible.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path) -> Path:
    return Path(value).resolve()


def sample_fixes_to_overrides(sample_fixes: dict[str, Any]) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    for label_stem, fix in sample_fixes.items():
        if not isinstance(fix, dict):
            continue
        key = str(label_stem)
        if "/" in key or "\\" in key:
            override: dict[str, Any] = {"label_path_endswith": key}
        else:
            override = {"label_stem": key}
        if "force_merge" in fix:
            override["force_merge_classes"] = list(fix["force_merge"] or [])
        if "remap" in fix:
            override["remap_classes"] = dict(fix["remap"] or {})
        if "drop" in fix:
            override["drop_classes"] = list(fix["drop"] or [])
        if "box_fixes" in fix:
            override["box_fixes"] = list(fix["box_fixes"] or [])
        if "add_boxes" in fix:
            override["add_boxes"] = list(fix["add_boxes"] or [])
        overrides.append(override)
    return overrides


def normalized_batch_rules(raw: dict[str, Any], global_rules: dict[str, Any]) -> dict[str, Any]:
    rules = dict(global_rules)
    rules.update(raw.get("rules") or {})

    if "merge_overlapping_clamp" in raw:
        rules["merge_overlapping_clamp"] = bool(raw["merge_overlapping_clamp"])

    if "drop_covered_by" in raw:
        rules["drop_covered_by"] = list(raw["drop_covered_by"] or [])

    if "drop_clamp_covered_by_clamp_2" in raw:
        value = raw["drop_clamp_covered_by_clamp_2"]
        rules["drop_clamp_covered_by_clamp_2"] = bool(value)
        if not isinstance(value, bool):
            rules["clamp_2_cover_threshold"] = float(value)

    overrides = list(rules.get("sample_overrides") or [])
    overrides.extend(sample_fixes_to_overrides(raw.get("sample_fixes") or {}))
    if overrides:
        rules["sample_overrides"] = overrides
    return rules


def duplicate_rules(config: dict[str, Any]) -> dict[str, Any]:
    legacy_policy = ((config.get("rules") or {}).get("duplicate_image_policy") or {}).get("same_classes_same_count") or {}
    return {
        "duplicate_min_iou": float(config.get("duplicate_min_iou", legacy_policy.get("min_pair_iou", 0.50))),
        "duplicate_keep": list(config.get("duplicate_keep") or []),
    }


def discover_batches(config: dict[str, Any], output_dir: Path, source_overrides: list[str]) -> list[Batch]:
    global_remap = dict(config.get("class_remap") or {})
    global_drop = set(config.get("drop_classes") or [])
    global_rules = dict(config.get("rules") or {})

    if source_overrides:
        raw_batches = [{"name": Path(path).name, "path": path} for path in source_overrides]
    else:
        raw_batches = list(config.get("batches") or [])
        if not raw_batches:
            exclude_names = set(config.get("exclude_dirs") or []) | NON_RAW_DIRS
            raw_batches = [
                {"name": child.name, "path": str(child)}
                for child in sorted(output_dir.iterdir())
                if child.is_dir() and child.name not in exclude_names
            ]

    batches: list[Batch] = []
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
            Batch(
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


def excluded(path: Path, exclude_dirs: set[str]) -> bool:
    return any(part in exclude_dirs for part in path.parts)


def scan_batches(batches: list[Batch], exclude_dirs: set[str]) -> tuple[list[Path], list[tuple[Path, Batch]]]:
    images: list[Path] = []
    labels: list[tuple[Path, Batch]] = []
    for batch in batches:
        for path in batch.path.rglob("*"):
            if not path.is_file() or excluded(path, exclude_dirs):
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                images.append(path)
            elif suffix == ".xml":
                labels.append((path, batch))
    return sorted(set(images)), sorted(labels, key=lambda item: str(item[0]))


def image_indexes(images: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    by_name: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_stem[image.stem.lower()].append(image)
        by_name[image.name.lower()].append(image)
    return by_stem, by_name


def nearest(paths: list[Path], anchor: Path) -> Path:
    anchor_parent = anchor.parent.resolve()

    def key(path: Path) -> tuple[int, int, str]:
        parent = path.parent.resolve()
        same_parent = 0 if parent == anchor_parent else 1
        try:
            common_depth = len(Path(os.path.commonpath([str(parent), str(anchor_parent)])).parts)
        except ValueError:
            common_depth = 0
        return (same_parent, -common_depth, str(path))

    return sorted(paths, key=key)[0]


def match_image(label_path: Path, filename: str | None, by_stem: dict[str, list[Path]], by_name: dict[str, list[Path]]) -> Path | None:
    candidates: list[Path] = []
    if filename:
        raw_name = Path(filename.strip()).name
        direct = label_path.parent / raw_name
        if direct.is_file():
            return direct
        candidates.extend(by_name.get(raw_name.lower(), []))
        candidates.extend(by_stem.get(Path(raw_name).stem.lower(), []))
    for suffix in IMAGE_SUFFIXES:
        direct = label_path.with_suffix(suffix)
        if direct.is_file():
            return direct
    candidates.extend(by_stem.get(label_path.stem.lower(), []))
    candidates = sorted(set(candidates))
    return nearest(candidates, label_path) if candidates else None


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_read_error(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return str(exc)
    return None


def assign_image_hashes(samples: list[Sample]) -> None:
    """Assign exact hashes only where duplicate image bytes are possible.

    Identical files must have the same byte size, so unique-size images do not
    need full-file hashing. This keeps repeated-image detection exact while
    avoiding a slow full scan of every JPEG on each build.
    """
    by_size: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        try:
            size = sample.image_path.stat().st_size
        except OSError:
            size = -1
        by_size[size].append(sample)

    for size, group in by_size.items():
        if len(group) == 1:
            sample = group[0]
            sample.image_hash = f"unique:{size}:{sample.image_path.resolve()}"
            continue
        hash_cache: dict[Path, str] = {}
        for sample in group:
            sample.image_hash = hash_cache.setdefault(sample.image_path, file_sha1(sample.image_path))


def int_text(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(round(float(value.strip())))
    except ValueError:
        return None


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a.xmin, b.xmin), max(a.ymin, b.ymin)
    ix1, iy1 = min(a.xmax, b.xmax), min(a.ymax, b.ymax)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0, a.xmax - a.xmin) * max(0, a.ymax - a.ymin)
    area_b = max(0, b.xmax - b.xmin) * max(0, b.ymax - b.ymin)
    return inter / max(area_a + area_b - inter, 1)


def coverage_ratio(inner: Box, outer: Box) -> float:
    ix0, iy0 = max(inner.xmin, outer.xmin), max(inner.ymin, outer.ymin)
    ix1, iy1 = min(inner.xmax, outer.xmax), min(inner.ymax, outer.ymax)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(0, inner.xmax - inner.xmin) * max(0, inner.ymax - inner.ymin)
    return inter / max(area, 1)


def parse_object_box(obj: ET.Element, class_name: str) -> Box | None:
    box_el = obj.find("bndbox")
    if box_el is None:
        return None
    coords = [int_text(box_el.findtext(name)) for name in ("xmin", "ymin", "xmax", "ymax")]
    if any(value is None for value in coords):
        return None
    xmin, ymin, xmax, ymax = [int(value) for value in coords if value is not None]
    if xmax <= xmin or ymax <= ymin:
        return None
    return Box(class_name, xmin, ymin, xmax, ymax)


def read_raw_objects(root: ET.Element, label_path: Path) -> tuple[list[RawObject], list[dict[str, str]]]:
    raw_objects: list[RawObject] = []
    issues: list[dict[str, str]] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        if not raw_name:
            issues.append(issue(label_path, "object_without_name", ""))
            continue
        box = parse_object_box(obj, raw_name)
        if box is None:
            issues.append(issue(label_path, "bad_or_missing_box", raw_name))
            continue
        raw_objects.append(RawObject(raw_name, box))
    return raw_objects, issues


def merge_overlapping_clamps(boxes: list[Box]) -> tuple[list[Box], int]:
    clamp_ids = [index for index, box in enumerate(boxes) if box.cls == "Clamp"]
    if len(clamp_ids) < 2:
        return boxes, 0

    used: set[int] = set()
    merged: list[Box] = []
    removed = 0
    for index, box in enumerate(boxes):
        if index in used:
            continue
        if box.cls != "Clamp":
            merged.append(box)
            continue
        group = [index]
        for other_index in clamp_ids:
            if other_index <= index or other_index in used:
                continue
            if iou(boxes[index], boxes[other_index]) > 0:
                group.append(other_index)
        if len(group) == 1:
            merged.append(box)
            continue
        group_boxes = [boxes[item] for item in group]
        merged.append(
            Box(
                "Clamp",
                min(item.xmin for item in group_boxes),
                min(item.ymin for item in group_boxes),
                max(item.xmax for item in group_boxes),
                max(item.ymax for item in group_boxes),
            )
        )
        used.update(group[1:])
        removed += len(group) - 1
    return merged, removed


def force_merge_class_boxes(boxes: list[Box], class_name: str) -> tuple[list[Box], int]:
    class_indices = [index for index, box in enumerate(boxes) if box.cls == class_name]
    if len(class_indices) < 2:
        return boxes, 0

    target_boxes = [boxes[index] for index in class_indices]
    merged_box = Box(
        class_name,
        min(box.xmin for box in target_boxes),
        min(box.ymin for box in target_boxes),
        max(box.xmax for box in target_boxes),
        max(box.ymax for box in target_boxes),
    )
    first_index = min(class_indices)
    skip_indices = set(class_indices) - {first_index}
    rebuilt: list[Box] = []
    for index, box in enumerate(boxes):
        if index in skip_indices:
            continue
        rebuilt.append(merged_box if index == first_index else box)
    return rebuilt, len(skip_indices)


def drop_covered_boxes(boxes: list[Box], rules: list[dict[str, Any]]) -> tuple[list[Box], Counter]:
    dropped = Counter()
    for rule in rules:
        inner = str(rule.get("inner") or rule.get("drop") or "")
        outer = str(rule.get("outer") or rule.get("covered_by") or "")
        if not inner or not outer:
            continue
        threshold = float(rule.get("threshold", rule.get("min_coverage", 0.95)))
        outer_boxes = [box for box in boxes if box.cls == outer]
        if not outer_boxes:
            continue
        kept: list[Box] = []
        for box in boxes:
            if box.cls == inner and any(coverage_ratio(box, outer_box) >= threshold for outer_box in outer_boxes):
                dropped[f"{inner}_covered_by_{outer}"] += 1
                continue
            kept.append(box)
        boxes = kept
    return boxes, dropped


def apply_box_fixes(boxes: list[Box], fixes: list[dict[str, Any]]) -> tuple[list[Box], int]:
    fixed: list[Box] = []
    changed = 0
    for box in boxes:
        replacement = box
        for fix in fixes:
            coords = fix.get("box") or []
            if len(coords) != 4:
                continue
            from_cls = str(fix.get("from") or box.cls)
            to_cls = str(fix.get("to") or fix.get("cls") or box.cls)
            if box.cls == from_cls and [box.xmin, box.ymin, box.xmax, box.ymax] == [int(value) for value in coords]:
                replacement = Box(to_cls, box.xmin, box.ymin, box.xmax, box.ymax)
                if replacement.cls != box.cls:
                    changed += 1
                break
        fixed.append(replacement)
    return fixed, changed


def add_configured_boxes(boxes: list[Box], additions: list[dict[str, Any]]) -> tuple[list[Box], int]:
    added = 0
    for item in additions:
        coords = item.get("box") or []
        cls = str(item.get("cls") or item.get("class") or "")
        if len(coords) != 4 or not cls:
            continue
        xmin, ymin, xmax, ymax = [int(value) for value in coords]
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(Box(cls, xmin, ymin, xmax, ymax))
        added += 1
    return boxes, added


def override_matches_label(override: dict[str, Any], label_path: Path) -> bool:
    label_stem = override.get("label_stem")
    if label_stem and str(label_stem) == label_path.stem:
        return True
    label_name = override.get("label_name")
    if label_name and str(label_name) == label_path.name:
        return True
    label_path_suffix = override.get("label_path_endswith")
    if label_path_suffix and str(label_path).replace("\\", "/").endswith(str(label_path_suffix).replace("\\", "/")):
        return True
    return False


def matching_overrides(batch: Batch, label_path: Path) -> list[dict[str, Any]]:
    return [
        override
        for override in batch.rules.get("sample_overrides") or []
        if isinstance(override, dict) and override_matches_label(override, label_path)
    ]


def apply_sample_overrides(boxes: list[Box], label_path: Path, batch: Batch, stats: Counter) -> list[Box]:
    for override in matching_overrides(batch, label_path):
        remap_classes = override.get("remap_classes") or {}
        if remap_classes:
            remapped: list[Box] = []
            for box in boxes:
                new_class = remap_classes.get(box.cls, box.cls)
                if new_class != box.cls:
                    stats[f"sample_override_remapped_{box.cls}_to_{new_class}"] += 1
                remapped.append(Box(new_class, box.xmin, box.ymin, box.xmax, box.ymax))
            boxes = remapped

        box_fixes = override.get("box_fixes") or []
        if box_fixes:
            boxes, fixed_count = apply_box_fixes(boxes, box_fixes)
            stats["sample_override_box_fixes"] += fixed_count

        drop_classes = set(override.get("drop_classes") or [])
        if drop_classes:
            before_counts = Counter(box.cls for box in boxes)
            boxes = [box for box in boxes if box.cls not in drop_classes]
            for class_name in sorted(drop_classes):
                stats[f"sample_override_dropped_{class_name}"] += before_counts[class_name]

        add_boxes = override.get("add_boxes") or []
        if add_boxes:
            boxes, added_count = add_configured_boxes(boxes, add_boxes)
            stats["sample_override_added_boxes"] += added_count

        for class_name in override.get("force_merge_classes") or []:
            boxes, removed = force_merge_class_boxes(boxes, str(class_name))
            stats[f"sample_override_force_merged_{class_name}"] += removed
    return boxes


def clean_raw_objects(raw_objects: list[RawObject], batch: Batch, label_path: Path, stats: Counter) -> list[Box]:
    raw_clamp_2_boxes = [obj.box for obj in raw_objects if obj.cls == "Clamp_2" and obj.box is not None]
    sample_remap = {
        raw_cls: new_cls
        for override in matching_overrides(batch, label_path)
        for raw_cls, new_cls in (override.get("remap_classes") or {}).items()
    }
    boxes: list[Box] = []
    for raw in raw_objects:
        if raw.box is None:
            continue
        if batch.rules.get("drop_clamp_covered_by_clamp_2", False) and raw.cls == "Clamp":
            threshold = float(batch.rules.get("clamp_2_cover_threshold", 0.90))
            if any(coverage_ratio(raw.box, clamp_2_box) >= threshold for clamp_2_box in raw_clamp_2_boxes):
                stats["dropped_clamp_covered_by_clamp_2"] += 1
                continue
        if raw.cls in batch.drop_classes:
            cls = sample_remap.get(raw.cls)
            if not cls:
                stats[f"dropped_{raw.cls}"] += 1
                continue
            stats[f"sample_override_remapped_{raw.cls}_to_{cls}"] += 1
        else:
            cls = batch.class_remap.get(raw.cls, raw.cls)
        boxes.append(Box(cls, raw.box.xmin, raw.box.ymin, raw.box.xmax, raw.box.ymax))

    if batch.rules.get("merge_overlapping_clamp", False):
        boxes, removed = merge_overlapping_clamps(boxes)
        stats["merged_overlapping_clamp_boxes"] += removed

    boxes = apply_sample_overrides(boxes, label_path, batch, stats)
    boxes, covered_drops = drop_covered_boxes(boxes, list(batch.rules.get("drop_covered_by") or []))
    for key, count in covered_drops.items():
        stats[f"dropped_{key}"] += count
    return boxes


def annotation_size(root: ET.Element, image_path: Path) -> tuple[int, int]:
    width = int_text(root.findtext("size/width")) or 0
    height = int_text(root.findtext("size/height")) or 0
    if width <= 0 or height <= 0:
        return image_size(image_path)
    return width, height


def parse_label(label_path: Path, batch: Batch, image_path: Path) -> tuple[list[Box], int, int, list[dict[str, str]], Counter]:
    stats = Counter()
    root = ET.parse(label_path).getroot()
    raw_objects, issues = read_raw_objects(root, label_path)
    boxes = clean_raw_objects(raw_objects, batch, label_path, stats)
    width, height = annotation_size(root, image_path)
    return boxes, width, height, issues, stats


def issue(label: Path | str, issue_type: str, detail: str) -> dict[str, str]:
    return {"label": str(label), "issue": issue_type, "detail": detail}


def make_sample_id(sample: Sample, same_name_policy: str, seen: Counter) -> str:
    stem = sample.label_path.stem
    seen[stem] += 1
    if seen[stem] == 1:
        return stem
    if same_name_policy == "batch_prefix":
        prefix = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", sample.batch_name).strip("_")
        return f"{prefix}__{stem}"
    return f"{stem}__dup{seen[stem]:03d}"


def box_area(box: Box) -> int:
    return max(0, box.xmax - box.xmin) * max(0, box.ymax - box.ymin)


def mean_box_area(sample: Sample) -> float:
    return sum(box_area(box) for box in sample.boxes) / max(len(sample.boxes), 1)


def class_multiset(sample: Sample) -> Counter:
    return Counter(box.cls for box in sample.boxes)


def counter_contains(left: Counter, right: Counter) -> bool:
    return all(left[key] >= value for key, value in right.items())


def same_source_name(group: list[Sample]) -> bool:
    stems = {sample.label_path.stem for sample in group}
    image_names = {sample.image_path.name for sample in group}
    return len(stems) == 1 or len(image_names) == 1


def boxes_match_by_class(left: Sample, right: Sample, min_iou: float) -> bool:
    if len(left.boxes) != len(right.boxes) or class_multiset(left) != class_multiset(right):
        return False

    for class_name in class_multiset(left):
        left_boxes = [box for box in left.boxes if box.cls == class_name]
        right_boxes = [box for box in right.boxes if box.cls == class_name]
        pairs = sorted(
            ((iou(a, b), left_index, right_index) for left_index, a in enumerate(left_boxes) for right_index, b in enumerate(right_boxes)),
            reverse=True,
        )
        matched_left: set[int] = set()
        matched_right: set[int] = set()
        for score, left_index, right_index in pairs:
            if score < min_iou:
                break
            if left_index in matched_left or right_index in matched_right:
                continue
            matched_left.add(left_index)
            matched_right.add(right_index)
        if len(matched_left) != len(left_boxes):
            return False
    return True


def auto_resolve_same_name_superset(group: list[Sample]) -> Sample | None:
    if not same_source_name(group):
        return None

    class_counts = [(sample, class_multiset(sample)) for sample in group]
    candidates: list[Sample] = []
    for sample, counts in class_counts:
        if all(counter_contains(counts, other_counts) for other, other_counts in class_counts if other is not sample):
            if any(sum(counts.values()) > sum(other_counts.values()) for other, other_counts in class_counts if other is not sample):
                candidates.append(sample)

    if len(candidates) != 1:
        return None
    return candidates[0]


def auto_resolve_repeated_image_group(group: list[Sample], policy: dict[str, Any]) -> tuple[Sample | None, str]:
    selected = auto_resolve_same_name_superset(group)
    if selected is not None:
        return selected, "repeated_image_same_name_class_superset_auto_resolved"

    min_iou = float(policy.get("duplicate_min_iou", 0.50))
    reference = group[0]
    if not all(boxes_match_by_class(reference, other, min_iou) for other in group[1:]):
        return None, ""

    return min(group, key=lambda sample: (mean_box_area(sample), str(sample.label_path))), "repeated_image_same_classes_boxes_auto_resolved"


def choose_duplicate_sample(group: list[Sample], policy: dict[str, Any]) -> tuple[Sample | None, str]:
    label_signatures = {sample.label_signature for sample in group}
    if len(label_signatures) == 1:
        return group[0], "repeated_image_same_label_kept_one"
    return auto_resolve_repeated_image_group(group, policy)


def path_suffix_matches(path: Path, suffix: str) -> bool:
    return str(path).replace("\\", "/").endswith(str(suffix).replace("\\", "/"))


def configured_duplicate_keep(group: list[Sample], rules: dict[str, Any]) -> Sample | None:
    image_hash = group[0].image_hash
    for item in rules.get("duplicate_keep") or []:
        if str(item.get("image_hash") or "") != image_hash:
            continue
        keep = str(item.get("keep") or "")
        matches = [sample for sample in group if path_suffix_matches(sample.label_path, keep)]
        if len(matches) == 1:
            return matches[0]
    return None


def clean_and_pair(labels: list[tuple[Path, Batch]], images: list[Path], config: dict[str, Any]) -> tuple[list[Sample], list[Path], list[dict[str, str]], Counter]:
    by_stem, by_name = image_indexes(images)
    used_images: set[Path] = set()
    bad_images: set[Path] = set()
    samples: list[Sample] = []
    unlabel_images: list[Path] = []
    issues: list[dict[str, str]] = []
    stats = Counter()

    for label_path, batch in labels:
        try:
            root = ET.parse(label_path).getroot()
        except Exception as exc:
            issues.append(issue(label_path, "bad_xml", str(exc)))
            continue
        image_path = match_image(label_path, root.findtext("filename"), by_stem, by_name)
        if image_path is None:
            issues.append(issue(label_path, "label_without_image", "discarded"))
            continue
        if error := image_read_error(image_path):
            bad_images.add(image_path)
            used_images.add(image_path)
            issues.append(issue(label_path, "image_unreadable", f"{image_path}; {error}"))
            continue
        used_images.add(image_path)
        try:
            boxes, width, height, label_issues, label_stats = parse_label(label_path, batch, image_path)
        except Exception as exc:
            issues.append(issue(label_path, "parse_failed", str(exc)))
            continue
        issues.extend(label_issues)
        stats.update(label_stats)
        if not boxes:
            unlabel_images.append(image_path)
            issues.append(issue(label_path, "empty_label", str(image_path)))
            continue
        samples.append(
            Sample(
                sample_id=label_path.stem,
                image_path=image_path,
                label_path=label_path,
                image_hash="",
                boxes=boxes,
                width=width,
                height=height,
                batch_name=batch.name,
            )
        )

    for image in images:
        if image in used_images:
            continue
        if error := image_read_error(image):
            bad_images.add(image)
            issues.append(issue(image, "unlabel_image_unreadable", error))
            continue
        unlabel_images.append(image)
    assign_image_hashes(samples)
    return samples, sorted(set(unlabel_images)), issues, stats


def resolve_duplicates(samples: list[Sample], rules: dict[str, Any]) -> tuple[list[Sample], list[list[Sample]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    conflicts: list[list[Sample]] = []
    kept: list[Sample] = []

    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.image_hash].append(sample)

    for image_hash, group in sorted(by_hash.items()):
        if len(group) == 1:
            kept.append(group[0])
            continue
        selected = configured_duplicate_keep(group, rules)
        if selected is not None:
            kept.append(selected)
            issues.append(issue(selected.label_path, "repeated_image_configured_keep", f"image_hash={image_hash}; count={len(group)}; kept={selected.label_path.name}"))
            continue
        selected, issue_type = choose_duplicate_sample(group, rules)
        if selected is None:
            conflicts.append(group)
            issues.append(issue(group[0].label_path, "repeated_image_label_conflict", f"image_hash={image_hash}; count={len(group)}"))
            continue
        kept.append(selected)
        issues.append(issue(selected.label_path, issue_type, f"image_hash={image_hash}; count={len(group)}; kept={selected.label_path.name}"))

    seen_stems = Counter()
    output_id_seen: set[str] = set()
    same_name_policy = "batch_prefix"
    for sample in sorted(kept, key=lambda item: (item.label_path.stem, item.batch_name, str(item.label_path))):
        sample.sample_id = make_sample_id(sample, same_name_policy, seen_stems)
        while sample.sample_id in output_id_seen:
            sample.sample_id = f"{sample.sample_id}__{len(output_id_seen):06d}"
        output_id_seen.add(sample.sample_id)
    return kept, conflicts, issues


def copy_or_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def write_xml(sample: Sample, path: Path, image_name: str) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(sample.width)
    ET.SubElement(size, "height").text = str(sample.height)
    ET.SubElement(size, "depth").text = "3"
    for box in sample.boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box.cls
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(box.xmin)
        ET.SubElement(bb, "ymin").text = str(box.ymin)
        ET.SubElement(bb, "xmax").text = str(box.xmax)
        ET.SubElement(bb, "ymax").text = str(box.ymax)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8")


def draw_visualization(sample: Sample, output_path: Path, title: str) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    try:
        image = Image.open(sample.image_path).convert("RGB")
    except Exception:
        return
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, min(image.width - 1, 900), 34], fill=(20, 20, 20))
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)
    for index, box in enumerate(sample.boxes, start=1):
        color = (231, 76, 60) if index % 3 == 1 else (52, 152, 219) if index % 3 == 2 else (46, 204, 113)
        draw.rectangle([box.xmin, box.ymin, box.xmax, box.ymax], outline=color, width=3)
        draw.text((box.xmin, max(0, box.ymin - 18)), f"{index}:{box.cls}", fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_double_check(output_dir: Path, conflicts: list[list[Sample]], copy: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = output_dir / "double_check" / "repeated_image_label_conflict"
    if root.exists():
        shutil.rmtree(root)
    for group_index, group in enumerate(conflicts, start=1):
        group_id = f"group_{group_index:04d}"
        for candidate_index, sample in enumerate(group, start=1):
            candidate = f"candidate_{candidate_index:02d}"
            stem = f"{group_id}__{candidate}__{sample.label_path.stem}"
            image_dst = root / "images" / f"{stem}{sample.image_path.suffix.lower()}"
            label_dst = root / "labels" / f"{stem}.xml"
            vis_dst = root / "visualizations" / f"{stem}.jpg"
            copy_or_link(sample.image_path, image_dst, copy)
            write_xml(sample, label_dst, image_dst.name)
            draw_visualization(sample, vis_dst, f"{group_id} {candidate} boxes={len(sample.boxes)} batch={sample.batch_name}")
            rows.append(
                {
                    "group_id": group_id,
                    "candidate_id": candidate,
                    "image_hash": sample.image_hash,
                    "source_batch": sample.batch_name,
                    "source_image_path": str(sample.image_path),
                    "source_label_path": str(sample.label_path),
                    "classes": "|".join(sample.classes),
                    "num_boxes": len(sample.boxes),
                    "visualization_path": str(vis_dst),
                    "reason": "same image hash with different label signature",
                }
            )
    if rows:
        write_csv(root / "index.csv", rows, list(rows[0].keys()))
    return rows


def group_key(sample: Sample, split_cfg: dict[str, Any]) -> str:
    parts = []
    for key in split_cfg.get("group_by") or ["image_hash", "train_id", "capture_time"]:
        if key == "image_hash":
            parts.append(sample.image_hash)
        elif key == "train_id":
            match = re.search(r"(HXD[0-9A-Z]+)", sample.sample_id)
            parts.append(match.group(1) if match else "")
        elif key == "capture_time":
            match = re.search(r"(20\d{6})-(\d{3,4})", sample.sample_id)
            parts.append(f"{match.group(1)}{match.group(2).zfill(4)}" if match else "")
        else:
            parts.append("")
    key = "_".join(part for part in parts if part)
    return key or sample.image_hash


def split_samples(samples: list[Sample], split_cfg: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    seed = int(split_cfg.get("seed", 20260603))

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[group_key(sample, split_cfg)].append(sample)

    target = round(len(samples) * val_ratio)
    train_only_max_files = int(split_cfg.get("train_only_class_max_files", 2))
    class_files = Counter()
    for sample in samples:
        class_files.update(sample.classes)
    train_only_classes = {cls for cls, count in class_files.items() if count <= train_only_max_files}
    val_coverable_classes = set(class_files) - train_only_classes
    group_classes = {name: Counter(cls for sample in group for cls in sample.classes) for name, group in groups.items()}
    eligible = [
        name
        for name, counts in group_classes.items()
        if not (set(counts) & train_only_classes)
        and all(class_files[cls] - count > 0 for cls, count in counts.items())
    ]
    val_groups = set()
    covered = Counter()
    total = 0

    while eligible and (set(covered) & val_coverable_classes) != val_coverable_classes:
        useful = [name for name in eligible if set(group_classes[name]) & (val_coverable_classes - set(covered))]
        if not useful:
            break
        best = min(
            useful,
            key=lambda name: (
                -len(set(group_classes[name]) & (val_coverable_classes - set(covered))),
                max(0, total + len(groups[name]) - target),
                abs(target - (total + len(groups[name]))),
                len(groups[name]),
                name,
            ),
        )
        val_groups.add(best)
        covered.update(group_classes[best])
        total += len(groups[best])
        eligible.remove(best)

    while eligible and total < target:
        best = min(
            eligible,
            key=lambda name: (
                max(0, total + len(groups[name]) - target),
                abs(target - (total + len(groups[name]))),
                -len(set(group_classes[name]) & (val_coverable_classes - set(covered))),
                len(groups[name]),
                name,
            ),
        )
        val_groups.add(best)
        covered.update(group_classes[best])
        total += len(groups[best])
        eligible.remove(best)

    val_ids = {sample.sample_id for group in val_groups for sample in groups[group]}
    return val_ids, {
        "strategy": "group_stratified",
        "seed": seed,
        "total_groups": len(groups),
        "train_groups": len(groups) - len(val_groups),
        "val_groups": len(val_groups),
        "train_only_class_max_files": int(split_cfg.get("train_only_class_max_files", 2)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unlabel_image_name(image: Path, seen: Counter) -> str:
    name = f"{image.stem}{image.suffix.lower()}"
    seen[name.lower()] += 1
    if seen[name.lower()] == 1:
        return name
    path_digest = hashlib.sha1(str(image.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{image.stem}__unlabel_{path_digest}{image.suffix.lower()}"


def write_outputs(output_dir: Path, samples: list[Sample], unlabels: list[Path], conflicts: list[list[Sample]], issues: list[dict[str, str]], config: dict[str, Any], copy: bool) -> None:
    for sample in samples:
        image_name = f"{sample.sample_id}{sample.image_path.suffix.lower()}"
        copy_or_link(sample.image_path, output_dir / "images" / image_name, copy)
        write_xml(sample, output_dir / "labels" / f"{sample.sample_id}.xml", image_name)
    seen_unlabels = Counter()
    for image in unlabels:
        copy_or_link(image, output_dir / "unlabel_images" / unlabel_image_name(image, seen_unlabels), copy)
    write_double_check(output_dir, conflicts, copy)
    write_manifests(output_dir, samples, unlabels, issues, config)


def class_stats(samples: list[Sample]) -> tuple[list[str], Counter, Counter]:
    class_instances = Counter(box.cls for sample in samples for box in sample.boxes)
    class_files = Counter()
    for sample in samples:
        class_files.update(sample.classes)
    return sorted(class_instances), class_instances, class_files


def write_class_manifests(manifest_dir: Path, classes: list[str], class_instances: Counter, class_files: Counter) -> None:
    write_csv(
        manifest_dir / "class_mapping.csv",
        [{"class_id": i, "class_name": cls, "instances": class_instances[cls], "label_files": class_files[cls]} for i, cls in enumerate(classes)],
        ["class_id", "class_name", "instances", "label_files"],
    )
    (manifest_dir / "classes.txt").write_text("".join(f"{cls}\n" for cls in classes), encoding="utf-8")
    write_csv(
        manifest_dir / "class_counts.csv",
        [{"class_name": cls, "instances": class_instances[cls], "label_files": class_files[cls]} for cls in sorted(classes, key=lambda c: (-class_instances[c], c))],
        ["class_name", "instances", "label_files"],
    )


def sample_manifest_rows(samples: list[Sample]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "image": f"{sample.sample_id}{sample.image_path.suffix.lower()}",
            "label": f"{sample.sample_id}.xml",
            "num_instances": len(sample.boxes),
            "classes": "|".join(sample.classes),
            "image_hash": sample.image_hash,
            "source_batch": sample.batch_name,
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
    ]


def split_manifest_data(samples: list[Sample], split_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Counter], dict[str, defaultdict[str, set[str]]]]:
    val_ids, split_meta = split_samples(samples, split_cfg)
    split_rows = []
    dist = {"train": Counter(), "val": Counter()}
    dist_files = {"train": defaultdict(set), "val": defaultdict(set)}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        split = "val" if sample.sample_id in val_ids else "train"
        label_name = f"{sample.sample_id}.xml"
        for box in sample.boxes:
            dist[split][box.cls] += 1
        for cls in sample.classes:
            dist_files[split][cls].add(label_name)
        split_rows.append(
            {
                "split": split,
                "sample_id": sample.sample_id,
                "image": f"{sample.sample_id}{sample.image_path.suffix.lower()}",
                "label": label_name,
                "group": group_key(sample, split_cfg),
                "classes": "|".join(sample.classes),
            }
        )
    return split_rows, split_meta, dist, dist_files


def write_split_manifests(manifest_dir: Path, split_rows: list[dict[str, Any]], classes: list[str], class_instances: Counter, class_files: Counter, dist: dict[str, Counter], dist_files: dict[str, defaultdict[str, set[str]]]) -> None:
    write_csv(manifest_dir / "train_split.csv", [{"relative_label_path": row["label"]} for row in split_rows if row["split"] == "train"], ["relative_label_path"])
    write_csv(manifest_dir / "val_split.csv", [{"relative_label_path": row["label"]} for row in split_rows if row["split"] == "val"], ["relative_label_path"])
    write_csv(manifest_dir / "split_samples.csv", split_rows, ["split", "sample_id", "image", "label", "group", "classes"])
    write_csv(
        manifest_dir / "split_class_distribution.csv",
        [
            {
                "class_name": cls,
                "total_instances": class_instances[cls],
                "train_instances": dist["train"][cls],
                "val_instances": dist["val"][cls],
                "total_files": class_files[cls],
                "train_files": len(dist_files["train"][cls]),
                "val_files": len(dist_files["val"][cls]),
            }
            for cls in classes
        ],
        ["class_name", "total_instances", "train_instances", "val_instances", "total_files", "train_files", "val_files"],
    )


def write_summary_manifests(manifest_dir: Path, samples: list[Sample], unlabels: list[Path], issues: list[dict[str, str]], classes: list[str], class_instances: Counter, split_rows: list[dict[str, Any]], split_meta: dict[str, Any], dist_files: dict[str, defaultdict[str, set[str]]]) -> None:
    train_n = sum(1 for row in split_rows if row["split"] == "train")
    val_n = len(split_rows) - train_n
    summary = {
        "images": len(samples),
        "labels": len(samples),
        "paired_samples": len(samples),
        "images_without_label": 0,
        "labels_without_image": sum(1 for item in issues if item["issue"] == "label_without_image"),
        "unlabel_images": len(unlabels),
        "num_classes": len(classes),
        "total_instances": sum(class_instances.values()),
        "bad_labels": sum(1 for item in issues if item["issue"] == "bad_xml"),
        "empty_labels": sum(1 for item in issues if item["issue"] == "empty_label"),
    }
    (manifest_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(manifest_dir / "dataset_summary.csv", [summary], list(summary.keys()))
    write_csv(manifest_dir / "label_quality_issues.csv", issues, ["label", "issue", "detail"])

    classes_with_val = sum(1 for cls in classes if dist_files["val"][cls])
    split_summary = {
        **split_meta,
        "total_samples": len(samples),
        "train_samples": train_n,
        "val_samples": val_n,
        "val_fraction": round(val_n / len(samples), 4) if samples else 0,
        "num_classes": len(classes),
        "classes_with_val_files": classes_with_val,
        "classes_without_val_files": len(classes) - classes_with_val,
    }
    (manifest_dir / "split_summary.json").write_text(json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(manifest_dir / "split_summary.csv", [split_summary], list(split_summary.keys()))


def write_manifests(output_dir: Path, samples: list[Sample], unlabels: list[Path], issues: list[dict[str, str]], config: dict[str, Any]) -> None:
    manifest_dir = output_dir / "manifests"
    classes, class_instances, class_files = class_stats(samples)
    write_class_manifests(manifest_dir, classes, class_instances, class_files)
    write_csv(
        manifest_dir / "samples.csv",
        sample_manifest_rows(samples),
        ["sample_id", "image", "label", "num_instances", "classes", "image_hash", "source_batch"],
    )
    split_rows, split_meta, dist, dist_files = split_manifest_data(samples, config.get("split") or {})
    write_split_manifests(manifest_dir, split_rows, classes, class_instances, class_files, dist, dist_files)
    write_summary_manifests(manifest_dir, samples, unlabels, issues, classes, class_instances, split_rows, split_meta, dist_files)


def clean_output(output_dir: Path) -> None:
    for name in GENERATED_DIRS:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    conflict_dir = output_dir / "double_check" / "repeated_image_label_conflict"
    if conflict_dir.exists():
        shutil.rmtree(conflict_dir)


def validate_sources(output_dir: Path, batches: list[Batch]) -> None:
    generated = [(output_dir / name).resolve() for name in GENERATED_DIRS]
    for batch in batches:
        source = batch.path.resolve()
        for path in generated:
            if source == path or path in source.parents:
                raise SystemExit(f"Refusing to use generated output as raw source: {source}")


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = resolve_path(args.output_dir or config.get("output_dir", "datasets"))
    exclude_dirs = set(config.get("exclude_dirs") or [])
    batches = discover_batches(config, output_dir, args.source)
    if not args.dry_run:
        validate_sources(output_dir, batches)

    images, labels = scan_batches(batches, exclude_dirs)
    samples, unlabels, issues, stats = clean_and_pair(labels, images, config)
    samples, conflicts, duplicate_issues = resolve_duplicates(samples, duplicate_rules(config))
    issues.extend(duplicate_issues)

    write_dataset = not args.dry_run
    if write_dataset:
        clean_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_outputs(output_dir, samples, unlabels, conflicts, issues, config, args.copy)

    summary = {
        "mode": "dry-run" if args.dry_run else "apply",
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir),
        "batches": [{"name": batch.name, "path": str(batch.path)} for batch in batches],
        "raw_images": len(images),
        "raw_labels": len(labels),
        "kept_samples": len(samples),
        "unlabel_images": len(unlabels),
        "repeated_image_label_conflict_groups": len(conflicts),
        "quality_issues": len(issues),
        "num_classes": len({box.cls for sample in samples for box in sample.boxes}),
        "total_instances": sum(len(sample.boxes) for sample in samples),
        "rule_stats": dict(sorted(stats.items())),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("\nDry run only. Re-run without --dry-run to rebuild outputs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
