from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from dataloader.annotation_parse import parse_annotation
from dataloader.entities import BatchConfig, Box, BuildIssue, ParseResult, RawPair, Sample
from dataloader.source_scan import issue
from utils.bbox_ops import coverage_ratio, iou


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
        merged.append(Box("Clamp", min(item.xmin for item in group_boxes), min(item.ymin for item in group_boxes), max(item.xmax for item in group_boxes), max(item.ymax for item in group_boxes)))
        used.update(group[1:])
        removed += len(group) - 1
    return merged, removed


def force_merge_class_boxes(boxes: list[Box], class_name: str) -> tuple[list[Box], int]:
    class_indices = [index for index, box in enumerate(boxes) if box.cls == class_name]
    if len(class_indices) < 2:
        return boxes, 0
    target_boxes = [boxes[index] for index in class_indices]
    merged_box = Box(class_name, min(box.xmin for box in target_boxes), min(box.ymin for box in target_boxes), max(box.xmax for box in target_boxes), max(box.ymax for box in target_boxes))
    first_index = min(class_indices)
    skip_indices = set(class_indices) - {first_index}
    rebuilt = [merged_box if index == first_index else box for index, box in enumerate(boxes) if index not in skip_indices]
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


def override_matches_label(override: dict[str, Any], label_path: Path) -> bool:
    path_text = str(label_path).replace("\\", "/")
    return any(
        (
            override.get("label_stem") and str(override["label_stem"]) == label_path.stem,
            override.get("label_name") and str(override["label_name"]) == label_path.name,
            override.get("label_path_endswith") and path_text.endswith(str(override["label_path_endswith"]).replace("\\", "/")),
        )
    )


def matching_overrides(batch: BatchConfig, label_path: Path) -> list[dict[str, Any]]:
    return [override for override in batch.rules.get("sample_overrides") or [] if isinstance(override, dict) and override_matches_label(override, label_path)]


def apply_sample_overrides(boxes: list[Box], label_path: Path, batch: BatchConfig, stats: Counter) -> list[Box]:
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
        if override.get("box_fixes"):
            boxes, fixed_count = apply_box_fixes(boxes, list(override.get("box_fixes") or []))
            stats["sample_override_box_fixes"] += fixed_count
        drop_classes = set(override.get("drop_classes") or [])
        if drop_classes:
            before_counts = Counter(box.cls for box in boxes)
            boxes = [box for box in boxes if box.cls not in drop_classes]
            for class_name in sorted(drop_classes):
                stats[f"sample_override_dropped_{class_name}"] += before_counts[class_name]
        if override.get("add_boxes"):
            boxes, added_count = add_configured_boxes(boxes, list(override.get("add_boxes") or []))
            stats["sample_override_added_boxes"] += added_count
        for class_name in override.get("force_merge_classes") or []:
            boxes, removed = force_merge_class_boxes(boxes, str(class_name))
            stats[f"sample_override_force_merged_{class_name}"] += removed
    return boxes


def clean_boxes(raw_boxes: list[Box], batch: BatchConfig, label_path: Path, stats: Counter) -> list[Box]:
    raw_clamp_2_boxes = [box for box in raw_boxes if box.cls == "Clamp_2"]
    sample_remap = {raw_cls: new_cls for override in matching_overrides(batch, label_path) for raw_cls, new_cls in (override.get("remap_classes") or {}).items()}
    boxes: list[Box] = []
    for raw in raw_boxes:
        if batch.rules.get("drop_clamp_covered_by_clamp_2", False) and raw.cls == "Clamp":
            threshold = float(batch.rules.get("clamp_2_cover_threshold", 0.90))
            if any(coverage_ratio(raw, clamp_2_box) >= threshold for clamp_2_box in raw_clamp_2_boxes):
                stats["dropped_clamp_covered_by_clamp_2"] += 1
                continue
        if raw.cls in batch.drop_classes:
            cls = sample_remap.get(raw.cls)
            if not cls:
                stats[f"source_dropped_{raw.cls}"] += 1
                continue
            stats[f"sample_override_remapped_{raw.cls}_to_{cls}"] += 1
        else:
            cls = batch.class_remap.get(raw.cls, raw.cls)
        boxes.append(Box(cls, raw.xmin, raw.ymin, raw.xmax, raw.ymax))
    if batch.rules.get("merge_overlapping_clamp", False):
        boxes, removed = merge_overlapping_clamps(boxes)
        stats["merged_overlapping_clamp_boxes"] += removed
    boxes = apply_sample_overrides(boxes, label_path, batch, stats)
    boxes, covered_drops = drop_covered_boxes(boxes, list(batch.rules.get("drop_covered_by") or []))
    for key, count in covered_drops.items():
        stats[f"source_dropped_{key}"] += count
    return boxes


def output_stem_for_label(label_path: Path, batch: BatchConfig) -> str:
    stem = label_path.stem
    label_text = str(label_path).replace("\\", "/")
    for item in batch.rules.get("output_stem_suffixes") or []:
        if not isinstance(item, dict):
            continue
        path_contains = str(item.get("path_contains") or "").replace("\\", "/")
        path_endswith = str(item.get("path_endswith") or item.get("label_path_endswith") or "").replace("\\", "/")
        suffix = str(item.get("suffix") or "")
        if suffix and ((path_contains and path_contains in label_text) or (path_endswith and label_text.endswith(path_endswith))):
            return f"{stem}{suffix}"
    return stem


def parse_and_clean_pairs(pairs: list[RawPair], unlabel_images: list[Path], incoming_issues: list[BuildIssue]) -> ParseResult:
    samples: list[Sample] = []
    issues = list(incoming_issues)
    stats = Counter()
    extra_unlabels = list(unlabel_images)
    for pair in pairs:
        try:
            raw_boxes, width, height, label_issues = parse_annotation(pair.label_path, pair.image_path)
        except Exception as exc:
            issues.append(issue(pair.label_path, "parse_failed", str(exc)))
            continue
        for issue_type, detail in label_issues:
            issues.append(issue(pair.label_path, issue_type, detail))
        boxes = clean_boxes(raw_boxes, pair.batch, pair.label_path, stats)
        if not boxes:
            extra_unlabels.append(pair.image_path)
            issues.append(issue(pair.label_path, "empty_label_after_source_clean", str(pair.image_path)))
            continue
        output_stem = output_stem_for_label(pair.label_path, pair.batch)
        samples.append(Sample(output_stem, output_stem, pair.image_path, pair.label_path, "", boxes, width, height, pair.batch.name))
    return ParseResult(samples=samples, unlabel_images=sorted(set(extra_unlabels)), issues=issues, stats=stats)
