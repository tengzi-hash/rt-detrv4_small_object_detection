from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataloader.entities import Box, BuildIssue, DedupResult, Sample
from dataloader.source_scan import issue
from utils.bbox_ops import box_area, iou
from utils.file_hash import sha1_file


def assign_image_hashes(samples: list[Sample]) -> None:
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
        cache: dict[Path, str] = {}
        for sample in group:
            sample.image_hash = cache.setdefault(sample.image_path, sha1_file(sample.image_path))


def class_multiset(sample: Sample) -> Counter:
    return Counter(box.cls for box in sample.boxes)


def counter_contains(left: Counter, right: Counter) -> bool:
    return all(left[key] >= value for key, value in right.items())


def mean_box_area(sample: Sample) -> float:
    return sum(box_area(box) for box in sample.boxes) / max(len(sample.boxes), 1)


def path_suffix_matches(path: Path, suffix: str) -> bool:
    return str(path).replace("\\", "/").endswith(str(suffix).replace("\\", "/"))


def duplicate_preference_key(sample: Sample, rules: dict[str, Any]) -> tuple[int, str, str]:
    path_text = str(sample.label_path).replace("\\", "/")
    preferred = [str(value).replace("\\", "/") for value in rules.get("duplicate_prefer_label_path_contains") or []]
    rank = 0 if any(value and value in path_text for value in preferred) else 1
    return (rank, sample.batch_name, path_text)


def has_preferred_duplicate_source(group: list[Sample], rules: dict[str, Any]) -> bool:
    return any(duplicate_preference_key(sample, rules)[0] == 0 for sample in group)


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


def boxes_match_by_class(left: Sample, right: Sample, min_iou: float) -> bool:
    if len(left.boxes) != len(right.boxes) or class_multiset(left) != class_multiset(right):
        return False
    for class_name in class_multiset(left):
        left_boxes = [box for box in left.boxes if box.cls == class_name]
        right_boxes = [box for box in right.boxes if box.cls == class_name]
        pairs = sorted(((iou(a, b), li, ri) for li, a in enumerate(left_boxes) for ri, b in enumerate(right_boxes)), reverse=True)
        matched_left: set[int] = set()
        matched_right: set[int] = set()
        for score, li, ri in pairs:
            if score < min_iou:
                break
            if li in matched_left or ri in matched_right:
                continue
            matched_left.add(li)
            matched_right.add(ri)
        if len(matched_left) != len(left_boxes):
            return False
    return True


def has_label_conflict(a: Box, b: Box, min_iou: float) -> bool:
    return a.cls != b.cls and iou(a, b) >= min_iou


def merge_boxes(samples: list[Sample], min_iou: float) -> list[Box] | None:
    merged: list[Box] = []
    for sample in sorted(samples, key=lambda item: (-len(item.boxes), str(item.label_path))):
        for box in sample.boxes:
            if any(has_label_conflict(box, existing, min_iou) for existing in merged):
                return None
            if any(box.cls == existing.cls and iou(box, existing) >= min_iou for existing in merged):
                continue
            merged.append(box)
    return merged


def merge_boxes_union(samples: list[Sample], rules: dict[str, Any]) -> tuple[Sample, int]:
    min_iou = float(rules.get("duplicate_merge_iou", rules.get("duplicate_min_iou", 0.50)))
    ordered = sorted(samples, key=lambda sample: duplicate_preference_key(sample, rules))
    target = ordered[0]
    merged: list[Box] = []
    removed_duplicates = 0
    for sample in ordered:
        for box in sample.boxes:
            if any(box.cls == existing.cls and iou(box, existing) >= min_iou for existing in merged):
                removed_duplicates += 1
                continue
            merged.append(box)
    target.boxes = sorted(merged, key=lambda box: (box.cls, box.xmin, box.ymin, box.xmax, box.ymax))
    target.notes.append("deduplicate_union_merged_boxes_from_same_image")
    return target, removed_duplicates


def choose_or_merge_duplicate(group: list[Sample], rules: dict[str, Any]) -> tuple[Sample | None, str]:
    signatures = {sample.label_signature for sample in group}
    if len(signatures) == 1:
        return sorted(group, key=lambda sample: duplicate_preference_key(sample, rules))[0], "repeated_image_same_label_kept_one"

    if rules.get("duplicate_merge_conflicts") and has_preferred_duplicate_source(group, rules):
        selected, _ = merge_boxes_union(group, rules)
        return selected, "repeated_image_preferred_labels_union_merged"

    selected = configured_duplicate_keep(group, rules)
    if selected is not None:
        return selected, "repeated_image_configured_keep"

    if rules.get("duplicate_merge_conflicts"):
        selected, _ = merge_boxes_union(group, rules)
        return selected, "repeated_image_labels_union_merged"

    counts = [(sample, class_multiset(sample)) for sample in group]
    superset = [
        sample
        for sample, count in counts
        if all(counter_contains(count, other_count) for other, other_count in counts if other is not sample)
        and any(sum(count.values()) > sum(other_count.values()) for other, other_count in counts if other is not sample)
    ]
    if len(superset) == 1:
        return superset[0], "repeated_image_class_superset_kept"

    min_iou = float(rules.get("duplicate_min_iou", 0.50))
    reference = group[0]
    if all(boxes_match_by_class(reference, other, min_iou) for other in group[1:]):
        return min(group, key=lambda sample: (mean_box_area(sample), str(sample.label_path))), "repeated_image_same_classes_boxes_auto_resolved"

    merged_boxes = merge_boxes(group, min_iou)
    if merged_boxes:
        target = max(group, key=lambda sample: (len(sample.boxes), len(sample.classes), -len(str(sample.label_path))))
        target.boxes = sorted(merged_boxes, key=lambda box: (box.cls, box.xmin, box.ymin, box.xmax, box.ymax))
        target.notes.append("deduplicate_merged_boxes_from_same_image")
        return target, "repeated_image_labels_merged"
    return None, ""


def sanitize_prefix(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", value).strip("_")


def assign_sample_ids(samples: list[Sample]) -> None:
    by_stem: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_stem[sample.output_stem].append(sample)
    used: set[str] = set()
    for stem, group in sorted(by_stem.items()):
        for index, sample in enumerate(sorted(group, key=lambda item: (item.batch_name, str(item.label_path))), start=1):
            if index == 1:
                candidate = stem
            else:
                candidate = f"{stem}__dup{index - 1:03d}"
            if candidate in used:
                prefix = sanitize_prefix(sample.batch_name)
                candidate = f"{prefix}__{candidate}" if prefix else candidate
            while candidate in used:
                candidate = f"{candidate}__{len(used):06d}"
            sample.sample_id = candidate
            used.add(candidate)


def resolve_duplicates(samples: list[Sample], rules: dict[str, Any]) -> DedupResult:
    stats = Counter()
    issues: list[BuildIssue] = []
    conflicts: list[list[Sample]] = []
    kept: list[Sample] = []
    assign_image_hashes(samples)

    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.image_hash].append(sample)

    for image_hash, group in sorted(by_hash.items()):
        if len(group) == 1 or image_hash.startswith("unique:"):
            kept.extend(group)
            continue
        by_name: dict[str, list[Sample]] = defaultdict(list)
        for sample in group:
            by_name[sample.image_path.name.lower()].append(sample)
        for name_group in by_name.values():
            if len(name_group) == 1:
                kept.extend(name_group)
                continue
            selected, reason = choose_or_merge_duplicate(name_group, rules)
            if selected is None:
                conflicts.append(name_group)
                issues.append(issue(name_group[0].label_path, "repeated_image_label_conflict", f"image_hash={image_hash}; count={len(name_group)}"))
                stats["repeated_image_label_conflict_groups"] += 1
                continue
            kept.append(selected)
            issues.append(issue(selected.label_path, reason, f"image_hash={image_hash}; count={len(name_group)}; kept={selected.label_path.name}"))
            stats[reason] += 1

    assign_sample_ids(kept)
    return DedupResult(samples=kept, conflicts=conflicts, issues=issues, stats=stats)
