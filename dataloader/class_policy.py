from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from dataloader.entities import Box, BuildIssue, PolicyResult, PolicyRule, Sample
from dataloader.source_scan import issue


VALID_ACTIONS = {"keep", "remap", "drop", "hold"}


def load_class_policy(path: Path) -> dict[str, PolicyRule]:
    rules: dict[str, PolicyRule] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_class = (row.get("raw_class") or "").strip()
            action = (row.get("action") or "").strip().lower()
            final_class = (row.get("final_class") or "").strip()
            if not raw_class:
                continue
            rules[raw_class] = PolicyRule(raw_class=raw_class, action=action, final_class=final_class)
    return rules


def remap_box(box: Box, rule: PolicyRule) -> Box | None:
    if rule.action == "drop":
        return None
    final_class = rule.final_class or box.cls
    if rule.action == "keep" and rule.final_class:
        final_class = rule.final_class
    if rule.action == "remap":
        final_class = rule.final_class
    if rule.action == "hold":
        final_class = rule.final_class or box.cls
    return Box(final_class, box.xmin, box.ymin, box.xmax, box.ymax)


def clone_with_boxes(sample: Sample, boxes: list[Box], note: str) -> Sample:
    cloned = Sample(
        sample_id=sample.sample_id,
        output_stem=sample.output_stem,
        image_path=sample.image_path,
        label_path=sample.label_path,
        image_hash=sample.image_hash,
        boxes=boxes,
        width=sample.width,
        height=sample.height,
        batch_name=sample.batch_name,
        notes=[*sample.notes, note],
    )
    return cloned


def apply_class_policy(samples: list[Sample], policy_path: Path) -> PolicyResult:
    rules = load_class_policy(policy_path)
    train: list[Sample] = []
    hold_samples: list[Sample] = []
    unknown_samples: list[Sample] = []
    dropped_samples: list[Sample] = []
    issues: list[BuildIssue] = []
    stats = Counter()

    for sample in samples:
        train_boxes: list[Box] = []
        hold_boxes: list[Box] = []
        unknown_classes: set[str] = set()
        dropped_count = 0
        for box in sample.boxes:
            rule = rules.get(box.cls)
            if rule is None or rule.action not in VALID_ACTIONS:
                unknown_classes.add(box.cls)
                continue
            mapped = remap_box(box, rule)
            stats[f"policy_{rule.action}_{box.cls}"] += 1
            if rule.action == "hold":
                if mapped is not None:
                    hold_boxes.append(mapped)
                continue
            if rule.action == "drop":
                dropped_count += 1
                continue
            if mapped is not None:
                train_boxes.append(mapped)

        if unknown_classes:
            unknown_samples.append(clone_with_boxes(sample, sample.boxes, "policy_unknown_class"))
            issues.append(issue(sample.label_path, "policy_unknown_class", "|".join(sorted(unknown_classes))))
            stats["policy_unknown_samples"] += 1
            continue
        if hold_boxes:
            hold_samples.append(clone_with_boxes(sample, hold_boxes, "policy_hold"))
            issues.append(issue(sample.label_path, "policy_hold", "|".join(sorted({box.cls for box in hold_boxes}))))
            stats["policy_hold_samples"] += 1
        if train_boxes:
            train.append(clone_with_boxes(sample, train_boxes, "policy_train"))
            continue
        if hold_boxes:
            issues.append(issue(sample.label_path, "policy_hold_only", str(sample.image_path)))
            stats["policy_hold_only_samples"] += 1
            continue
        dropped_samples.append(clone_with_boxes(sample, sample.boxes, "policy_empty_after_drop_or_hold"))
        issues.append(issue(sample.label_path, "policy_empty_after_drop", str(sample.image_path)))
        stats["policy_empty_after_drop"] += 1

    return PolicyResult(samples=train, hold_samples=hold_samples, unknown_samples=unknown_samples, dropped_samples=dropped_samples, issues=issues, stats=stats)
