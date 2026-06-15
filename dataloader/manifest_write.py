from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataloader.dataset_write import unlabel_output_dir_name
from dataloader.entities import BuildIssue, Sample


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_and_csv(path_base: Path, row: dict[str, Any]) -> None:
    path_base.with_suffix(".json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(path_base.with_suffix(".csv"), [row], list(row.keys()))


def class_stats(samples: list[Sample]) -> tuple[list[str], Counter, Counter]:
    class_instances = Counter(box.cls for sample in samples for box in sample.boxes)
    class_files = Counter()
    for sample in samples:
        class_files.update(sample.classes)
    return sorted(class_instances), class_instances, class_files


def write_class_manifests(manifest_dir: Path, classes: list[str], class_instances: Counter, class_files: Counter) -> None:
    rows = [
        {"class_id": index, "class_name": cls, "instances": class_instances[cls], "label_files": class_files[cls], "image_files": class_files[cls]}
        for index, cls in enumerate(classes)
    ]
    write_csv(manifest_dir / "class_mapping.csv", rows, ["class_id", "class_name", "instances", "label_files", "image_files"])
    (manifest_dir / "classes.txt").write_text("".join(f"{cls}\n" for cls in classes), encoding="utf-8")
    count_rows = sorted(rows, key=lambda row: (-int(row["instances"]), str(row["class_name"])))
    write_csv(manifest_dir / "class_counts.csv", [{k: row[k] for k in ("class_name", "instances", "label_files", "image_files")} for row in count_rows], ["class_name", "instances", "label_files", "image_files"])


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
    key = "_".join(part for part in parts if part)
    return key or sample.image_hash


def split_samples(samples: list[Sample], split_cfg: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
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
        if not (set(counts) & train_only_classes) and all(class_files[cls] - count > 0 for cls, count in counts.items())
    ]
    val_groups: set[str] = set()
    covered = Counter()
    total = 0

    def add_val_group(name: str) -> None:
        nonlocal total
        val_groups.add(name)
        covered.update(group_classes[name])
        total += len(groups[name])
        eligible.remove(name)

    while eligible and (set(covered) & val_coverable_classes) != val_coverable_classes:
        useful = [name for name in eligible if set(group_classes[name]) & (val_coverable_classes - set(covered))]
        if not useful:
            break
        add_val_group(min(useful, key=lambda name: (-len(set(group_classes[name]) & (val_coverable_classes - set(covered))), abs(target - (total + len(groups[name]))), len(groups[name]), name)))
    while eligible and total < target:
        add_val_group(min(eligible, key=lambda name: (abs(target - (total + len(groups[name]))), len(groups[name]), name)))
    return {sample.sample_id for group in val_groups for sample in groups[group]}, {
        "strategy": "group_stratified",
        "total_groups": len(groups),
        "train_groups": len(groups) - len(val_groups),
        "val_groups": len(val_groups),
        "train_only_class_max_files": train_only_max_files,
    }


def write_split_manifests(manifest_dir: Path, samples: list[Sample], classes: list[str], class_instances: Counter, class_files: Counter, split_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, defaultdict[str, set[str]]]]:
    val_ids, split_meta = split_samples(samples, split_cfg)
    rows: list[dict[str, Any]] = []
    dist = {"train": Counter(), "val": Counter()}
    dist_files = {"train": defaultdict(set), "val": defaultdict(set)}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        split = "val" if sample.sample_id in val_ids else "train"
        label_name = f"{sample.sample_id}.xml"
        for box in sample.boxes:
            dist[split][box.cls] += 1
        for cls in sample.classes:
            dist_files[split][cls].add(label_name)
        rows.append({"split": split, "sample_id": sample.sample_id, "image": f"{sample.sample_id}{sample.image_path.suffix.lower()}", "label": label_name, "group": group_key(sample, split_cfg), "classes": "|".join(sample.classes)})
    write_csv(manifest_dir / "train_split.csv", [{"relative_label_path": row["label"]} for row in rows if row["split"] == "train"], ["relative_label_path"])
    write_csv(manifest_dir / "val_split.csv", [{"relative_label_path": row["label"]} for row in rows if row["split"] == "val"], ["relative_label_path"])
    write_csv(manifest_dir / "split_samples.csv", rows, ["split", "sample_id", "image", "label", "group", "classes"])
    write_csv(
        manifest_dir / "split_class_distribution.csv",
        [{"class_name": cls, "total_instances": class_instances[cls], "train_instances": dist["train"][cls], "val_instances": dist["val"][cls], "total_files": class_files[cls], "train_files": len(dist_files["train"][cls]), "val_files": len(dist_files["val"][cls])} for cls in classes],
        ["class_name", "total_instances", "train_instances", "val_instances", "total_files", "train_files", "val_files"],
    )
    return rows, split_meta, dist_files


def write_manifests(output_dir: Path, samples: list[Sample], unlabels: list[Path], issues: list[BuildIssue], config: dict[str, Any], doublecheck_rows: list[dict[str, object]], pipeline_stats: dict[str, Any]) -> None:
    manifest_dir = output_dir / "manifests"
    classes, class_instances, class_files = class_stats(samples)
    write_class_manifests(manifest_dir, classes, class_instances, class_files)
    write_csv(
        manifest_dir / "samples.csv",
        [{"sample_id": sample.sample_id, "image": f"{sample.sample_id}{sample.image_path.suffix.lower()}", "label": f"{sample.sample_id}.xml", "num_instances": len(sample.boxes), "classes": "|".join(sample.classes), "image_hash": sample.image_hash, "source_batch": sample.batch_name, "source_image_path": str(sample.image_path), "source_label_path": str(sample.label_path), "notes": "|".join(sample.notes)} for sample in sorted(samples, key=lambda item: item.sample_id)],
        ["sample_id", "image", "label", "num_instances", "classes", "image_hash", "source_batch", "source_image_path", "source_label_path", "notes"],
    )
    split_rows, split_meta, dist_files = write_split_manifests(manifest_dir, samples, classes, class_instances, class_files, config.get("split") or {})
    issue_rows = [item.as_row() for item in issues]
    write_csv(manifest_dir / "label_quality_issues.csv", issue_rows, ["label", "issue", "detail"])
    if doublecheck_rows:
        write_csv(manifest_dir / "doublecheck_index.csv", doublecheck_rows, list(doublecheck_rows[0].keys()))
    unlabel_counts = Counter(unlabel_output_dir_name(image, config) for image in unlabels)
    train_n = sum(1 for row in split_rows if row["split"] == "train")
    summary = {
        **pipeline_stats,
        "images": len(samples),
        "labels": len(samples),
        "paired_samples": len(samples),
        "unlabel_images": len(unlabels),
        "unlabel_images_dir_count": unlabel_counts["unlabel_images"],
        "unlabel_std_dir_count": unlabel_counts["unlabel_std"],
        "num_classes": len(classes),
        "total_instances": sum(class_instances.values()),
        "quality_issues": len(issues),
        "doublecheck_items": len(doublecheck_rows),
    }
    write_json_and_csv(manifest_dir / "dataset_summary", summary)
    split_summary = {
        **split_meta,
        "total_samples": len(samples),
        "train_samples": train_n,
        "val_samples": len(split_rows) - train_n,
        "val_fraction": round((len(split_rows) - train_n) / len(samples), 4) if samples else 0,
        "num_classes": len(classes),
        "classes_with_val_files": sum(1 for cls in classes if dist_files["val"][cls]),
        "classes_without_val_files": sum(1 for cls in classes if not dist_files["val"][cls]),
    }
    write_json_and_csv(manifest_dir / "split_summary", split_summary)
