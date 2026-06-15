from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

from dataloader.config_loader import resolve_path
from dataloader.entities import Sample
from utils.voc_xml import write_voc_xml


CLEAN_DIRS = {"images", "labels", "doublecheck", "manifests", "unlabel_images", "unlabel_std"}


def clean_output(output_dir: Path) -> None:
    for name in CLEAN_DIRS:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


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


def write_sample(sample: Sample, image_dir: Path, label_dir: Path, copy: bool) -> None:
    image_name = f"{sample.sample_id}{sample.image_path.suffix.lower()}"
    copy_or_link(sample.image_path, image_dir / image_name, copy)
    write_voc_xml(label_dir / f"{sample.sample_id}.xml", image_name, sample.width, sample.height, sample.boxes)


def draw_visualization(sample: Sample, output_path: Path, image_path: Path, title: str) -> Path | None:
    if not sample.boxes:
        return None
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, min(image.width - 1, 1000), 36], fill=(20, 20, 20))
    draw.text((8, 9), title, fill=(255, 255, 255), font=font)
    colors = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15)]
    for index, box in enumerate(sample.boxes, start=1):
        color = colors[index % len(colors)]
        x = max(0, int(box.xmin))
        y = max(0, int(box.ymin))
        xmax = min(image.width - 1, int(box.xmax))
        ymax = min(image.height - 1, int(box.ymax))
        draw.rectangle([x, y, xmax, ymax], outline=color, width=3)
        draw.text((x, max(38, y - 20)), f"{index}:{box.cls}", fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return output_path


def write_doublecheck_group(output_dir: Path, reason: str, samples: list[Sample], copy: bool, group_id: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = output_dir / "doublecheck" / reason
    for index, sample in enumerate(samples, start=1):
        prefix = f"{group_id}__candidate_{index:02d}" if group_id else f"item_{index:05d}"
        stem = f"{prefix}__{sample.label_path.stem}"
        image_name = f"{stem}{sample.image_path.suffix.lower()}"
        image_dst = root / "images" / image_name
        copy_or_link(sample.image_path, image_dst, copy)
        write_voc_xml(root / "labels" / f"{stem}.xml", image_name, sample.width, sample.height, sample.boxes)
        visualization_path = draw_visualization(sample, root / "visualizations" / f"{stem}.jpg", image_dst, f"{reason} boxes={len(sample.boxes)} batch={sample.batch_name}")
        rows.append(
            {
                "reason": reason,
                "group_id": group_id,
                "sample_id": sample.sample_id,
                "source_batch": sample.batch_name,
                "source_image_path": str(sample.image_path),
                "source_label_path": str(sample.label_path),
                "classes": "|".join(sample.classes),
                "num_boxes": len(sample.boxes),
                "visualization_path": str(visualization_path or ""),
                "visualization_status": "skipped_empty_label" if visualization_path is None else "written",
            }
        )
    return rows


def write_conflicts(output_dir: Path, conflicts: list[list[Sample]], copy: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(conflicts, start=1):
        rows.extend(write_doublecheck_group(output_dir, "repeated_image_label_conflict", group, copy, f"group_{group_index:04d}"))
    return rows


def unlabel_image_name(image: Path, seen: Counter) -> str:
    name = f"{image.stem}{image.suffix.lower()}"
    seen[name.lower()] += 1
    if seen[name.lower()] == 1:
        return name
    digest = hashlib.sha1(str(image.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{image.stem}__unlabel_{digest}{image.suffix.lower()}"


def unlabel_output_dir_name(image: Path, config: dict) -> str:
    image_text = str(image.resolve()).replace("\\", "/")
    for raw in config.get("batches") or []:
        rules = raw.get("rules") or {}
        output_dir = str(rules.get("unlabel_output_dir") or "")
        if not output_dir:
            continue
        batch_path = resolve_path(raw["path"])
        if image.resolve() == batch_path or batch_path in image.resolve().parents:
            return output_dir
        path_contains = str(rules.get("unlabel_path_contains") or "").replace("\\", "/")
        if path_contains and path_contains in image_text:
            return output_dir
    return "unlabel_images"


def write_unlabels(output_dir: Path, unlabels: list[Path], config: dict, copy: bool) -> None:
    seen = Counter()
    for image in unlabels:
        unlabel_dir = unlabel_output_dir_name(image, config)
        copy_or_link(image, output_dir / unlabel_dir / unlabel_image_name(image, seen), copy)


def write_dataset_outputs(
    output_dir: Path,
    samples: list[Sample],
    unlabels: list[Path],
    conflicts: list[list[Sample]],
    hold_samples: list[Sample],
    unknown_samples: list[Sample],
    dropped_samples: list[Sample],
    config: dict,
    copy: bool,
) -> list[dict[str, object]]:
    for sample in samples:
        write_sample(sample, output_dir / "images", output_dir / "labels", copy)
    write_unlabels(output_dir, unlabels, config, copy)
    doublecheck_rows: list[dict[str, object]] = []
    doublecheck_rows.extend(write_conflicts(output_dir, conflicts, copy))
    doublecheck_rows.extend(write_doublecheck_group(output_dir, "hold_class", hold_samples, copy))
    doublecheck_rows.extend(write_doublecheck_group(output_dir, "unknown_class", unknown_samples, copy))
    doublecheck_rows.extend(write_doublecheck_group(output_dir, "dropped_empty", dropped_samples, copy))
    return doublecheck_rows
