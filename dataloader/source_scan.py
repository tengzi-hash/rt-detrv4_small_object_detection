from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from dataloader.entities import BatchConfig, BuildIssue, RawPair, ScanResult
from utils.image_meta import image_read_error


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_SUFFIXES = {".xml", ".json", ".txt"}


def issue(label: Path | str, issue_type: str, detail: str) -> BuildIssue:
    return BuildIssue(str(label), issue_type, detail)


def excluded(path: Path, exclude_dirs: set[str]) -> bool:
    return any(part in exclude_dirs for part in path.parts)


def batch_label_suffixes(batch: BatchConfig) -> set[str]:
    values = batch.rules.get("label_suffixes")
    if not values:
        return LABEL_SUFFIXES
    return {str(value).lower() if str(value).startswith(".") else f".{str(value).lower()}" for value in values}


def scan_files(batches: list[BatchConfig], exclude_dirs: set[str]) -> tuple[list[Path], list[tuple[Path, BatchConfig]]]:
    images: list[Path] = []
    labels: list[tuple[Path, BatchConfig]] = []
    for batch in batches:
        label_suffixes = batch_label_suffixes(batch)
        for path in batch.path.rglob("*"):
            if not path.is_file() or excluded(path, exclude_dirs):
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                images.append(path)
            elif suffix in label_suffixes:
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
        common_depth = 0
        for left, right in zip(parent.parts, anchor_parent.parts):
            if left != right:
                break
            common_depth += 1
        return (same_parent, -common_depth, str(path))

    return sorted(paths, key=key)[0]


def read_label_filename(label_path: Path) -> str | None:
    suffix = label_path.suffix.lower()
    if suffix == ".xml":
        root = ET.parse(label_path).getroot()
        return root.findtext("filename")
    if suffix == ".json":
        data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            return str(data.get("filename") or data.get("image") or "") or None
    if suffix == ".txt":
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" in line:
                return line.split(";", 1)[0].strip()
            return None
    return None


def match_image(label_path: Path, filename: str | None, by_stem: dict[str, list[Path]], by_name: dict[str, list[Path]]) -> Path | None:
    candidates: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        direct = label_path.with_suffix(suffix)
        if direct.is_file():
            return direct
    candidates.extend(by_stem.get(label_path.stem.lower(), []))
    if candidates:
        return nearest(sorted(set(candidates)), label_path)
    if filename:
        raw_name = Path(filename.strip()).name
        direct = label_path.parent / raw_name
        if direct.is_file():
            return direct
        candidates.extend(by_name.get(raw_name.lower(), []))
        candidates.extend(by_stem.get(Path(raw_name).stem.lower(), []))
    candidates = sorted(set(candidates))
    return nearest(candidates, label_path) if candidates else None


def scan_sources(batches: list[BatchConfig], exclude_dirs: set[str]) -> ScanResult:
    images, labels = scan_files(batches, exclude_dirs)
    by_stem, by_name = image_indexes(images)
    used_images: set[Path] = set()
    pairs: list[RawPair] = []
    issues: list[BuildIssue] = []

    for label_path, batch in labels:
        try:
            label_filename = read_label_filename(label_path)
        except Exception as exc:
            issues.append(issue(label_path, "bad_label", str(exc)))
            continue
        image_path = match_image(label_path, label_filename, by_stem, by_name)
        if image_path is None:
            issues.append(issue(label_path, "label_without_image", "discarded"))
            continue
        used_images.add(image_path)
        if error := image_read_error(image_path):
            issues.append(issue(label_path, "image_unreadable", f"{image_path}; {error}"))
            continue
        pairs.append(RawPair(label_path=label_path, image_path=image_path, batch=batch))

    unlabel_images: list[Path] = []
    for image in images:
        if image in used_images:
            continue
        if error := image_read_error(image):
            issues.append(issue(image, "unlabel_image_unreadable", error))
            continue
        unlabel_images.append(image)
    return ScanResult(pairs=pairs, unlabel_images=sorted(set(unlabel_images)), issues=issues, raw_images=len(images), raw_labels=len(labels))
