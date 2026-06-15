from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from dataloader.entities import Box
from utils import label_convert
from utils.image_meta import image_size
from utils.voc_xml import int_text


def converted_label_root(label_path: Path, image_path: Path) -> ET.Element:
    suffix = label_path.suffix.lower()
    if suffix == ".xml":
        return ET.parse(label_path).getroot()
    if suffix == ".txt":
        annotations = label_convert.txt_to_annotations(label_path, image_path.parent, "auto", [])
    elif suffix == ".json":
        data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        items = data if isinstance(data, list) else [data]
        annotations = [label_convert.dict_to_annotation(item, label_path, image_path.parent) for item in items if isinstance(item, dict)]
    else:
        raise ValueError(f"Unsupported label format: {label_path.suffix}")
    if not annotations:
        raise ValueError(f"No annotations found in {label_path}")
    matched = [
        annotation
        for annotation in annotations
        if Path(annotation.filename).stem.lower() == image_path.stem.lower()
        or Path(annotation.filename).name.lower() == image_path.name.lower()
    ]
    return label_convert.annotation_to_xml((matched or annotations)[0])


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


def read_raw_objects(root: ET.Element) -> tuple[list[Box], list[tuple[str, str]]]:
    boxes: list[Box] = []
    issues: list[tuple[str, str]] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        if not raw_name:
            issues.append(("object_without_name", ""))
            continue
        box = parse_object_box(obj, raw_name)
        if box is None:
            issues.append(("bad_or_missing_box", raw_name))
            continue
        boxes.append(box)
    return boxes, issues


def annotation_size(root: ET.Element, image_path: Path) -> tuple[int, int]:
    width = int_text(root.findtext("size/width")) or 0
    height = int_text(root.findtext("size/height")) or 0
    if width <= 0 or height <= 0:
        return image_size(image_path)
    return width, height


def parse_annotation(label_path: Path, image_path: Path) -> tuple[list[Box], int, int, list[tuple[str, str]]]:
    root = converted_label_root(label_path, image_path)
    boxes, issues = read_raw_objects(root)
    width, height = annotation_size(root, image_path)
    return boxes, width, height, issues
