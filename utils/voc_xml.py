from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def int_text(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(round(float(value.strip())))
    except ValueError:
        return None


def write_voc_xml(path: Path, image_name: str, width: int, height: int, boxes: Iterable[object]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for box in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(box.cls)
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(box.xmin)
        ET.SubElement(bb, "ymin").text = str(box.ymin)
        ET.SubElement(bb, "xmax").text = str(box.xmax)
        ET.SubElement(bb, "ymax").text = str(box.ymax)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8")
