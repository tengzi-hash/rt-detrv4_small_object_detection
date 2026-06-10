"""Convert detection labels between TXT, JSON, and Pascal VOC XML.

Supported TXT formats:
- data5 semicolon format:
  image.jpg;xmin;ymin;xmax;ymax;ClassName
- YOLO format:
  class_id x_center y_center width height

Examples:
  python utils/label_convert.py txt2xml data_raw/data5
  python utils/label_convert.py txt2json data_raw/data5 --output-dir tmp/json_labels
  python utils/label_convert.py json2xml tmp/json_labels --image-root data_raw/data5
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class Box:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Annotation:
    filename: str
    folder: str
    image_path: str
    width: int
    height: int
    depth: int
    boxes: list[Box]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TXT/JSON detection labels to JSON/XML.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("txt2xml", "txt2json", "json2xml"):
        sub = subparsers.add_parser(name)
        sub.add_argument("input", help="Input label file or directory.")
        sub.add_argument("--output-dir", default=None, help="Directory for converted labels. Defaults beside input labels.")
        sub.add_argument("--image-root", default=None, help="Directory used to find images. Defaults to input directory.")
        sub.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
        if name.startswith("txt2"):
            sub.add_argument(
                "--txt-format",
                choices=("auto", "semicolon", "yolo"),
                default="auto",
                help="TXT label format. Default auto-detects per line.",
            )
            sub.add_argument(
                "--classes",
                default=None,
                help="Class name file for YOLO txt, one class per line. If omitted, class ids are kept as strings.",
            )
    return parser.parse_args()


def read_class_names(path: str | None) -> list[str]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [line.strip() for line in handle if line.strip()]


def iter_files(path: Path, suffix: str) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected {suffix} file, got: {path}")
        return [path]
    return sorted(p for p in path.rglob(f"*{suffix}") if p.is_file())


def image_size(path: Path | None) -> tuple[int, int, int]:
    if path is None:
        return 0, 0, 3
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = int(image.width), int(image.height)
            depth = len(image.getbands()) if image.getbands() else 3
            return width, height, depth
    except Exception:
        return 0, 0, 3


def find_image(filename_or_stem: str, label_path: Path, image_root: Path | None) -> Path | None:
    raw = Path(filename_or_stem).name
    search_roots = []
    if image_root is not None:
        search_roots.append(image_root)
    search_roots.append(label_path.parent)

    candidates: list[Path] = []
    raw_path = Path(raw)
    if raw_path.suffix:
        for root in search_roots:
            direct = root / raw
            if direct.is_file():
                return direct
        for root in search_roots:
            candidates.extend(root.rglob(raw))
        return sorted(set(candidates))[0] if candidates else None

    for root in search_roots:
        for suffix in IMAGE_SUFFIXES:
            direct = root / f"{raw}{suffix}"
            if direct.is_file():
                return direct
    for root in search_roots:
        for suffix in IMAGE_SUFFIXES:
            candidates.extend(root.rglob(f"{raw}{suffix}"))
    return sorted(set(candidates))[0] if candidates else None


def clean_box(box: Box, width: int, height: int) -> Box | None:
    xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
    if width > 0:
        xmin, xmax = max(0, xmin), min(width, xmax)
    if height > 0:
        ymin, ymax = max(0, ymin), min(height, ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    return Box(box.name, xmin, ymin, xmax, ymax)


def parse_semicolon_line(line: str) -> tuple[str, Box]:
    parts = [part.strip() for part in line.split(";")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 semicolon fields, got {len(parts)}: {line}")
    filename, xmin, ymin, xmax, ymax, cls = parts
    return filename, Box(cls, round_float(xmin), round_float(ymin), round_float(xmax), round_float(ymax))


def parse_yolo_line(line: str, class_names: list[str], width: int, height: int) -> Box:
    if width <= 0 or height <= 0:
        raise ValueError("YOLO txt requires a readable image size.")
    parts = line.replace(",", " ").split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 YOLO fields, got {len(parts)}: {line}")
    class_id = int(float(parts[0]))
    name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
    xc, yc, bw, bh = (float(value) for value in parts[1:])
    xmin = int(round((xc - bw / 2.0) * width))
    ymin = int(round((yc - bh / 2.0) * height))
    xmax = int(round((xc + bw / 2.0) * width))
    ymax = int(round((yc + bh / 2.0) * height))
    return Box(name, xmin, ymin, xmax, ymax)


def round_float(value: str) -> int:
    return int(round(float(value)))


def detect_txt_format(line: str) -> str:
    if ";" in line:
        return "semicolon"
    return "yolo"


def txt_to_annotations(label_path: Path, image_root: Path | None, txt_format: str, class_names: list[str]) -> list[Annotation]:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        image_path = find_image(label_path.stem, label_path, image_root)
        width, height, depth = image_size(image_path)
        filename = image_path.name if image_path else f"{label_path.stem}.jpg"
        folder = image_path.parent.name if image_path else label_path.parent.name
        return [Annotation(filename, folder, str(image_path or ""), width, height, depth, [])]

    effective_format = txt_format if txt_format != "auto" else detect_txt_format(lines[0])
    if effective_format == "semicolon":
        by_filename: dict[str, list[Box]] = {}
        for line in lines:
            filename, box = parse_semicolon_line(line)
            by_filename.setdefault(filename, []).append(box)
        annotations = []
        for filename, boxes in by_filename.items():
            image_path = find_image(filename, label_path, image_root)
            width, height, depth = image_size(image_path)
            cleaned = [clean for box in boxes if (clean := clean_box(box, width, height)) is not None]
            annotations.append(
                Annotation(
                    filename=Path(filename).name,
                    folder=image_path.parent.name if image_path else label_path.parent.name,
                    image_path=str(image_path or ""),
                    width=width,
                    height=height,
                    depth=depth,
                    boxes=cleaned,
                )
            )
        return annotations

    image_path = find_image(label_path.stem, label_path, image_root)
    width, height, depth = image_size(image_path)
    boxes = [parse_yolo_line(line, class_names, width, height) for line in lines]
    cleaned = [clean for box in boxes if (clean := clean_box(box, width, height)) is not None]
    return [
        Annotation(
            filename=image_path.name if image_path else f"{label_path.stem}.jpg",
            folder=image_path.parent.name if image_path else label_path.parent.name,
            image_path=str(image_path or ""),
            width=width,
            height=height,
            depth=depth,
            boxes=cleaned,
        )
    ]


def annotation_to_dict(annotation: Annotation) -> dict:
    return {
        "filename": annotation.filename,
        "folder": annotation.folder,
        "path": annotation.image_path,
        "size": {"width": annotation.width, "height": annotation.height, "depth": annotation.depth},
        "objects": [
            {
                "name": box.name,
                "bndbox": {"xmin": box.xmin, "ymin": box.ymin, "xmax": box.xmax, "ymax": box.ymax},
            }
            for box in annotation.boxes
        ],
    }


def dict_to_annotation(data: dict, json_path: Path, image_root: Path | None) -> Annotation:
    filename = str(data.get("filename") or data.get("image") or json_path.with_suffix(".jpg").name)
    image_path_text = str(data.get("path") or "")
    image_path = Path(image_path_text) if image_path_text else find_image(Path(filename).name, json_path, image_root)
    if image_path and not image_path.is_absolute():
        image_path = (json_path.parent / image_path).resolve()

    size = data.get("size") or {}
    width = int(size.get("width") or data.get("width") or 0)
    height = int(size.get("height") or data.get("height") or 0)
    depth = int(size.get("depth") or data.get("depth") or 3)
    if width <= 0 or height <= 0:
        width, height, depth = image_size(image_path if image_path and image_path.exists() else None)

    raw_objects = data.get("objects") or data.get("annotations") or []
    boxes: list[Box] = []
    for obj in raw_objects:
        name = str(obj.get("name") or obj.get("label") or obj.get("class") or "")
        bndbox = obj.get("bndbox") or obj.get("box") or obj
        if not name:
            continue
        box = Box(
            name=name,
            xmin=round_float(str(bndbox["xmin"])),
            ymin=round_float(str(bndbox["ymin"])),
            xmax=round_float(str(bndbox["xmax"])),
            ymax=round_float(str(bndbox["ymax"])),
        )
        cleaned = clean_box(box, width, height)
        if cleaned is not None:
            boxes.append(cleaned)

    return Annotation(
        filename=Path(filename).name,
        folder=str(data.get("folder") or (image_path.parent.name if image_path else json_path.parent.name)),
        image_path=str(image_path or image_path_text),
        width=width,
        height=height,
        depth=depth,
        boxes=boxes,
    )


def write_json(annotation: Annotation, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(annotation_to_dict(annotation), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def indent_xml(element: ET.Element) -> None:
    try:
        ET.indent(element, space="\t")
    except AttributeError:
        pass


def annotation_to_xml(annotation: Annotation) -> ET.Element:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = annotation.folder
    ET.SubElement(root, "filename").text = annotation.filename
    ET.SubElement(root, "path").text = annotation.image_path
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(annotation.width)
    ET.SubElement(size, "height").text = str(annotation.height)
    ET.SubElement(size, "depth").text = str(annotation.depth)
    ET.SubElement(root, "segmented").text = "0"

    for box in annotation.boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box.name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box.xmin)
        ET.SubElement(bndbox, "ymin").text = str(box.ymin)
        ET.SubElement(bndbox, "xmax").text = str(box.xmax)
        ET.SubElement(bndbox, "ymax").text = str(box.ymax)
    indent_xml(root)
    return root


def write_xml(annotation: Annotation, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(annotation_to_xml(annotation))
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return True


def output_path_for(label_path: Path, input_root: Path, output_dir: Path | None, filename: str, suffix: str) -> Path:
    output_name = f"{Path(filename).stem}{suffix}"
    if output_dir is None:
        return label_path.with_name(output_name)
    if input_root.is_file():
        return output_dir / output_name
    rel_parent = label_path.parent.relative_to(input_root)
    return output_dir / rel_parent / output_name


def txt_to_json(args: argparse.Namespace) -> tuple[int, int]:
    input_path = Path(args.input)
    image_root = Path(args.image_root) if args.image_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    class_names = read_class_names(args.classes)
    written = skipped = 0
    for label_path in iter_files(input_path, ".txt"):
        annotations = txt_to_annotations(label_path, image_root, args.txt_format, class_names)
        for annotation in annotations:
            output_path = output_path_for(label_path, input_path, output_dir, annotation.filename, ".json")
            if write_json(annotation, output_path, args.overwrite):
                written += 1
            else:
                skipped += 1
    return written, skipped


def txt_to_xml(args: argparse.Namespace) -> tuple[int, int]:
    input_path = Path(args.input)
    image_root = Path(args.image_root) if args.image_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    class_names = read_class_names(args.classes)
    written = skipped = 0
    for label_path in iter_files(input_path, ".txt"):
        annotations = txt_to_annotations(label_path, image_root, args.txt_format, class_names)
        for annotation in annotations:
            output_path = output_path_for(label_path, input_path, output_dir, annotation.filename, ".xml")
            if write_xml(annotation, output_path, args.overwrite):
                written += 1
            else:
                skipped += 1
    return written, skipped


def json_to_xml(args: argparse.Namespace) -> tuple[int, int]:
    input_path = Path(args.input)
    image_root = Path(args.image_root) if args.image_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    written = skipped = 0
    for label_path in iter_files(input_path, ".json"):
        data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            annotations = [dict_to_annotation(item, label_path, image_root) for item in data]
        else:
            annotations = [dict_to_annotation(data, label_path, image_root)]
        for annotation in annotations:
            output_path = output_path_for(label_path, input_path, output_dir, annotation.filename, ".xml")
            if write_xml(annotation, output_path, args.overwrite):
                written += 1
            else:
                skipped += 1
    return written, skipped


def main() -> int:
    args = parse_args()
    converters = {
        "txt2json": txt_to_json,
        "txt2xml": txt_to_xml,
        "json2xml": json_to_xml,
    }
    try:
        written, skipped = converters[args.command](args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.command}: written={written}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
