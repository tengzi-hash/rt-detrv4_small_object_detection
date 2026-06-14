from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ANNOTATION_EXTENSIONS = {".txt", ".json", ".html", ".htm", ".xml"}
BOX_COLORS = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (149, 165, 166),
]


@dataclass
class AnnotationBox:
    label: str
    box_xyxy: tuple[float, float, float, float]
    score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match images with annotation files and save boxed visualizations."
    )
    parser.add_argument("--images", required=True, help="Image file or directory.")
    parser.add_argument("--annotations", required=True, help="Annotation file or directory.")
    parser.add_argument("--output-dir", default="outputs/annotation_visualizations")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively.")
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Bounding-box outline width.",
    )
    parser.add_argument(
        "--txt-format",
        choices=("auto", "xyxy", "xywh", "yolo"),
        default="auto",
        help=(
            "TXT box format. Supports labels at the beginning or end of each line. "
            "Use yolo for class/label cx cy w h normalized boxes."
        ),
    )
    parser.add_argument(
        "--filter-label",
        default=None,
        help="Only keep images whose annotations contain this label.",
    )
    parser.add_argument(
        "--label-case-sensitive",
        action="store_true",
        help="Make --filter-label matching case-sensitive.",
    )
    parser.add_argument(
        "--copy-matches-dir",
        default=None,
        help=(
            "Directory where matched source images and annotation files are copied. "
            "Defaults to output-dir/filtered_<label> when --filter-label is set."
        ),
    )
    parser.add_argument(
        "--export-dataset-dir",
        default=None,
        help=(
            "Export matched samples into images/, labels/, and manifests/. "
            "Useful when images and labels are mixed in one folder."
        ),
    )
    return parser.parse_args()


def iter_files(path: Path, extensions: set[str], *, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in extensions
    )


def build_annotation_index(annotation_paths: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in annotation_paths:
        # Prefer the first matching file when mixed annotation formats exist.
        index.setdefault(path.stem, path)
    return index


def coerce_box(
    values: list[float],
    *,
    image_width: int,
    image_height: int,
    format_hint: str | None = None,
) -> tuple[float, float, float, float] | None:
    if len(values) < 4:
        return None

    coords = [float(value) for value in values[:4]]
    if max(abs(value) for value in coords) <= 1.5:
        if format_hint == "yolo":
            cx, cy, width, height = coords
            x0 = (cx - width / 2.0) * image_width
            y0 = (cy - height / 2.0) * image_height
            x1 = (cx + width / 2.0) * image_width
            y1 = (cy + height / 2.0) * image_height
        else:
            x0, y0, x1, y1 = (
                coords[0] * image_width,
                coords[1] * image_height,
                coords[2] * image_width,
                coords[3] * image_height,
            )
    elif format_hint == "xywh":
        x0, y0, width, height = coords
        x1, y1 = x0 + width, y0 + height
    elif format_hint == "yolo":
        cx, cy, width, height = coords
        x0, y0 = cx - width / 2.0, cy - height / 2.0
        x1, y1 = cx + width / 2.0, cy + height / 2.0
    else:
        x0, y0, x1, y1 = coords
        if x1 < x0 or y1 < y0:
            x0, y0, width, height = coords
            x1, y1 = x0 + width, y0 + height

    x0 = max(0.0, min(float(image_width), x0))
    x1 = max(0.0, min(float(image_width), x1))
    y0 = max(0.0, min(float(image_height), y0))
    y1 = max(0.0, min(float(image_height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def parse_xml(path: Path, image_width: int, image_height: int) -> list[AnnotationBox]:
    root = ET.fromstring(path.read_text(encoding="utf-8-sig"))
    boxes: list[AnnotationBox] = []
    for obj in root.findall(".//object"):
        label = (obj.findtext("name") or "object").strip() or "object"
        box_node = obj.find("bndbox")
        if box_node is None:
            continue
        raw_box = [
            float(box_node.findtext("xmin", "0") or 0),
            float(box_node.findtext("ymin", "0") or 0),
            float(box_node.findtext("xmax", "0") or 0),
            float(box_node.findtext("ymax", "0") or 0),
        ]
        box = coerce_box(raw_box, image_width=image_width, image_height=image_height)
        if box is not None:
            boxes.append(AnnotationBox(label=label, box_xyxy=box))
    return boxes


def parse_labelme_json(data: dict[str, Any], image_width: int, image_height: int) -> list[AnnotationBox]:
    boxes: list[AnnotationBox] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label") or "object")
        points = shape.get("points") or []
        if not points:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        box = coerce_box(
            [min(xs), min(ys), max(xs), max(ys)],
            image_width=image_width,
            image_height=image_height,
        )
        if box is not None:
            boxes.append(AnnotationBox(label=label, box_xyxy=box))
    return boxes


def extract_box_from_mapping(item: dict[str, Any], image_width: int, image_height: int) -> AnnotationBox | None:
    label = str(
        item.get("label")
        or item.get("label_name")
        or item.get("class_name")
        or item.get("name")
        or item.get("category")
        or item.get("category_id")
        or "object"
    )
    score = item.get("score")
    score_value = float(score) if isinstance(score, (int, float)) else None

    raw_box = (
        item.get("box_xyxy")
        or item.get("xyxy")
        or item.get("box")
        or item.get("bbox")
        or item.get("bndbox")
    )
    if isinstance(raw_box, dict):
        raw_box = [
            raw_box.get("xmin", raw_box.get("x0", raw_box.get("left", 0))),
            raw_box.get("ymin", raw_box.get("y0", raw_box.get("top", 0))),
            raw_box.get("xmax", raw_box.get("x1", raw_box.get("right", 0))),
            raw_box.get("ymax", raw_box.get("y1", raw_box.get("bottom", 0))),
        ]
    if not isinstance(raw_box, list | tuple) or len(raw_box) < 4:
        return None

    format_hint = "xywh" if "bbox" in item and "box_xyxy" not in item and "xyxy" not in item else None
    box = coerce_box(
        [float(value) for value in raw_box[:4]],
        image_width=image_width,
        image_height=image_height,
        format_hint=format_hint,
    )
    if box is None:
        return None
    return AnnotationBox(label=label, box_xyxy=box, score=score_value)


def walk_json_boxes(data: Any, image_width: int, image_height: int) -> list[AnnotationBox]:
    if isinstance(data, dict) and "shapes" in data:
        return parse_labelme_json(data, image_width, image_height)

    boxes: list[AnnotationBox] = []
    if isinstance(data, dict):
        direct = extract_box_from_mapping(data, image_width, image_height)
        if direct is not None:
            boxes.append(direct)
        for key in ("objects", "annotations", "detections", "predictions", "boxes", "results"):
            child = data.get(key)
            if child is not None:
                boxes.extend(walk_json_boxes(child, image_width, image_height))
    elif isinstance(data, list):
        for item in data:
            boxes.extend(walk_json_boxes(item, image_width, image_height))
    return boxes


def parse_json(path: Path, image_width: int, image_height: int) -> list[AnnotationBox]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
    return walk_json_boxes(data, image_width, image_height)


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def parse_txt_line(
    line: str,
    image_width: int,
    image_height: int,
    *,
    txt_format: str = "auto",
    image_path: Path | None = None,
) -> AnnotationBox | None:
    clean = line.strip()
    if not clean or clean.startswith("#"):
        return None
    tokens = re.split(r"[\s,;]+", clean)
    tokens = [token for token in tokens if token]
    if tokens and not NUMBER_RE.fullmatch(tokens[0]):
        first_token = Path(tokens[0]).name
        first_stem = Path(first_token).stem
        if image_path is not None and first_stem and first_stem != image_path.stem and first_token != image_path.name:
            return None
    numbers = [float(token) for token in tokens if NUMBER_RE.fullmatch(token)]
    text_tokens = [token for token in tokens if token and not NUMBER_RE.fullmatch(token)]
    if len(numbers) < 4:
        return None

    # The label may be first or last. Numeric class ids are also accepted for
    # YOLO-like rows such as "0 0.5 0.5 0.1 0.1".
    label = text_tokens[-1] if text_tokens else "object"
    if txt_format == "yolo":
        if len(numbers) >= 5:
            if not text_tokens:
                label = str(int(numbers[0]))
            coords = numbers[1:5]
        else:
            coords = numbers[:4]
        box = coerce_box(
            coords,
            image_width=image_width,
            image_height=image_height,
            format_hint="yolo",
        )
    elif txt_format in {"xyxy", "xywh"}:
        box = coerce_box(
            numbers[:4],
            image_width=image_width,
            image_height=image_height,
            format_hint=txt_format,
        )
    elif len(numbers) >= 5 and max(abs(value) for value in numbers[1:5]) <= 1.5:
        if not text_tokens:
            label = str(int(numbers[0]))
        box = coerce_box(
            numbers[1:5],
            image_width=image_width,
            image_height=image_height,
            format_hint="yolo",
        )
    else:
        box = coerce_box(numbers[:4], image_width=image_width, image_height=image_height)
    if box is None:
        return None
    return AnnotationBox(label=label, box_xyxy=box)


def parse_txt(
    path: Path,
    image_width: int,
    image_height: int,
    *,
    txt_format: str,
    image_path: Path | None = None,
) -> list[AnnotationBox]:
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        box = parse_txt_line(line, image_width, image_height, txt_format=txt_format, image_path=image_path)
        if box is not None:
            boxes.append(box)
    return boxes


def parse_html(
    path: Path,
    image_width: int,
    image_height: int,
    *,
    txt_format: str,
    image_path: Path | None = None,
) -> list[AnnotationBox]:
    text = path.read_text(encoding="utf-8-sig")
    boxes: list[AnnotationBox] = []

    for match in re.finditer(r"<script[^>]*type=[\"']application/json[\"'][^>]*>(.*?)</script>", text, re.S | re.I):
        try:
            boxes.extend(walk_json_boxes(json.loads(html.unescape(match.group(1))), image_width, image_height))
        except json.JSONDecodeError:
            pass

    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
        row_text = re.sub(r"<[^>]+>", " ", row_match.group(1))
        row_text = html.unescape(row_text)
        box = parse_txt_line(row_text, image_width, image_height, txt_format=txt_format, image_path=image_path)
        if box is not None:
            boxes.append(box)

    if boxes:
        return boxes

    plain = html.unescape(re.sub(r"<[^>]+>", "\n", text))
    for line in plain.splitlines():
        box = parse_txt_line(line, image_width, image_height, txt_format=txt_format, image_path=image_path)
        if box is not None:
            boxes.append(box)
    return boxes


def parse_annotation(
    path: Path,
    image_width: int,
    image_height: int,
    *,
    txt_format: str,
    image_path: Path | None = None,
) -> list[AnnotationBox]:
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return parse_xml(path, image_width, image_height)
    if suffix == ".json":
        return parse_json(path, image_width, image_height)
    if suffix == ".txt":
        return parse_txt(path, image_width, image_height, txt_format=txt_format, image_path=image_path)
    if suffix in {".html", ".htm"}:
        return parse_html(path, image_width, image_height, txt_format=txt_format, image_path=image_path)
    raise ValueError(f"Unsupported annotation extension: {path}")


def draw_boxes(image: Image.Image, boxes: list[AnnotationBox], *, line_width: int) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    label_to_color: dict[str, tuple[int, int, int]] = {}

    for box in boxes:
        color = label_to_color.setdefault(box.label, BOX_COLORS[len(label_to_color) % len(BOX_COLORS)])
        x0, y0, x1, y1 = box.box_xyxy
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        caption = box.label if box.score is None else f"{box.label} {box.score:.2f}"
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_top = max(0, int(y0) - text_height - 4)
        draw.rectangle((x0, text_top, x0 + text_width + 6, text_top + text_height + 4), fill=color)
        draw.text((x0 + 3, text_top + 2), caption, fill=(255, 255, 255), font=font)
    return canvas


def label_matches(candidate: str, expected: str, *, case_sensitive: bool) -> bool:
    if case_sensitive:
        return candidate == expected
    return candidate.casefold() == expected.casefold()


def sanitize_path_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    return cleaned.strip("._") or "label"


def copy_match_files(
    *,
    image_path: Path,
    annotation_path: Path,
    boxes: list[AnnotationBox],
    image_root: Path,
    annotation_root: Path,
    copy_root: Path,
) -> dict[str, str]:
    image_relative = image_path.name if image_root.is_file() else image_path.relative_to(image_root)
    if annotation_root.is_file() and annotation_path.suffix.lower() == ".txt":
        annotation_relative = image_path.with_suffix(".xml").name
    else:
        annotation_relative = annotation_path.name if annotation_root.is_file() else annotation_path.relative_to(annotation_root)

    image_output = copy_root / "images" / image_relative
    annotation_output = copy_root / "labels" / annotation_relative
    image_output.parent.mkdir(parents=True, exist_ok=True)
    annotation_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_output)
    if annotation_root.is_file() and annotation_path.suffix.lower() == ".txt":
        write_voc_xml(
            annotation_output,
            image_path=image_output,
            source_image_path=image_path,
            boxes=boxes,
        )
    elif annotation_path.resolve() != annotation_output.resolve():
        shutil.copy2(annotation_path, annotation_output)
    return {
        "copied_image_path": str(image_output),
        "copied_label_path": str(annotation_output),
        "copied_annotation_path": str(annotation_output),
    }


def add_xml_child(parent: ET.Element, tag: str, text: str | int | float | None = None) -> ET.Element:
    child = ET.SubElement(parent, tag)
    if text is not None:
        child.text = str(text)
    return child


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "\t"
    child_indent = "\n" + (level + 1) * "\t"
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in children:
            indent_xml(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def write_voc_xml(
    output_path: Path,
    *,
    image_path: Path,
    source_image_path: Path,
    boxes: list[AnnotationBox],
) -> None:
    with Image.open(source_image_path) as image:
        width, height = image.size
        depth = len(image.getbands())

    root = ET.Element("annotation")
    add_xml_child(root, "folder", image_path.parent.name)
    add_xml_child(root, "filename", image_path.name)
    add_xml_child(root, "path", str(image_path))
    source = add_xml_child(root, "source")
    add_xml_child(source, "database", "Unknown")
    size = add_xml_child(root, "size")
    add_xml_child(size, "width", width)
    add_xml_child(size, "height", height)
    add_xml_child(size, "depth", depth)
    add_xml_child(root, "segmented", 0)

    for box in boxes:
        x0, y0, x1, y1 = box.box_xyxy
        obj = add_xml_child(root, "object")
        add_xml_child(obj, "name", box.label)
        add_xml_child(obj, "pose", "Unspecified")
        add_xml_child(obj, "truncated", 0)
        add_xml_child(obj, "difficult", 0)
        bndbox = add_xml_child(obj, "bndbox")
        add_xml_child(bndbox, "xmin", int(round(x0)))
        add_xml_child(bndbox, "ymin", int(round(y0)))
        add_xml_child(bndbox, "xmax", int(round(x1)))
        add_xml_child(bndbox, "ymax", int(round(y1)))

    indent_xml(root)
    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)


def write_export_manifests(copy_root: Path, matches: list[dict[str, Any]]) -> None:
    manifest_root = copy_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "matches.json").write_text(
        json.dumps(matches, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class_counts: dict[str, int] = {}
    rows = []
    for match in matches:
        labels = list(match.get("matched_labels") or [])
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        copied_image = Path(match["copied_image_path"])
        copied_label = Path(match.get("copied_label_path") or match["copied_annotation_path"])
        rows.append(
            {
                "image_path": copied_image.relative_to(copy_root).as_posix(),
                "label_path": copied_label.relative_to(copy_root).as_posix(),
                "source_image_path": match["image_path"],
                "source_label_path": match["annotation_path"],
                "labels": "|".join(labels),
                "box_count": str(match.get("matched_box_count", 0)),
            }
        )

    with (manifest_root / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label_path",
                "source_image_path",
                "source_label_path",
                "labels",
                "box_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (manifest_root / "class_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label_name", "sample_count"])
        writer.writeheader()
        for label in sorted(class_counts):
            writer.writerow({"label_name": label, "sample_count": class_counts[label]})


def output_path_for(image_path: Path, image_root: Path, output_root: Path) -> Path:
    relative = image_path.name if image_root.is_file() else image_path.relative_to(image_root)
    return (output_root / relative).with_suffix(".jpg")


def main() -> None:
    args = parse_args()
    image_root = Path(args.images).resolve()
    annotation_root = Path(args.annotations).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    copy_matches_root = None
    if args.export_dataset_dir:
        copy_matches_root = Path(args.export_dataset_dir).resolve()
    elif args.copy_matches_dir:
        copy_matches_root = Path(args.copy_matches_dir).resolve()
    elif args.filter_label:
        copy_matches_root = output_root / f"filtered_{sanitize_path_part(args.filter_label)}"
    if copy_matches_root is not None:
        copy_matches_root.mkdir(parents=True, exist_ok=True)

    image_paths = iter_files(image_root, IMAGE_EXTENSIONS, recursive=args.recursive)
    annotation_paths = iter_files(annotation_root, ANNOTATION_EXTENSIONS, recursive=args.recursive)
    annotation_index = build_annotation_index(annotation_paths)
    shared_annotation_paths = [
        path
        for path in annotation_paths
        if path.suffix.lower() in {".txt", ".json", ".html", ".htm"}
    ]

    summary = {
        "images": len(image_paths),
        "annotations": len(annotation_paths),
        "matched": 0,
        "filter_label": args.filter_label,
        "unmatched_images": [],
        "failed": [],
        "matches": [],
    }

    for image_path in image_paths:
        annotation_path = annotation_index.get(image_path.stem)
        candidate_paths = [annotation_path] if annotation_path is not None else shared_annotation_paths
        if not candidate_paths:
            summary["unmatched_images"].append(str(image_path))
            continue
        try:
            with Image.open(image_path) as image:
                boxes: list[AnnotationBox] = []
                used_annotation_path = None
                for candidate_path in candidate_paths:
                    candidate_boxes = parse_annotation(
                        candidate_path,
                        image.width,
                        image.height,
                        txt_format=args.txt_format,
                        image_path=image_path,
                    )
                    if candidate_boxes:
                        boxes = candidate_boxes
                        used_annotation_path = candidate_path
                        break
                if used_annotation_path is None:
                    summary["unmatched_images"].append(str(image_path))
                    continue
                matched_boxes = boxes
                if args.filter_label:
                    matched_boxes = [
                        box
                        for box in boxes
                        if label_matches(
                            box.label,
                            args.filter_label,
                            case_sensitive=args.label_case_sensitive,
                        )
                    ]
                    if not matched_boxes:
                        continue
                visualization = draw_boxes(image, boxes, line_width=args.line_width)
                save_path = output_path_for(image_path, image_root, output_root)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                visualization.save(save_path, quality=95)
            match_record = {
                "image_path": str(image_path),
                "annotation_path": str(used_annotation_path),
                "visualization_path": str(save_path),
                "matched_labels": sorted({box.label for box in matched_boxes}),
                "matched_box_count": len(matched_boxes),
            }
            if copy_matches_root is not None:
                match_record.update(
                    copy_match_files(
                        image_path=image_path,
                        annotation_path=used_annotation_path,
                        boxes=matched_boxes,
                        image_root=image_root,
                        annotation_root=annotation_root,
                        copy_root=copy_matches_root,
                    )
                )
            summary["matches"].append(match_record)
            summary["matched"] += 1
        except Exception as exc:
            summary["failed"].append(
                {
                    "image_path": str(image_path),
                    "annotation_path": str(annotation_path or candidate_paths[0]),
                    "error": str(exc),
                }
            )

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if copy_matches_root is not None:
        (copy_matches_root / "matches.json").write_text(
            json.dumps(summary["matches"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_export_manifests(copy_matches_root, summary["matches"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
