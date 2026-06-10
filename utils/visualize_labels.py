"""Visualize labeled detection samples into an onhold folder.

The tool is intended for quick dataset inspection. It reads VOC XML labels
from either a built ``datasets`` folder or raw folders such as ``data_raw``.
It also understands the JSON format emitted by ``utils/label_convert.py`` and
the data5 semicolon TXT format.

Examples:
  python utils/visualize_labels.py datasets --classes BoltHead CotterPin
  python utils/visualize_labels.py data_raw/data5 --classes PullTab --limit 50
  python utils/visualize_labels.py data_raw --output-dir onhold/raw_check --export-mode all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
LABEL_SUFFIXES = (".xml", ".json", ".txt")


@dataclass(frozen=True)
class Box:
    cls: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    boxes: tuple[Box, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw labeled boxes into an onhold folder.")
    parser.add_argument("source", help="Dataset root, raw root, label file, or image file.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Class names to extract. Accepts spaces or comma-separated names. Empty means all labeled samples.",
    )
    parser.add_argument("--output-dir", default=None, help="Output folder. Default: onhold/visualized_<source>_<classes>.")
    parser.add_argument("--image-root", default=None, help="Optional image root override.")
    parser.add_argument("--label-root", default=None, help="Optional label root override.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum samples to write. 0 means no limit.")
    parser.add_argument("--draw-only-selected", action="store_true", help="Draw only selected classes instead of all boxes.")
    parser.add_argument(
        "--export-mode",
        choices=("visualize", "all"),
        default="visualize",
        help="visualize writes only drawn images; all also copies matched labels and images.",
    )
    parser.add_argument("--copy-label", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--copy-image", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visualizations.")
    return parser.parse_args()


def normalize_classes(values: list[str]) -> set[str]:
    classes: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                classes.add(item)
    return classes


def int_text(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(round(float(value.strip())))
    except ValueError:
        return None


def parse_xml(label_path: Path) -> tuple[str | None, list[Box]]:
    root = ET.parse(label_path).getroot()
    filename = (root.findtext("filename") or "").strip() or None
    boxes: list[Box] = []
    for obj in root.findall("object"):
        cls = (obj.findtext("name") or "").strip()
        box_el = obj.find("bndbox")
        if not cls or box_el is None:
            continue
        coords = [int_text(box_el.findtext(name)) for name in ("xmin", "ymin", "xmax", "ymax")]
        if any(value is None for value in coords):
            continue
        xmin, ymin, xmax, ymax = [int(value) for value in coords if value is not None]
        if xmax > xmin and ymax > ymin:
            boxes.append(Box(cls, xmin, ymin, xmax, ymax))
    return filename, boxes


def parse_json(label_path: Path) -> tuple[str | None, list[Box]]:
    data = json.loads(label_path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        if not data:
            return None, []
        data = data[0]
    filename = str(data.get("filename") or data.get("image") or "") or None
    boxes: list[Box] = []
    for obj in data.get("objects") or data.get("annotations") or []:
        cls = str(obj.get("name") or obj.get("label") or obj.get("class") or "").strip()
        bndbox = obj.get("bndbox") or obj.get("box") or obj
        if not cls:
            continue
        try:
            box = Box(
                cls,
                int(round(float(bndbox["xmin"]))),
                int(round(float(bndbox["ymin"]))),
                int(round(float(bndbox["xmax"]))),
                int(round(float(bndbox["ymax"]))),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if box.xmax > box.xmin and box.ymax > box.ymin:
            boxes.append(box)
    return filename, boxes


def parse_txt(label_path: Path) -> tuple[str | None, list[Box]]:
    filename: str | None = None
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ";" not in line:
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 6:
            continue
        raw_filename, xmin, ymin, xmax, ymax, cls = parts
        filename = filename or Path(raw_filename).name
        try:
            box = Box(cls, int(round(float(xmin))), int(round(float(ymin))), int(round(float(xmax))), int(round(float(ymax))))
        except ValueError:
            continue
        if box.xmax > box.xmin and box.ymax > box.ymin:
            boxes.append(box)
    return filename, boxes


def parse_label(label_path: Path) -> tuple[str | None, list[Box]]:
    suffix = label_path.suffix.lower()
    if suffix == ".xml":
        return parse_xml(label_path)
    if suffix == ".json":
        return parse_json(label_path)
    if suffix == ".txt":
        return parse_txt(label_path)
    return None, []


def build_image_indexes(roots: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for root in roots:
        if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
            images = [root]
        elif root.is_dir():
            images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        else:
            images = []
        for image in images:
            by_name.setdefault(image.name.lower(), []).append(image)
            by_stem.setdefault(image.stem.lower(), []).append(image)
    return by_name, by_stem


def nearest(paths: list[Path], anchor: Path) -> Path | None:
    if not paths:
        return None
    anchor_parent = anchor.parent.resolve()

    def key(path: Path) -> tuple[int, int, str]:
        parent = path.parent.resolve()
        same_parent = 0 if parent == anchor_parent else 1
        common = 0
        for left, right in zip(parent.parts, anchor_parent.parts):
            if left != right:
                break
            common += 1
        return same_parent, -common, str(path)

    return sorted(paths, key=key)[0]


def find_image(label_path: Path, filename: str | None, by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> Path | None:
    if filename:
        raw_name = Path(filename).name
        direct = label_path.parent / raw_name
        if direct.is_file():
            return direct
        candidates = by_name.get(raw_name.lower(), []) + by_stem.get(Path(raw_name).stem.lower(), [])
        found = nearest(sorted(set(candidates)), label_path)
        if found is not None:
            return found
    for suffix in IMAGE_SUFFIXES:
        direct = label_path.with_suffix(suffix)
        if direct.is_file():
            return direct
    return nearest(by_stem.get(label_path.stem.lower(), []), label_path)


def infer_roots(source: Path, image_root: str | None, label_root: str | None) -> tuple[list[Path], list[Path]]:
    if source.is_file():
        if source.suffix.lower() in LABEL_SUFFIXES:
            labels = [source]
            images = [Path(image_root)] if image_root else [source.parent]
        else:
            labels = [Path(label_root)] if label_root else [source.parent]
            images = [source]
        return labels, images

    labels = [Path(label_root)] if label_root else []
    images = [Path(image_root)] if image_root else []
    if not labels:
        built_labels = source / "labels"
        labels = [built_labels] if built_labels.is_dir() else [source]
    if not images:
        built_images = source / "images"
        images = [built_images] if built_images.is_dir() else [source]
    return labels, images


def iter_label_files(roots: list[Path]) -> list[Path]:
    labels: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in LABEL_SUFFIXES:
            labels.append(root)
        elif root.is_dir():
            labels.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in LABEL_SUFFIXES)
    return sorted(set(labels))


def selected_boxes(boxes: list[Box], classes: set[str]) -> list[Box]:
    if not classes:
        return boxes
    return [box for box in boxes if box.cls in classes]


def color_for_class(cls: str) -> tuple[int, int, int]:
    digest = hashlib.md5(cls.encode("utf-8")).digest()
    return 64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160


def draw_sample(sample: Sample, output_path: Path, classes: set[str], draw_only_selected: bool, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    boxes = selected_boxes(list(sample.boxes), classes) if draw_only_selected else list(sample.boxes)
    for box in boxes:
        color = color_for_class(box.cls)
        width = max(2, round(min(canvas.size) / 700))
        draw.rectangle((box.xmin, box.ymin, box.xmax, box.ymax), outline=color, width=width)
        label = box.cls
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        x0 = max(0, min(box.xmin, canvas.width - text_w - 4))
        y0 = max(0, box.ymin - text_h - 5)
        draw.rectangle((x0, y0, x0 + text_w + 4, y0 + text_h + 4), fill=color)
        draw.text((x0 + 2, y0 + 2), label, fill=(255, 255, 255), font=font)
    canvas.save(output_path, quality=95)
    return True


def safe_output_name(image_path: Path, seen: dict[str, int]) -> str:
    stem = image_path.stem
    seen[stem] = seen.get(stem, 0) + 1
    if seen[stem] == 1:
        return f"{stem}.jpg"
    return f"{stem}__{seen[stem]:04d}.jpg"


def default_output_dir(source: Path, classes: set[str]) -> Path:
    class_part = "all" if not classes else "_".join(sorted(classes))
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in f"{source.name}_{class_part}")
    return Path("onhold") / f"visualized_{clean}"


def copy_sidecar(src: Path, dst_dir: Path, overwrite: bool) -> str:
    dst = dst_dir / src.name
    if dst.exists() and not overwrite:
        return str(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def should_copy_label(args: argparse.Namespace) -> bool:
    return args.export_mode == "all" or args.copy_label


def should_copy_image(args: argparse.Namespace) -> bool:
    return args.export_mode == "all" or args.copy_image


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    classes = normalize_classes(args.classes)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(source, classes)
    label_roots, image_roots = infer_roots(source, args.image_root, args.label_root)
    by_name, by_stem = build_image_indexes(image_roots)

    samples: list[Sample] = []
    scanned = 0
    no_image = 0
    no_box = 0
    no_match = 0
    for label_path in iter_label_files(label_roots):
        scanned += 1
        filename, boxes = parse_label(label_path)
        if not boxes:
            no_box += 1
            continue
        hits = selected_boxes(boxes, classes)
        if not hits:
            no_match += 1
            continue
        image_path = find_image(label_path, filename, by_name, by_stem)
        if image_path is None:
            no_image += 1
            continue
        samples.append(Sample(image_path=image_path, label_path=label_path, boxes=tuple(boxes)))
        if args.limit > 0 and len(samples) >= args.limit:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir = output_dir / "visualize"
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    seen_names: dict[str, int] = {}
    written = skipped = 0
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["visualization", "image", "label", "matched_classes", "box_count", "matched_box_count"])
        for sample in samples:
            output_name = safe_output_name(sample.image_path, seen_names)
            vis_path = visualize_dir / output_name
            if draw_sample(sample, vis_path, classes, args.draw_only_selected, args.overwrite):
                written += 1
            else:
                skipped += 1
            if should_copy_label(args):
                copy_sidecar(sample.label_path, labels_dir, args.overwrite)
            if should_copy_image(args):
                copy_sidecar(sample.image_path, images_dir, args.overwrite)
            hits = selected_boxes(list(sample.boxes), classes)
            writer.writerow(
                [
                    str(vis_path),
                    str(sample.image_path),
                    str(sample.label_path),
                    ";".join(sorted({box.cls for box in hits})),
                    len(sample.boxes),
                    len(hits),
                ]
            )

    print(f"source={source}")
    print(f"output_dir={output_dir}")
    print(f"visualize_dir={visualize_dir}")
    if should_copy_label(args):
        print(f"labels_dir={labels_dir}")
    if should_copy_image(args):
        print(f"images_dir={images_dir}")
    print(f"labels_scanned={scanned}, matched_samples={len(samples)}, written={written}, skipped={skipped}")
    print(f"no_box={no_box}, no_class_match={no_match}, no_image={no_image}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
