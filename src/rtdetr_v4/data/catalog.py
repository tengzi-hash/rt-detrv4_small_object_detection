"""Dataset scanning helpers shared by the migrated detection datasets.

This module is the bridge between raw annotation files on disk and the compact
manifest/sample dictionaries used by the rest of the data layer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .image_io import load_image_size
from .voc_xml import SUPPORTED_IMAGE_EXTENSIONS, _parse_xml_root, _resolve_image_path


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read LabelMe JSON with BOM-tolerant decoding."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative_sample_id(path: Path, root: Path) -> str:
    """Create a stable, path-based sample id relative to the image root."""
    return path.relative_to(root).with_suffix("").as_posix()


def _resolve_labelme_image_path(
    image_root: Path,
    label_root: Path,
    annotation_path: Path,
    image_path: str | None,
) -> Path:
    """Resolve the image file referenced by a LabelMe annotation."""
    relative_parent = annotation_path.relative_to(label_root).parent
    candidates: list[Path] = []
    if image_path:
        image_path_obj = Path(image_path)
        if image_path_obj.is_absolute():
            candidates.append(image_path_obj)
        candidates.append(image_root / relative_parent / image_path_obj.name)
        candidates.append(image_root / image_path_obj)
        stem = image_path_obj.stem
    else:
        stem = annotation_path.stem

    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidates.append(image_root / relative_parent / f"{stem}{extension}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for annotation: {annotation_path}")


def _polygon_to_bbox(points: list[list[float]] | list[tuple[float, float]]) -> list[float]:
    """Collapse polygon points into one axis-aligned bounding box."""
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _parse_voc_sample(
    annotation_path: Path,
    image_root: Path,
    label_root: Path,
    keep_difficult: bool,
) -> dict[str, Any]:
    """Parse one VOC XML file into the common sample dictionary format."""
    xml_root = _parse_xml_root(annotation_path)
    image_path = _resolve_image_path(image_root, label_root, annotation_path, xml_root)

    width = int(xml_root.findtext("size/width", "0") or 0)
    height = int(xml_root.findtext("size/height", "0") or 0)
    if width <= 0 or height <= 0:
        width, height = load_image_size(image_path)

    boxes: list[list[float]] = []
    label_names: list[str] = []
    # We filter unsupported or malformed objects here so every downstream
    # consumer receives a clean sample description.
    for obj in xml_root.findall("object"):
        if not keep_difficult and int(obj.findtext("difficult", "0") or 0):
            continue
        name = obj.findtext("name")
        if name is None:
            continue

        box_node = obj.find("bndbox")
        if box_node is None:
            continue
        xmin = float(box_node.findtext("xmin", "0") or 0)
        ymin = float(box_node.findtext("ymin", "0") or 0)
        xmax = float(box_node.findtext("xmax", "0") or 0)
        ymax = float(box_node.findtext("ymax", "0") or 0)
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        label_names.append(name.strip())

    return {
        "sample_id": _relative_sample_id(image_path, image_root),
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "annotation_format": "voc_xml",
        "relative_parent": str(annotation_path.relative_to(label_root).parent),
        "width": width,
        "height": height,
        "boxes": boxes,
        "label_names": label_names,
        "shape_type_counts": {"rectangle": len(boxes)},
        "contains_polygon": False,
        "polygon_count": 0,
        "quarantine_reason": None,
    }


def _parse_labelme_sample(
    annotation_path: Path,
    image_root: Path,
    label_root: Path,
    polygon_policy: str,
) -> dict[str, Any]:
    """Parse one LabelMe JSON file into the common sample dictionary format."""
    data = _read_json_file(annotation_path)
    image_path = _resolve_labelme_image_path(
        image_root=image_root,
        label_root=label_root,
        annotation_path=annotation_path,
        image_path=data.get("imagePath"),
    )

    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        width, height = load_image_size(image_path)

    boxes: list[list[float]] = []
    label_names: list[str] = []
    polygon_count = 0
    shape_type_counts = Counter()
    unsupported_shapes: list[dict[str, Any]] = []

    # Shape handling is explicit because later migration steps may want to
    # tighten or relax the quarantine rules without changing dataset callers.
    for index, shape in enumerate(data.get("shapes", [])):
        label = str(shape.get("label", "")).strip()
        shape_type = str(shape.get("shape_type", "rectangle"))
        points = shape.get("points", [])
        if not label or not points:
            continue

        if shape_type == "rectangle" and len(points) >= 2:
            (x0, y0), (x1, y1) = points[:2]
            xmin, xmax = sorted((float(x0), float(x1)))
            ymin, ymax = sorted((float(y0), float(y1)))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            label_names.append(label)
            shape_type_counts["rectangle"] += 1
            continue

        if shape_type == "polygon" and len(points) >= 3:
            polygon_count += 1
            shape_type_counts["polygon"] += 1
            if polygon_policy == "convert_to_bbox":
                xmin, ymin, xmax, ymax = _polygon_to_bbox(points)
                boxes.append([xmin, ymin, xmax, ymax])
                label_names.append(label)
            continue

        unsupported_shapes.append(
            {
                "shape_index": index,
                "shape_type": shape_type,
                "label": label,
            }
        )
        shape_type_counts["unsupported"] += 1

    quarantine_reason = None
    # Quarantine keeps the sample visible to operators while still preventing
    # accidental training on unsupported annotations.
    if polygon_count > 0 and polygon_policy == "exclude_sample":
        quarantine_reason = "contains_polygon_labelme"
    elif unsupported_shapes:
        quarantine_reason = "contains_unsupported_labelme_shape"

    return {
        "sample_id": _relative_sample_id(image_path, image_root),
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "annotation_format": "labelme_json",
        "relative_parent": str(annotation_path.relative_to(label_root).parent),
        "width": width,
        "height": height,
        "boxes": boxes,
        "label_names": label_names,
        "shape_type_counts": dict(shape_type_counts),
        "contains_polygon": polygon_count > 0,
        "polygon_count": polygon_count,
        "unsupported_shapes": unsupported_shapes,
        "quarantine_reason": quarantine_reason,
    }


def _build_class_mapping(
    included_samples: list[dict[str, Any]],
    class_names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Return the final class order used to map class names to indices."""
    if class_names:
        return list(class_names)

    discovered = sorted(
        {
            label_name
            for sample in included_samples
            for label_name in sample.get("label_names", [])
        }
    )
    return discovered


def _attach_label_indices(
    samples: list[dict[str, Any]],
    class_names: list[str],
    *,
    strict: bool,
) -> None:
    """Attach integer labels to samples once the class order is known."""
    class_to_index = {name: index for index, name in enumerate(class_names)}
    for sample in samples:
        labels: list[int] = []
        for label_name in sample.get("label_names", []):
            if label_name not in class_to_index:
                if strict:
                    raise ValueError(f"Unknown class '{label_name}' in sample {sample['annotation_path']}")
                labels.append(-1)
                continue
            labels.append(class_to_index[label_name])
        sample["labels"] = labels


def scan_detection_dataset(
    image_root: str | Path,
    label_root: str | Path,
    *,
    class_names: list[str] | tuple[str, ...] | None = None,
    keep_difficult: bool = False,
    polygon_policy: str = "exclude_sample",
) -> dict[str, Any]:
    """Scan a mixed annotation folder and build a reusable dataset manifest."""
    image_root = Path(image_root)
    label_root = Path(label_root)
    if polygon_policy not in {"exclude_sample", "convert_to_bbox", "ignore_shape"}:
        raise ValueError(f"Unsupported polygon_policy: {polygon_policy}")

    included_samples: list[dict[str, Any]] = []
    quarantined_samples: list[dict[str, Any]] = []
    unsupported_shapes: list[dict[str, Any]] = []

    annotation_paths = sorted(
        list(label_root.rglob("*.xml")) + list(label_root.rglob("*.json"))
    )
    if not annotation_paths:
        raise FileNotFoundError(f"No annotations found under: {label_root}")

    # Every annotation file is normalized into the same sample schema so the
    # training layer does not need format-specific branches later.
    for annotation_path in annotation_paths:
        if annotation_path.suffix.lower() == ".xml":
            sample = _parse_voc_sample(
                annotation_path=annotation_path,
                image_root=image_root,
                label_root=label_root,
                keep_difficult=keep_difficult,
            )
        else:
            sample = _parse_labelme_sample(
                annotation_path=annotation_path,
                image_root=image_root,
                label_root=label_root,
                polygon_policy=polygon_policy,
            )
            for item in sample.get("unsupported_shapes", []):
                unsupported_shapes.append(
                    {
                        "file": sample["annotation_path"],
                        **item,
                    }
                )

        if sample.get("quarantine_reason"):
            quarantined_samples.append(sample)
        else:
            included_samples.append(sample)

    resolved_class_names = _build_class_mapping(included_samples, class_names)
    _attach_label_indices(included_samples, resolved_class_names, strict=True)
    _attach_label_indices(quarantined_samples, resolved_class_names, strict=False)

    class_counts = Counter()
    for sample in included_samples:
        class_counts.update(sample.get("label_names", []))

    image_files = {
        _relative_sample_id(path, image_root)
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    }
    included_ids = {sample["sample_id"] for sample in included_samples}
    quarantined_ids = {sample["sample_id"] for sample in quarantined_samples}
    label_ids = included_ids | quarantined_ids

    return {
        "image_root": str(image_root),
        "label_root": str(label_root),
        "keep_difficult": keep_difficult,
        "polygon_policy": polygon_policy,
        "class_names": resolved_class_names,
        "class_counts": dict(class_counts),
        "included_samples": included_samples,
        "quarantined_samples": quarantined_samples,
        "unsupported_shapes": unsupported_shapes,
        "summary": {
            "total_images": len(image_files),
            "total_annotation_files": len(annotation_paths),
            "included_samples": len(included_samples),
            "quarantined_samples": len(quarantined_samples),
            "images_without_annotation": len(image_files - label_ids),
            "labels_without_image": len(label_ids - image_files),
            "num_classes": len(resolved_class_names),
        },
    }


def save_dataset_manifest(scan_result: dict[str, Any], path: str | Path) -> Path:
    """Persist a scan result so later runs can skip a full directory walk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    """Load a previously saved dataset manifest."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
