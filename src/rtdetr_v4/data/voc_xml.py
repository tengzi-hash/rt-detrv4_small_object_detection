"""Pascal VOC XML dataset support used by the migrated data layer.

This module is intentionally narrow: it only knows how to find XML files,
resolve the matching image, and convert annotations into the tensor-friendly
target format expected by DETR-style training code.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .image_io import load_image_size, load_rgb_image


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _parse_xml_root(annotation_path: Path) -> ET.Element:
    """Parse XML while tolerating the utf-8-sig header variant seen on Windows."""
    # Some annotation tools write `utf-8-sig` in the XML header, which
    # `ElementTree.parse(path)` does not always accept on Windows.
    text = annotation_path.read_text(encoding="utf-8-sig")
    text = text.replace("encoding='utf-8-sig'", "encoding='utf-8'", 1)
    text = text.replace('encoding="utf-8-sig"', 'encoding="utf-8"', 1)
    return ET.fromstring(text)


def _resolve_image_path(image_root: Path, label_root: Path, annotation_path: Path, xml_root: ET.Element) -> Path:
    """Guess the image path from the XML metadata and common sibling layouts."""
    relative_parent = annotation_path.relative_to(label_root).parent
    filename = xml_root.findtext("filename")
    candidates: list[Path] = []

    if filename:
        filename_path = Path(filename)
        candidates.append(image_root / relative_parent / filename_path.name)
        candidates.append(image_root / filename_path)
        stem = filename_path.stem
    else:
        stem = annotation_path.stem

    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidates.append(image_root / relative_parent / f"{stem}{extension}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for annotation: {annotation_path}")


class VOCXMLDetectionDataset(Dataset):
    """Load a folder of VOC XML annotations as a detection dataset."""

    def __init__(
        self,
        image_root: str | Path,
        label_root: str | Path,
        transforms=None,
        class_names: list[str] | tuple[str, ...] | None = None,
        keep_difficult: bool = False,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.transforms = transforms
        self.keep_difficult = keep_difficult

        # We index every XML file up front so class discovery and sample parsing
        # see the exact same annotation set.
        annotation_files = sorted(self.label_root.rglob("*.xml"))
        if not annotation_files:
            raise FileNotFoundError(f"No XML annotations found under: {self.label_root}")

        # Parse once during dataset construction so training does not repeatedly
        # touch XML files on every __getitem__ call.
        parsed_annotations: list[tuple[Path, ET.Element]] = []
        discovered_classes: set[str] = set()
        for annotation_path in annotation_files:
            xml_root = _parse_xml_root(annotation_path)
            parsed_annotations.append((annotation_path, xml_root))
            for obj in xml_root.findall("object"):
                name = obj.findtext("name")
                if name is not None:
                    discovered_classes.add(name.strip())

        if class_names is None:
            self.class_names = sorted(discovered_classes)
        else:
            self.class_names = list(class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.samples = []
        for sample_index, (annotation_path, xml_root) in enumerate(parsed_annotations):
            image_path = _resolve_image_path(self.image_root, self.label_root, annotation_path, xml_root)

            width = int(xml_root.findtext("size/width", "0") or 0)
            height = int(xml_root.findtext("size/height", "0") or 0)
            if width <= 0 or height <= 0:
                width, height = load_image_size(image_path)

            boxes = []
            labels = []
            # Invalid boxes are dropped here so downstream code can assume all
            # boxes are well-formed xyxy coordinates.
            for obj in xml_root.findall("object"):
                if not self.keep_difficult and int(obj.findtext("difficult", "0") or 0):
                    continue

                name = obj.findtext("name")
                if name is None:
                    continue
                name = name.strip()
                if name not in self.class_to_index:
                    raise ValueError(f"Unknown class '{name}' in annotation: {annotation_path}")

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
                labels.append(self.class_to_index[name])

            self.samples.append(
                {
                    "image_id": sample_index,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        """Load one image and return the raw DETR-style target dictionary."""
        sample = self.samples[index]
        image = load_rgb_image(sample["image_path"])

        height = sample["height"]
        width = sample["width"]
        boxes = (
            torch.tensor(sample["boxes"], dtype=torch.float32)
            if sample["boxes"]
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels = (
            torch.tensor(sample["labels"], dtype=torch.long)
            if sample["labels"]
            else torch.zeros((0,), dtype=torch.long)
        )
        target = {
            "image_id": torch.tensor(sample["image_id"], dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
