"""Split-CSV dataset used by the old baseline workflow.

The key difference from :mod:`voc_xml` is that the split file decides which
samples belong to train/val, while the real annotation parser still validates
each referenced file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .catalog import _parse_labelme_sample, _parse_voc_sample
from .image_io import load_rgb_image

SPLIT_LABEL_PATH_KEYS = (
    "relative_label_path",
    "relative_annotation_path",
    "label_path",
    "annotation_path",
)


def _normalize_relative_path(relative_path: str) -> Path:
    """Normalize CSV paths so Windows and POSIX separators behave the same."""
    return Path(relative_path.replace("\\", "/"))


def _load_split_rows(split_csv: str | Path) -> list[dict[str, str]]:
    """Read non-empty rows from a split CSV and trim header/value whitespace."""
    split_csv_path = Path(split_csv)
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    with split_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if row is None:
                continue
            normalized = {str(key).strip(): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            if any(value for value in normalized.values()):
                rows.append(normalized)

    if not rows:
        raise ValueError(f"Split CSV is empty: {split_csv_path}")
    return rows


def _resolve_split_annotation_path(row: dict[str, str], split_csv: Path) -> str:
    """Support several historical column names used by older split exports."""
    for key in SPLIT_LABEL_PATH_KEYS:
        value = row.get(key)
        if value:
            return value
    expected = ", ".join(SPLIT_LABEL_PATH_KEYS)
    raise ValueError(f"Split row in {split_csv} must contain one of: {expected}. Got: {sorted(row.keys())}")


def _resolve_class_names(
    parsed_samples: list[dict[str, Any]],
    class_names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Use the configured class list when given, otherwise infer it from data."""
    if class_names is not None:
        return list(class_names)

    discovered = sorted(
        {
            label_name
            for sample in parsed_samples
            for label_name in sample.get("label_names", [])
        }
    )
    if not discovered:
        raise ValueError("Could not discover any classes from the baseline split.")
    return discovered


class BaselineDetectionDataset(Dataset):
    """Dataset used by the streamlined baseline training pipeline.

    The dataset reads a split CSV that points to annotation files relative to
    ``label_root``. The CSV only decides which samples belong to the split;
    the annotation parser still validates each sample before it enters training.
    """

    def __init__(
        self,
        image_root: str | Path,
        label_root: str | Path,
        split_csv: str | Path,
        transforms=None,
        class_names: list[str] | tuple[str, ...] | None = None,
        keep_difficult: bool = False,
        polygon_policy: str = "exclude_sample",
        skip_quarantined: bool = True,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.split_csv = Path(split_csv)
        self.transforms = transforms
        self.keep_difficult = keep_difficult
        self.polygon_policy = polygon_policy
        self.skip_quarantined = skip_quarantined

        # The split CSV is the authoritative train/val selection mechanism for
        # this dataset type.
        split_rows = _load_split_rows(self.split_csv)
        parsed_samples: list[dict[str, Any]] = []
        self.skipped_samples: list[dict[str, Any]] = []

        for row in split_rows:
            relative_label_path = _resolve_split_annotation_path(row, self.split_csv)
            annotation_path = self.label_root / _normalize_relative_path(relative_label_path)
            if not annotation_path.exists():
                raise FileNotFoundError(
                    f"Annotation referenced by split CSV does not exist: {annotation_path}"
                )

            suffix = annotation_path.suffix.lower()
            # We keep both VOC XML and LabelMe JSON support because the old
            # project used both in production.
            if suffix == ".xml":
                sample = _parse_voc_sample(
                    annotation_path=annotation_path,
                    image_root=self.image_root,
                    label_root=self.label_root,
                    keep_difficult=self.keep_difficult,
                )
            elif suffix == ".json":
                sample = _parse_labelme_sample(
                    annotation_path=annotation_path,
                    image_root=self.image_root,
                    label_root=self.label_root,
                    polygon_policy=self.polygon_policy,
                )
            else:
                raise ValueError(f"Unsupported annotation suffix in split CSV: {annotation_path}")

            # Quarantined samples are recorded so the caller can inspect why they
            # were filtered out instead of silently losing data.
            if sample.get("quarantine_reason"):
                skipped_entry = {
                    "sample_id": sample.get("sample_id"),
                    "image_path": str(sample.get("image_path")),
                    "annotation_path": str(annotation_path),
                    "annotation_format": sample.get("annotation_format"),
                    "quarantine_reason": sample.get("quarantine_reason"),
                }
                if self.skip_quarantined:
                    self.skipped_samples.append(skipped_entry)
                    continue
                raise ValueError(
                    f"Baseline split contains a quarantined sample ({sample['quarantine_reason']}): "
                    f"{annotation_path}"
                )

            parsed_samples.append(sample)

        if not parsed_samples:
            raise ValueError(
                f"No usable samples were found in split CSV: {self.split_csv}. "
                "Check the split file, annotation paths, and quarantine policy."
            )

        self.class_names = _resolve_class_names(parsed_samples, class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Resolve label indices once so __getitem__ only has to load image bytes.
        self.samples: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(parsed_samples):
            labels: list[int] = []
            for label_name in sample.get("label_names", []):
                if label_name not in self.class_to_index:
                    raise ValueError(
                        f"Unknown class '{label_name}' in sample {sample['annotation_path']} "
                        f"for split {self.split_csv}"
                    )
                labels.append(self.class_to_index[label_name])

            self.samples.append(
                {
                    "image_id": sample_index,
                    "sample_id": sample["sample_id"],
                    "image_path": Path(sample["image_path"]),
                    "annotation_path": Path(sample["annotation_path"]),
                    "annotation_format": sample["annotation_format"],
                    "width": int(sample["width"]),
                    "height": int(sample["height"]),
                    "boxes": list(sample.get("boxes", [])),
                    "labels": labels,
                }
            )

        self.summary = {
            "split_csv": str(self.split_csv),
            "requested_samples": len(split_rows),
            "included_samples": len(self.samples),
            "skipped_samples": len(self.skipped_samples),
            "num_classes": self.num_classes,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        """Load one sample and return the pre-transform DETR-style target."""
        sample = self.samples[index]
        image = load_rgb_image(sample["image_path"])

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
            "orig_size": torch.tensor([sample["height"], sample["width"]], dtype=torch.long),
            "size": torch.tensor([sample["height"], sample["width"]], dtype=torch.long),
            "boxes": boxes,
            "labels": labels,
            "sample_id": sample["sample_id"],
            "annotation_path": str(sample["annotation_path"]),
            "image_path": str(sample["image_path"]),
            "annotation_format": sample["annotation_format"],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
