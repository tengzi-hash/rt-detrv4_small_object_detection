"""Dataset wrapper for folders containing both VOC XML and LabelMe JSON."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .catalog import load_dataset_manifest, scan_detection_dataset
from .image_io import load_rgb_image


def _manifest_matches_request(
    manifest: dict,
    *,
    image_root: Path,
    label_root: Path,
    class_names: list[str] | tuple[str, ...] | None,
    keep_difficult: bool,
    polygon_policy: str,
) -> bool:
    """Check whether a cached manifest was built with the same dataset options."""
    if Path(manifest.get("image_root", "")) != image_root:
        return False
    if Path(manifest.get("label_root", "")) != label_root:
        return False
    if manifest.get("polygon_policy") != polygon_policy:
        return False
    if bool(manifest.get("keep_difficult", False)) != keep_difficult:
        return False
    if class_names is not None and list(manifest.get("class_names", [])) != list(class_names):
        return False
    return True


class MixedDetectionDataset(Dataset):
    """Build a detection dataset from a manifest or by scanning the label root."""

    def __init__(
        self,
        image_root: str | Path,
        label_root: str | Path,
        transforms=None,
        class_names: list[str] | tuple[str, ...] | None = None,
        keep_difficult: bool = False,
        polygon_policy: str = "exclude_sample",
        manifest_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.transforms = transforms
        self.keep_difficult = keep_difficult
        self.polygon_policy = polygon_policy

        # Reusing a manifest keeps startup predictable on large datasets.
        manifest = None
        if manifest_file is not None and Path(manifest_file).exists():
            loaded_manifest = load_dataset_manifest(manifest_file)
            if _manifest_matches_request(
                loaded_manifest,
                image_root=self.image_root,
                label_root=self.label_root,
                class_names=class_names,
                keep_difficult=keep_difficult,
                polygon_policy=polygon_policy,
            ):
                manifest = loaded_manifest

        if manifest is None:
            manifest = scan_detection_dataset(
                image_root=self.image_root,
                label_root=self.label_root,
                class_names=class_names,
                keep_difficult=keep_difficult,
                polygon_policy=polygon_policy,
            )

        self.class_names = list(manifest["class_names"])
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.samples = list(manifest["included_samples"])
        self.quarantined_samples = list(manifest.get("quarantined_samples", []))
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        """Load one sample from the normalized manifest representation."""
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

        height = int(sample["height"])
        width = int(sample["width"])
        target = {
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
            "boxes": boxes,
            "labels": labels,
            "sample_id": sample["sample_id"],
            "annotation_path": sample["annotation_path"],
            "image_path": sample["image_path"],
            "annotation_format": sample["annotation_format"],
            "shape_type_counts": dict(sample.get("shape_type_counts", {})),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
