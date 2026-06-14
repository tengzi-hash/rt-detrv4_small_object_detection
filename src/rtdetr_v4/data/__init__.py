"""Dataset entrypoints migrated from the v3 project.

This package intentionally carries only the low-coupling data layer needed for
phase-1 migration:

- dataset discovery/parsing
- transform construction
- collate helpers

It does not include model-specific training code from the old project.
"""

from __future__ import annotations

from .baseline_dataset import BaselineDetectionDataset
from .catalog import load_dataset_manifest, save_dataset_manifest, scan_detection_dataset
from .collate import detr_collate_fn
from .mixed_detection import MixedDetectionDataset
from .transforms import build_transforms
from .voc_xml import VOCXMLDetectionDataset


SUPPORTED_DATASET_TYPES = {"voc_xml", "mixed", "baseline"}


def get_required_dataset_split_keys(dataset_config: dict, split: str) -> tuple[str, ...]:
    """Return the config keys that must exist for the requested split."""
    dataset_type = dataset_config.get("type", "mixed")
    base_keys = (f"{split}_images", f"{split}_annotations")
    if dataset_type == "baseline":
        return (*base_keys, f"{split}_split_csv")
    if dataset_type in SUPPORTED_DATASET_TYPES:
        return base_keys
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def dataset_has_split(dataset_config: dict, split: str) -> bool:
    """Check whether a split is fully configured before building a dataset."""
    required_keys = get_required_dataset_split_keys(dataset_config, split)
    return all(key in dataset_config and dataset_config[key] is not None for key in required_keys)


def validate_dataset_split_config(dataset_config: dict, split: str) -> None:
    """Fail early with a readable message when a split is only partially defined."""
    required_keys = get_required_dataset_split_keys(dataset_config, split)
    missing_keys = [key for key in required_keys if key not in dataset_config or dataset_config[key] is None]
    if missing_keys:
        dataset_type = dataset_config.get("type", "mixed")
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"Dataset type '{dataset_type}' requires the following keys for split '{split}': {missing}."
        )


def build_detection_dataset(dataset_config: dict, transforms, train: bool = True):
    """Build one of the migrated detection datasets."""
    split = "train" if train else "val"
    dataset_type = dataset_config.get("type", "mixed")
    validate_dataset_split_config(dataset_config, split)

    if dataset_type == "voc_xml":
        return VOCXMLDetectionDataset(
            image_root=dataset_config[f"{split}_images"],
            label_root=dataset_config[f"{split}_annotations"],
            transforms=transforms,
            class_names=dataset_config.get("classes"),
            keep_difficult=dataset_config.get("keep_difficult", False),
        )

    if dataset_type == "mixed":
        manifest_key = f"{split}_manifest"
        return MixedDetectionDataset(
            image_root=dataset_config[f"{split}_images"],
            label_root=dataset_config[f"{split}_annotations"],
            transforms=transforms,
            class_names=dataset_config.get("classes"),
            keep_difficult=dataset_config.get("keep_difficult", False),
            polygon_policy=dataset_config.get("polygon_policy", "exclude_sample"),
            manifest_file=dataset_config.get(manifest_key),
        )

    if dataset_type == "baseline":
        return BaselineDetectionDataset(
            image_root=dataset_config[f"{split}_images"],
            label_root=dataset_config[f"{split}_annotations"],
            split_csv=dataset_config[f"{split}_split_csv"],
            transforms=transforms,
            class_names=dataset_config.get("classes"),
            keep_difficult=dataset_config.get("keep_difficult", False),
            polygon_policy=dataset_config.get("polygon_policy", "exclude_sample"),
        )

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


__all__ = [
    "BaselineDetectionDataset",
    "MixedDetectionDataset",
    "VOCXMLDetectionDataset",
    "build_detection_dataset",
    "build_transforms",
    "dataset_has_split",
    "detr_collate_fn",
    "get_required_dataset_split_keys",
    "load_dataset_manifest",
    "save_dataset_manifest",
    "scan_detection_dataset",
    "validate_dataset_split_config",
]
