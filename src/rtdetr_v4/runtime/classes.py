"""Dataset class-name resolution helpers for local entrypoints."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CLASS_NAME_COLUMNS = ("label_name", "class_name", "name")
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def _load_class_names_from_summary_csv(csv_path: str | Path) -> list[str]:
    """Read ordered class names from one class-summary CSV export."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Class summary CSV not found: {path}")

    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"Class summary CSV is missing a header row: {path}")

                normalized_fieldnames = {
                    str(fieldname).strip(): fieldname
                    for fieldname in reader.fieldnames
                    if fieldname is not None
                }
                column_name = None
                for candidate in CLASS_NAME_COLUMNS:
                    if candidate in normalized_fieldnames:
                        column_name = normalized_fieldnames[candidate]
                        break
                if column_name is None:
                    expected = ", ".join(CLASS_NAME_COLUMNS)
                    raise ValueError(
                        f"Class summary CSV must contain one of: {expected}. "
                        f"Got: {sorted(normalized_fieldnames)}"
                    )

                class_names: list[str] = []
                seen: set[str] = set()
                for row in reader:
                    if row is None:
                        continue
                    raw_value = row.get(column_name)
                    if raw_value is None:
                        continue
                    class_name = str(raw_value).strip()
                    if not class_name:
                        continue
                    if class_name in seen:
                        raise ValueError(f"Duplicate class '{class_name}' in class summary CSV: {path}")
                    seen.add(class_name)
                    class_names.append(class_name)

                if not class_names:
                    raise ValueError(f"No class names were found in class summary CSV: {path}")
                return class_names
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise ValueError(f"Could not decode class summary CSV with supported encodings: {path}") from last_error
    raise RuntimeError(f"Unexpected failure while reading class summary CSV: {path}")


def resolve_dataset_classes(config: dict[str, Any]) -> dict[str, Any]:
    """Populate dataset.classes from dataset.class_summary_csv when configured."""
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return config

    class_summary_csv = dataset_cfg.get("class_summary_csv")
    if not class_summary_csv:
        return config

    class_names = _load_class_names_from_summary_csv(class_summary_csv)
    existing_classes = dataset_cfg.get("classes")
    if existing_classes is not None and list(existing_classes) != class_names:
        raise ValueError(
            "dataset.classes and dataset.class_summary_csv disagree. "
            "Keep only one source of truth or make them identical."
        )

    dataset_cfg["classes"] = class_names
    config["num_classes"] = len(class_names)
    return config


__all__ = ["resolve_dataset_classes"]
