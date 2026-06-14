from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BatchConfig:
    name: str
    path: Path
    class_remap: dict[str, str]
    drop_classes: set[str]
    rules: dict[str, Any]


@dataclass(frozen=True)
class Box:
    cls: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def signature(self) -> tuple[str, int, int, int, int]:
        return (self.cls, self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class RawPair:
    label_path: Path
    image_path: Path
    batch: BatchConfig


@dataclass
class Sample:
    sample_id: str
    output_stem: str
    image_path: Path
    label_path: Path
    image_hash: str
    boxes: list[Box]
    width: int
    height: int
    batch_name: str
    notes: list[str] = field(default_factory=list)

    @property
    def label_signature(self) -> tuple[tuple[str, int, int, int, int], ...]:
        return tuple(sorted(box.signature() for box in self.boxes))

    @property
    def classes(self) -> list[str]:
        return sorted({box.cls for box in self.boxes})


@dataclass
class BuildIssue:
    label: str
    issue: str
    detail: str

    def as_row(self) -> dict[str, str]:
        return {"label": self.label, "issue": self.issue, "detail": self.detail}


@dataclass
class ScanResult:
    pairs: list[RawPair]
    unlabel_images: list[Path]
    issues: list[BuildIssue]
    raw_images: int
    raw_labels: int


@dataclass
class ParseResult:
    samples: list[Sample]
    unlabel_images: list[Path]
    issues: list[BuildIssue]
    stats: Counter


@dataclass
class DedupResult:
    samples: list[Sample]
    conflicts: list[list[Sample]]
    issues: list[BuildIssue]
    stats: Counter


@dataclass
class PolicyRule:
    raw_class: str
    action: str
    final_class: str


@dataclass
class PolicyResult:
    samples: list[Sample]
    hold_samples: list[Sample]
    unknown_samples: list[Sample]
    dropped_samples: list[Sample]
    issues: list[BuildIssue]
    stats: Counter
