from __future__ import annotations

from typing import Protocol


class BoxLike(Protocol):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def box_area(box: BoxLike) -> int:
    return max(0, box.xmax - box.xmin) * max(0, box.ymax - box.ymin)


def iou(a: BoxLike, b: BoxLike) -> float:
    ix0, iy0 = max(a.xmin, b.xmin), max(a.ymin, b.ymin)
    ix1, iy1 = min(a.xmax, b.xmax), min(a.ymax, b.ymax)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    return inter / max(box_area(a) + box_area(b) - inter, 1)


def coverage_ratio(inner: BoxLike, outer: BoxLike) -> float:
    ix0, iy0 = max(inner.xmin, outer.xmin), max(inner.ymin, outer.ymin)
    ix1, iy1 = min(inner.xmax, outer.xmax), min(inner.ymax, outer.ymax)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / max(box_area(inner), 1)
