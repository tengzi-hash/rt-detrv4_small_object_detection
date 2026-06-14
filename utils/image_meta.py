from __future__ import annotations

from pathlib import Path


def image_read_error(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return str(exc)
    return None


def image_size(path: Path) -> tuple[int, int]:
    from utils import label_convert

    width, height, _ = label_convert.image_size(path)
    return width, height
