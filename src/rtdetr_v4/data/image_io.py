"""Small image helpers shared by the migrated datasets.

These wrappers centralize image loading so every dataset benefits from the same
EXIF orientation handling and RGB conversion rules.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load an image as RGB after normalizing EXIF orientation."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def load_image_size(path: str | Path) -> tuple[int, int]:
    """Read image size without duplicating orientation handling logic elsewhere."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size
