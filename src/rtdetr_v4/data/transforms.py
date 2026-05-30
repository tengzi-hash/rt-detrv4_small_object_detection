"""Image and target transforms used by the migrated data pipeline.

The main goal here is not to mirror every official RT-DETR transform, but to
preserve the already-validated preprocessing behavior from the old project in a
small, readable module.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor

from ..utils.box_ops import box_xyxy_to_cxcywh


class Compose:
    """Apply transforms in order to both image and target."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def _as_tensor_size(value: Any, *, default: tuple[int, int]) -> Tensor:
    """Normalize size-like metadata to a tensor."""
    if isinstance(value, Tensor):
        return value.to(dtype=torch.long)
    if isinstance(value, Sequence) and len(value) == 2:
        return torch.tensor([int(value[0]), int(value[1])], dtype=torch.long)
    return torch.tensor([default[0], default[1]], dtype=torch.long)


def _normalize_size(size: int | Sequence[int]) -> tuple[int, int]:
    """Accept either one integer or an explicit (height, width) pair."""
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError(f"Expected size with 2 values, got: {size}")
    return int(size[0]), int(size[1])


def _normalize_pad_value(pad_value: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize one grayscale pad value or an explicit RGB tuple."""
    if isinstance(pad_value, int):
        return (pad_value, pad_value, pad_value)
    if len(pad_value) != 3:
        raise ValueError(f"Expected RGB pad_value with 3 entries, got: {pad_value}")
    return tuple(int(channel) for channel in pad_value)


def _normalize_rotation_range(rotation_degrees: float | Sequence[float]) -> tuple[float, float]:
    """Convert a symmetric degree value into a min/max range."""
    if isinstance(rotation_degrees, Sequence) and not isinstance(rotation_degrees, (str, bytes)):
        if len(rotation_degrees) != 2:
            raise ValueError(f"Expected rotation range with 2 values, got: {rotation_degrees}")
        return float(rotation_degrees[0]), float(rotation_degrees[1])

    max_degrees = float(rotation_degrees)
    return -max_degrees, max_degrees


def _parse_transform_config(image_size_or_config: int | Sequence[int] | Mapping[str, Any]) -> dict[str, Any]:
    """Support either a plain image size or a richer transform config mapping."""
    if isinstance(image_size_or_config, Mapping):
        image_size = image_size_or_config.get("image_size")
        if image_size is None:
            raise ValueError("Transform config must define 'image_size'.")
        return {
            "image_size": image_size,
            "keep_ratio": bool(image_size_or_config.get("keep_ratio", True)),
            "pad_value": image_size_or_config.get("pad_value", 114),
            "pad_position": image_size_or_config.get("pad_position", "top_left"),
            "hflip_prob": float(image_size_or_config.get("hflip_prob", 0.5)),
            "rotate_prob": float(image_size_or_config.get("rotate_prob", 0.0)),
            "rotate_degrees": image_size_or_config.get("rotate_degrees", 0.0),
            "photometric_prob": float(image_size_or_config.get("photometric_prob", 0.0)),
            "brightness": float(image_size_or_config.get("brightness", 0.0)),
            "contrast": float(image_size_or_config.get("contrast", 0.0)),
            "saturation": float(image_size_or_config.get("saturation", 0.0)),
        }

    return {
        "image_size": image_size_or_config,
        "keep_ratio": True,
        "pad_value": 114,
        "pad_position": "top_left",
        "hflip_prob": 0.5,
        "rotate_prob": 0.0,
        "rotate_degrees": 0.0,
        "photometric_prob": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
    }


def _update_geometry_metadata(
    target: dict[str, Any],
    *,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    target_height: int,
    target_width: int,
    padding: tuple[int, int, int, int],
) -> None:
    """Record resize/padding metadata needed by DETR-style post-processing."""
    pad_left, pad_top, pad_right, pad_bottom = padding
    # False means a pixel comes from the real image; True means it is padding.
    padding_mask = torch.ones((target_height, target_width), dtype=torch.bool)
    padding_mask[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = False

    target["orig_size"] = _as_tensor_size(target.get("orig_size"), default=(orig_height, orig_width))
    target["resized_size"] = torch.tensor([resized_height, resized_width], dtype=torch.long)
    target["valid_size"] = torch.tensor([resized_height, resized_width], dtype=torch.long)
    target["size"] = torch.tensor([target_height, target_width], dtype=torch.long)
    target["scale_factor"] = torch.tensor(
        [resized_height / max(orig_height, 1), resized_width / max(orig_width, 1)],
        dtype=torch.float32,
    )
    target["padding"] = torch.tensor([pad_left, pad_top, pad_right, pad_bottom], dtype=torch.long)
    target["padding_mask"] = padding_mask


class Resize:
    """Resize directly to the target shape without preserving aspect ratio."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = _normalize_size(size)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        old_width, old_height = image.size
        new_height, new_width = self.size
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= new_width / max(old_width, 1)
            boxes[:, [1, 3]] *= new_height / max(old_height, 1)
            target["boxes"] = boxes

        _update_geometry_metadata(
            target,
            orig_height=old_height,
            orig_width=old_width,
            resized_height=new_height,
            resized_width=new_width,
            target_height=new_height,
            target_width=new_width,
            padding=(0, 0, 0, 0),
        )
        return image, target


class ResizeAndPad:
    """Resize while keeping aspect ratio, then pad to the final canvas size."""

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        pad_value: int | Sequence[int] = 114,
        pad_position: str = "top_left",
    ) -> None:
        self.size = _normalize_size(size)
        self.pad_value = _normalize_pad_value(pad_value)
        if pad_position != "top_left":
            raise ValueError(
                f"Unsupported pad_position: {pad_position}. "
                "Phase 1 only supports top-left aligned padding for deterministic masks."
            )
        self.pad_position = pad_position

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        old_width, old_height = image.size
        target_height, target_width = self.size

        # DETR-style training often expects a fixed batch shape; this keeps the
        # content aspect ratio while still producing that fixed shape.
        scale = min(target_width / max(old_width, 1), target_height / max(old_height, 1))
        resized_width = max(1, min(target_width, int(round(old_width * scale))))
        resized_height = max(1, min(target_height, int(round(old_height * scale))))

        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        pad_left = 0
        pad_top = 0
        pad_right = max(0, target_width - resized_width)
        pad_bottom = max(0, target_height - resized_height)
        if pad_right or pad_bottom:
            image = ImageOps.expand(
                image,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.pad_value,
            )

        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= resized_width / max(old_width, 1)
            boxes[:, [1, 3]] *= resized_height / max(old_height, 1)
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            target["boxes"] = boxes

        _update_geometry_metadata(
            target,
            orig_height=old_height,
            orig_width=old_width,
            resized_height=resized_height,
            resized_width=resized_width,
            target_height=target_height,
            target_width=target_width,
            padding=(pad_left, pad_top, pad_right, pad_bottom),
        )
        return image, target


class RandomHorizontalFlip:
    """Flip image and boxes horizontally with the configured probability."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        if random.random() >= self.probability:
            return image, target

        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        width, _ = image.size
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            x0 = boxes[:, 0].clone()
            x1 = boxes[:, 2].clone()
            boxes[:, 0] = width - x1
            boxes[:, 2] = width - x0
            target["boxes"] = boxes
        return image, target


def _rotate_boxes_xyxy(
    boxes: Tensor,
    *,
    angle_degrees: float,
    image_width: int,
    image_height: int,
) -> Tensor:
    """Rotate xyxy boxes by rotating their four corners and re-boxing them."""
    if boxes.numel() == 0:
        return boxes

    angle_radians = math.radians(angle_degrees)
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    center_x = (image_width - 1) / 2.0
    center_y = (image_height - 1) / 2.0

    rotated_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = [float(value) for value in box.tolist()]
        corners = torch.tensor(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=torch.float32,
        )
        corners[:, 0] -= center_x
        corners[:, 1] -= center_y
        # PIL rotates in image coordinates where +y points downward.
        rotated_x = corners[:, 0] * cos_theta + corners[:, 1] * sin_theta
        rotated_y = -corners[:, 0] * sin_theta + corners[:, 1] * cos_theta
        rotated_x += center_x
        rotated_y += center_y

        rotated_boxes.append(
            torch.tensor(
                [
                    rotated_x.min().clamp(min=0.0, max=float(image_width)),
                    rotated_y.min().clamp(min=0.0, max=float(image_height)),
                    rotated_x.max().clamp(min=0.0, max=float(image_width)),
                    rotated_y.max().clamp(min=0.0, max=float(image_height)),
                ],
                dtype=torch.float32,
            )
        )

    return torch.stack(rotated_boxes, dim=0)


class RandomRotate:
    """Apply an in-place image rotation and update bounding boxes accordingly."""

    def __init__(
        self,
        probability: float = 0.0,
        degrees: float | Sequence[float] = 0.0,
        *,
        fill_value: int | Sequence[int] = 114,
    ) -> None:
        self.probability = probability
        self.min_degrees, self.max_degrees = _normalize_rotation_range(degrees)
        self.fill_value = _normalize_pad_value(fill_value)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        if self.probability <= 0 or random.random() >= self.probability:
            return image, target

        angle = random.uniform(self.min_degrees, self.max_degrees)
        if abs(angle) < 1e-6:
            return image, target

        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=self.fill_value,
        )
        if target["boxes"].numel() > 0:
            target["boxes"] = _rotate_boxes_xyxy(
                target["boxes"].clone(),
                angle_degrees=angle,
                image_width=image.width,
                image_height=image.height,
            )
        return image, target


class RandomPhotometricDistort:
    """Randomly jitter brightness/contrast/saturation on the PIL image.

    This is pixel-space only and runs before ToTensor/Normalize, so the model
    still receives the same ImageNet-normalized distribution and the pretrained
    weights stay fully compatible. Targets (boxes/labels) are untouched.
    """

    def __init__(
        self,
        probability: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
    ) -> None:
        self.probability = float(probability)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)

    def _factor(self, magnitude: float) -> float:
        return random.uniform(max(0.0, 1.0 - magnitude), 1.0 + magnitude)

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        if self.probability <= 0 or random.random() >= self.probability:
            return image, target

        ops: list[tuple[str, float]] = []
        if self.brightness > 0:
            ops.append(("Brightness", self.brightness))
        if self.contrast > 0:
            ops.append(("Contrast", self.contrast))
        if self.saturation > 0:
            ops.append(("Color", self.saturation))
        random.shuffle(ops)
        for enhancer_name, magnitude in ops:
            enhancer = getattr(ImageEnhance, enhancer_name)(image)
            image = enhancer.enhance(self._factor(magnitude))
        return image, target


class ToTensor:
    """Convert a PIL RGB image into a float tensor in CHW layout."""

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, target


class Normalize:
    """Normalize image channels with ImageNet-style statistics."""

    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Tensor, target: dict[str, Any]):
        return (image - self.mean) / self.std, target


class FormatTargetForDETR:
    """Convert pixel-space xyxy boxes into normalized cxcywh DETR targets."""

    def __call__(self, image: Tensor, target: dict[str, Any]):
        height, width = image.shape[-2:]
        boxes = target["boxes"]
        labels = target["labels"]

        if boxes.numel() > 0:
            boxes = boxes.clone()
            # Clamp first, then discard zero-area boxes created by rounding or
            # augmentation edge cases.
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
            target["boxes_xyxy"] = boxes.clone()
            scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
            boxes = box_xyxy_to_cxcywh(boxes / scale)
        else:
            target["boxes_xyxy"] = torch.zeros((0, 4), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target["boxes"] = boxes
        target["labels"] = labels
        target["size"] = torch.tensor([height, width], dtype=torch.long)
        return image, target


def build_transforms(image_size_or_config: int | Sequence[int] | Mapping[str, Any], train: bool = True) -> Compose:
    """Build the train/val transform chain used by the migrated datasets."""
    transform_config = _parse_transform_config(image_size_or_config)

    resize_transform = (
        ResizeAndPad(
            transform_config["image_size"],
            pad_value=transform_config["pad_value"],
            pad_position=transform_config["pad_position"],
        )
        if transform_config["keep_ratio"]
        else Resize(transform_config["image_size"])
    )

    transforms: list[Any] = []
    if train:
        transforms.append(RandomHorizontalFlip(transform_config["hflip_prob"]))
        transforms.append(
            RandomRotate(
                probability=transform_config["rotate_prob"],
                degrees=transform_config["rotate_degrees"],
                fill_value=transform_config["pad_value"],
            )
        )
        transforms.append(
            RandomPhotometricDistort(
                probability=transform_config["photometric_prob"],
                brightness=transform_config["brightness"],
                contrast=transform_config["contrast"],
                saturation=transform_config["saturation"],
            )
        )
    transforms.extend(
        [
            resize_transform,
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            FormatTargetForDETR(),
        ]
    )
    return Compose(transforms)
