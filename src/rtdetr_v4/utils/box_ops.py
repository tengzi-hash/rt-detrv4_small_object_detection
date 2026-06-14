"""Bounding-box math helpers shared by the migrated RT-DETR v4 modules."""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert boxes from center-size format to corner format."""
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack((x0, y0, x1, y1), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert boxes from corner format to center-size format."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = x1 - x0
    h = y1 - y0
    return torch.stack((cx, cy, w, h), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Compute box area while guarding against inverted coordinates."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (
        boxes[..., 3] - boxes[..., 1]
    ).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Return pairwise IoU and union for two box sets."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU used by DETR-style loss/evaluation code."""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def clip_boxes_to_image(boxes: Tensor, height: int, width: int) -> Tensor:
    """Clamp box coordinates so they stay inside image bounds."""
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0, max=width)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0, max=height)
    return boxes
