from __future__ import annotations

import torch
from torch import Tensor


def detr_collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, list[dict]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
