"""General utility helpers copied from the old project for phase-1 reuse.

These functions are low-coupling primitives that are still useful even before
the official RT-DETR v4 training stack is in place.
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for more repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    """Resolve a path relative to the repository root when needed."""
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_device(configured_device: str, override_device: str | None) -> torch.device:
    """Validate the requested device string before training/evaluation starts."""
    requested_device = override_device or configured_device
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {requested_device}, but CUDA is not available in this environment."
            )
        device = torch.device(requested_device)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device {requested_device}, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        return device
    return torch.device(requested_device)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Numerically stable inverse of sigmoid."""
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    """Create N deep-copied layers, matching common transformer code patterns."""
    return nn.ModuleList(copy.deepcopy(module) for _ in range(num_layers))


def batch_index_select(source: Tensor, indices: Tensor) -> Tensor:
    """Select per-batch items from a tensor using per-batch indices."""
    gather_index = indices.unsqueeze(-1).expand(-1, -1, source.shape[-1])
    return source.gather(dim=1, index=gather_index)


def flatten_multi_scale_features(
    features: Sequence[Tensor],
) -> tuple[Tensor, list[tuple[int, int]]]:
    """Flatten multi-scale feature maps into one token sequence plus shapes."""
    flattened = []
    spatial_shapes: list[tuple[int, int]] = []
    for feature in features:
        _, _, height, width = feature.shape
        spatial_shapes.append((height, width))
        flattened.append(feature.flatten(2).transpose(1, 2))
    return torch.cat(flattened, dim=1), spatial_shapes


def build_anchors(
    spatial_shapes: Sequence[tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
    base_scale: float = 0.05,
) -> Tensor:
    """Create normalized anchor priors for each feature level."""
    anchors = []
    for level_index, (height, width) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        center_x = (grid_x + 0.5) / width
        center_y = (grid_y + 0.5) / height
        scale = base_scale * (2.0 ** level_index)
        anchor_w = torch.full_like(center_x, scale)
        anchor_h = torch.full_like(center_y, scale)
        anchors.append(
            torch.stack((center_x, center_y, anchor_w, anchor_h), dim=-1).reshape(-1, 4)
        )
    return torch.cat(anchors, dim=0)


def coordinate_to_sine_embedding(coords: Tensor, hidden_dim: int) -> Tensor:
    """Encode 1D/2D normalized coordinates with sine/cosine features."""
    if hidden_dim % 2 != 0:
        raise ValueError("hidden_dim 必须是偶数。")
    num_feats = hidden_dim // 2
    dim_t = torch.arange(num_feats, device=coords.device, dtype=coords.dtype)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_feats)
    embeds = []
    for coord in coords.unbind(-1):
        scaled = coord.unsqueeze(-1) * 2.0 * math.pi / dim_t
        embeds.append(
            torch.stack((scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=-1).flatten(-2)
        )
    return torch.cat(embeds, dim=-1)[..., :hidden_dim]


def build_2d_sine_position_embedding(
    height: int,
    width: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build a dense 2D sine positional embedding for one feature map."""
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim 必须能被 4 整除，以便构造 2D sine 位置编码。")
    y_embed, x_embed = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack((x_embed, y_embed), dim=-1).reshape(-1, 2)
    return coordinate_to_sine_embedding(coords, hidden_dim)


def box_to_sine_embedding(boxes: Tensor, hidden_dim: int) -> Tensor:
    """Encode normalized boxes with sine/cosine features per coordinate."""
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim 必须能被 4 整除，以便对 4 个 box 坐标分别编码。")
    component_dim = hidden_dim // 4
    dim_t = torch.arange(component_dim, device=boxes.device, dtype=boxes.dtype)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / component_dim)
    outputs = []
    for coord in boxes.unbind(-1):
        scaled = coord.unsqueeze(-1) * 2.0 * math.pi / dim_t
        outputs.append(
            torch.stack((scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=-1).flatten(-2)
        )
    return torch.cat(outputs, dim=-1)[..., :hidden_dim]


def is_dist_avail_and_initialized() -> bool:
    """Return whether torch.distributed is usable in the current process."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def reduce_mean(value: Tensor) -> Tensor:
    """Average a tensor across processes when distributed training is enabled."""
    if not is_dist_avail_and_initialized():
        return value
    value = value.clone()
    torch.distributed.all_reduce(value)
    value /= torch.distributed.get_world_size()
    return value
