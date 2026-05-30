"""Checkpoint loading helpers shared by local entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract the tensor-only state dict from several common checkpoint shapes."""
    if isinstance(payload, dict):
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
        for key in ("state_dict", "model_state_dict", "model", "ema", "module"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return unwrap_state_dict(candidate)
    raise ValueError("Could not locate a tensor-only state_dict inside the checkpoint payload.")


def strip_common_prefixes(state_dict: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    """Remove wrapper prefixes when doing non-strict loads across entrypoints."""
    cleaned = dict(state_dict)
    target_keys = set(model.state_dict().keys())
    prefixes = ("module.", "model.", "student.")

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if not cleaned:
                continue
            prefixed = sum(1 for key in cleaned if key.startswith(prefix))
            if prefixed == 0:
                continue
            stripped = {
                key[len(prefix) :] if key.startswith(prefix) else key: value
                for key, value in cleaned.items()
            }
            overlap_before = len(target_keys.intersection(cleaned.keys()))
            overlap_after = len(target_keys.intersection(stripped.keys()))
            if overlap_after >= overlap_before:
                cleaned = stripped
                changed = True
    return cleaned


def filter_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    """Keep only current-model keys whose tensor shapes are compatible."""
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, value in state_dict.items():
        target_value = target_state.get(key)
        if target_value is None:
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            skipped_mismatched.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        compatible[key] = value

    return compatible, skipped_mismatched


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool,
) -> tuple[list[str], list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    """Load a checkpoint into one model while tolerating expected head mismatches."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = strip_common_prefixes(unwrap_state_dict(payload), model)
    state_dict, skipped_mismatched = filter_compatible_state_dict(model, raw_state_dict)
    overlap = set(model.state_dict()).intersection(state_dict)
    if not overlap:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not share parameter names with the current model."
        )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    return list(missing_keys), list(unexpected_keys), skipped_mismatched


__all__ = [
    "filter_compatible_state_dict",
    "load_model_weights",
    "strip_common_prefixes",
    "unwrap_state_dict",
]
