"""Small distributed helpers required by the migrated RTv4 training code."""

from __future__ import annotations

import torch.distributed


def is_dist_available_and_initialized() -> bool:
    """Return whether torch.distributed is both available and initialized."""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Return distributed world size, defaulting to 1 in single-process runs."""
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """Return distributed rank, defaulting to 0 outside distributed runs."""
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    """Return whether the current process is rank 0."""
    return get_rank() == 0


__all__ = ["get_rank", "get_world_size", "is_dist_available_and_initialized", "is_main_process"]
