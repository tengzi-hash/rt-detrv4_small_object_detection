"""Project-local RT-DETRv4 package entrypoints.

The active local package for this repository is ``src/rtdetr_v4``.
Imports like ``from rtdetr_v4...`` resolve to this directory after the
entrypoints add ``src`` to ``sys.path``.
"""

from .builder import (
    build_criterion,
    build_model,
    build_postprocessor,
    build_teacher,
)

__all__ = [
    "build_criterion",
    "build_model",
    "build_postprocessor",
    "build_teacher",
]
