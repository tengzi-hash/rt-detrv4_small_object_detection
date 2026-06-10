"""CLI entrypoint for building the locomotive RT-DETRv4 dataset.

The implementation lives in ``utils.dataset_build_core`` so this script stays
small while old imports such as ``from _build_dataset import load_config`` keep
working.
"""

from __future__ import annotations

import sys

from utils.dataset_build_core import *  # noqa: F401,F403
from utils.dataset_build_core import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
