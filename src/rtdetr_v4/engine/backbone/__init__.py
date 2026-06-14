"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# Mainline backbone implementations live at the project engine layer so the
# rest of the codebase does not need a second runtime copy elsewhere.
# Keep the exported backbone surface intentionally small.
from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet
from .hgnetv2 import HGNetv2
