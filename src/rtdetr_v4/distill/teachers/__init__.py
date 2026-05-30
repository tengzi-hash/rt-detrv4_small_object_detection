"""Teacher networks used only when distillation is enabled."""

from .dinov2 import DINOv2TeacherModel
from .dinov3 import DINOv3TeacherModel

__all__ = ["DINOv2TeacherModel", "DINOv3TeacherModel"]
