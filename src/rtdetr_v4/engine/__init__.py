"""Project engine package.

This layer now owns the mainline backbone/encoder/decoder/loss/postprocess
implementations as well as shared runtime helpers such as evaluation.
"""

from . import backbone
from . import encoder
from . import decoder
from . import loss
from . import postprocess
from .evaluator import evaluate_detection

__all__ = ["backbone", "encoder", "decoder", "loss", "postprocess", "evaluate_detection"]
