"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

from ...utils.misc import resolve_project_path

_logger = logging.getLogger(__name__)
DEFAULT_DINOV3_REPO = "vendor/dinov3"


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "file"}


def _resolve_repo_path(repo_path: str | Path | None) -> Path:
    resolved = resolve_project_path(repo_path or DEFAULT_DINOV3_REPO)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            f"DINOv3 repository path does not exist: {resolved}. "
            f"Clone the official repo into '{DEFAULT_DINOV3_REPO}' or set teacher_model.dinov3_repo_path."
        )

    if not (resolved / "hubconf.py").exists() or not (resolved / "dinov3").is_dir():
        raise FileNotFoundError(
            f"DINOv3 repository path is invalid: {resolved}. Expected 'hubconf.py' and the 'dinov3/' package."
        )
    return resolved


def _resolve_weights_path(weights_path: str | Path | None, *, pretrained: bool) -> str | None:
    if not pretrained:
        return None
    if not weights_path:
        raise ValueError(
            "DINOv3 teacher is configured with pretrained=True but no dinov3_weights_path was provided."
        )
    if isinstance(weights_path, str) and _is_url(weights_path):
        return weights_path

    resolved = resolve_project_path(weights_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            f"DINOv3 weights path does not exist: {resolved}. "
            "Provide a local checkpoint path or a supported URL."
        )
    return str(resolved)


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _load_dinov3_backbone(
    *,
    repo_path: Path,
    model_type: str,
    weights: str | None,
    pretrained: bool,
) -> nn.Module:
    with _prepend_sys_path(repo_path):
        backbones = importlib.import_module("dinov3.hub.backbones")
        factory = getattr(backbones, model_type, None)
        if factory is None:
            supported = ", ".join(sorted(name for name in dir(backbones) if name.startswith("dinov3_")))
            raise ValueError(
                f"Unsupported DINOv3 model type '{model_type}'. Supported local backbones: {supported}"
            )

        kwargs = {"pretrained": bool(pretrained)}
        if weights is not None:
            kwargs["weights"] = weights
        model = factory(**kwargs)
    return model


class DINOv3TeacherModel(nn.Module):
    """Training-only DINOv3 teacher kept outside the student model package."""

    def __init__(self,
                 dinov3_repo_path: str | None = None,
                 dinov3_weights_path: str | None = None,
                 dinov3_model_type: str = "dinov3_vitb16",
                 patch_size: int = 16,
                 pretrained: bool = True,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        resolved_repo_path = _resolve_repo_path(dinov3_repo_path)
        resolved_weights_path = _resolve_weights_path(dinov3_weights_path, pretrained=pretrained)

        self.dinov3_repo_path = str(resolved_repo_path)
        self.dinov3_weights_path = resolved_weights_path
        self.dinov3_model_type = dinov3_model_type
        self.patch_size = patch_size
        self.pretrained = bool(pretrained)

        _logger.info("[Teacher Model] Loading DINOv3 backbone from vendored local repo.")
        _logger.info("[Teacher Model] DINOv3 repo path: %s", self.dinov3_repo_path)
        _logger.info("[Teacher Model] DINOv3 model type: %s", self.dinov3_model_type)
        _logger.info("[Teacher Model] DINOv3 pretrained: %s", self.pretrained)
        _logger.info("[Teacher Model] DINOv3 weights path: %s", self.dinov3_weights_path)

        try:
            self.model = _load_dinov3_backbone(
                repo_path=resolved_repo_path,
                model_type=dinov3_model_type,
                weights=resolved_weights_path,
                pretrained=self.pretrained,
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            _logger.info("[Teacher Model] Successfully loaded DINOv3 teacher from local repo.")
            self.teacher_feature_dim = self.model.embed_dim

        except Exception as e:
            _logger.error(f"[Teacher Model] Failed to load DINOv3: {e}")
            raise

        self.normalize_transform = transforms.Normalize(mean=mean, std=std)
        self.avgpool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)

        _logger.info(f"[Teacher Model] DINOv3 initialized. Feature dimension: {self.teacher_feature_dim}.")
        _logger.info(
            f"[Teacher Model] Teacher model is configured to output features at a resolution that is 2x2 of the student's highest-level features after 2x downsampling.")

    def forward(self, images: torch.Tensor):
        processed_images = self.avgpool_2x2(self.normalize_transform(images))

        with torch.no_grad():
            dinov3_output_dict = self.model(processed_images, is_training=True, masks=None)
            patch_tokens = dinov3_output_dict["x_norm_patchtokens"]

            if patch_tokens.dim() != 3:
                _logger.error(
                    f"[Teacher Model] Expected 3D patch tokens, but got {patch_tokens.dim()}D tensor. Shape: {patch_tokens.shape}")
                raise ValueError("DINOv3 model's output 'x_norm_patchtokens' is not in expected 3D format.")

            B, N_patches, C_teacher = patch_tokens.shape

            H_patches_out = W_patches_out = int(N_patches ** 0.5)
            if H_patches_out * W_patches_out != N_patches:
                _logger.error(
                    f"[Teacher Model] Number of patches {N_patches} is not a perfect square for spatial reshape. Input image size: {images.shape[2:]}. Patch size: {self.patch_size}.")
                raise ValueError(
                    f"Number of patches {N_patches} is not a perfect square, cannot reshape to HxW. Check DINOv3 model output or input image size vs patch_size.")

            teacher_feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C_teacher, H_patches_out, W_patches_out)

            _logger.debug(
                "[Teacher Model] Spatial size: %s", teacher_feature_map.shape[2:])

            return teacher_feature_map.detach()
