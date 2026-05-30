"""Project-local RTv4 model assembly with explicit builders."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch.nn as nn

from ..config import load_config, merge_dict
from ..engine.backbone import HGNetv2, PResNet
from ..engine.decoder import DFINETransformer, RTDETRTransformerv2
from ..engine.encoder import HybridEncoder
from ..engine.loss import HungarianMatcher, RTv4Criterion
from ..engine.postprocess import PostProcessor


BACKBONE_CLASSES = {
    "HGNetv2": HGNetv2,
    "PResNet": PResNet,
}
ENCODER_CLASSES = {
    "HybridEncoder": HybridEncoder,
}
DECODER_CLASSES = {
    "DFINETransformer": DFINETransformer,
    "RTDETRTransformerv2": RTDETRTransformerv2,
}
MATCHER_CLASSES = {
    "HungarianMatcher": HungarianMatcher,
}
CRITERION_CLASSES = {
    "RTv4Criterion": RTv4Criterion,
}
POSTPROCESSOR_CLASSES = {
    "PostProcessor": PostProcessor,
}


class RTv4(nn.Module):
    """Top-level detector that wires backbone, encoder, and decoder together."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        postprocessor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.postprocessor = postprocessor
        resolved_num_classes = getattr(postprocessor, "num_classes", None)
        if resolved_num_classes is None:
            resolved_num_classes = getattr(decoder, "num_classes", None)
        self.num_classes = int(resolved_num_classes) if resolved_num_classes is not None else 0

    def forward(self, x, targets=None, teacher_encoder_output=None):
        backbone_features = self.backbone(x)
        encoder_output = self.encoder(backbone_features)

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            fpn_features, student_distill_output = encoder_output
        else:
            fpn_features = encoder_output

        decoder_output = self.decoder(fpn_features, targets)

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            decoder_output["student_distill_output"] = student_distill_output
            decoder_output["teacher_encoder_output"] = teacher_encoder_output

        return decoder_output

    def post_process(self, outputs, *, image_sizes, topk: int = 300):
        if self.postprocessor is None:
            raise NotImplementedError("RTv4 does not have a postprocessor attached.")

        original_top_queries = getattr(self.postprocessor, "num_top_queries", None)
        if original_top_queries is not None:
            self.postprocessor.num_top_queries = int(topk)
        try:
            return self.postprocessor(outputs, orig_target_sizes=image_sizes)
        finally:
            if original_top_queries is not None:
                self.postprocessor.num_top_queries = original_top_queries

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        if self.postprocessor is not None and hasattr(self.postprocessor, "deploy"):
            self.postprocessor.deploy()
        return self


def _prepare_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    working_cfg = merge_dict({}, config, inplace=True)
    if overrides:
        merge_dict(working_cfg, overrides, inplace=True)
    return working_cfg


def _resolve_component_entry(
    config: dict[str, Any],
    entry: str | dict[str, Any] | None,
    *,
    label: str,
) -> tuple[str, dict[str, Any]]:
    if entry is None:
        raise ValueError(f"Missing component entry for '{label}'.")

    if isinstance(entry, str):
        name = entry
        overrides: dict[str, Any] = {}
    elif isinstance(entry, dict):
        if "type" not in entry:
            raise ValueError(f"Component entry '{label}' must define 'type'.")
        name = str(entry["type"])
        overrides = {key: value for key, value in entry.items() if key != "type"}
    else:
        raise TypeError(f"Unsupported component entry for '{label}': {type(entry).__name__}")

    base_cfg = {}
    if name in config:
        section = config[name]
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise TypeError(f"Expected config section '{name}' to be a dict, got: {type(section).__name__}")
        base_cfg = merge_dict({}, section, inplace=True)
    if overrides:
        merge_dict(base_cfg, overrides, inplace=True)
    return name, base_cfg


def _instantiate_component(
    component_name: str,
    component_classes: dict[str, type[nn.Module]],
    component_cfg: dict[str, Any],
    *,
    shared_kwargs: dict[str, Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
    label: str,
) -> nn.Module:
    component_cls = component_classes.get(component_name)
    if component_cls is None:
        supported = ", ".join(sorted(component_classes))
        raise ValueError(f"Unsupported {label}: {component_name}. Supported: {supported}")

    constructor = inspect.signature(component_cls.__init__)
    accepted_names = {
        name
        for name, parameter in constructor.parameters.items()
        if name != "self" and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    kwargs = {
        key: value
        for key, value in component_cfg.items()
        if key in accepted_names
    }

    if shared_kwargs:
        for key, value in shared_kwargs.items():
            if value is not None and key in accepted_names and key not in kwargs:
                kwargs[key] = value

    if extra_kwargs:
        for key, value in extra_kwargs.items():
            if key not in accepted_names:
                raise TypeError(f"{label} '{component_name}' does not accept argument '{key}'.")
            kwargs[key] = value

    return component_cls(**kwargs)


def _shared_values(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_classes": config.get("num_classes"),
        "use_focal_loss": config.get("use_focal_loss"),
        "eval_spatial_size": config.get("eval_spatial_size"),
    }


def _build_backbone(config: dict[str, Any], model_cfg: dict[str, Any]) -> nn.Module:
    backbone_name, backbone_cfg = _resolve_component_entry(
        config,
        model_cfg.get("backbone"),
        label="backbone",
    )
    return _instantiate_component(
        backbone_name,
        BACKBONE_CLASSES,
        backbone_cfg,
        label="backbone",
    )


def _build_encoder(config: dict[str, Any], model_cfg: dict[str, Any], shared_values: dict[str, Any]) -> nn.Module:
    encoder_name, encoder_cfg = _resolve_component_entry(
        config,
        model_cfg.get("encoder"),
        label="encoder",
    )
    return _instantiate_component(
        encoder_name,
        ENCODER_CLASSES,
        encoder_cfg,
        shared_kwargs={"eval_spatial_size": shared_values["eval_spatial_size"]},
        label="encoder",
    )


def _build_decoder(config: dict[str, Any], model_cfg: dict[str, Any], shared_values: dict[str, Any]) -> nn.Module:
    decoder_name, decoder_cfg = _resolve_component_entry(
        config,
        model_cfg.get("decoder"),
        label="decoder",
    )
    return _instantiate_component(
        decoder_name,
        DECODER_CLASSES,
        decoder_cfg,
        shared_kwargs={
            "num_classes": shared_values["num_classes"],
            "eval_spatial_size": shared_values["eval_spatial_size"],
        },
        label="decoder",
    )


def build_postprocessor_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Build the project-local postprocessor defined by one config dictionary."""
    working_cfg = _prepare_config(config, overrides=overrides)
    shared_values = _shared_values(working_cfg)
    postprocessor_name, postprocessor_cfg = _resolve_component_entry(
        working_cfg,
        working_cfg.get("postprocessor"),
        label="postprocessor",
    )
    return _instantiate_component(
        postprocessor_name,
        POSTPROCESSOR_CLASSES,
        postprocessor_cfg,
        shared_kwargs={
            "num_classes": shared_values["num_classes"],
            "use_focal_loss": shared_values["use_focal_loss"],
        },
        label="postprocessor",
    )


def build_model_from_config(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> nn.Module:
    """Build the project-local RTv4 model from one loaded config dictionary."""
    working_cfg = _prepare_config(config, overrides=overrides)
    model_entry = working_cfg.get("model")
    model_name, model_cfg = _resolve_component_entry(working_cfg, model_entry, label="model")
    if model_name != "RTv4":
        raise ValueError(f"Unsupported model: {model_name}. Supported: RTv4")

    shared_values = _shared_values(working_cfg)
    backbone = _build_backbone(working_cfg, model_cfg)
    encoder = _build_encoder(working_cfg, model_cfg, shared_values)
    decoder = _build_decoder(working_cfg, model_cfg, shared_values)

    postprocessor_entry = model_cfg.get("postprocessor", working_cfg.get("postprocessor"))
    postprocessor_name, postprocessor_cfg = _resolve_component_entry(
        working_cfg,
        postprocessor_entry,
        label="postprocessor",
    )
    postprocessor = _instantiate_component(
        postprocessor_name,
        POSTPROCESSOR_CLASSES,
        postprocessor_cfg,
        shared_kwargs={
            "num_classes": shared_values["num_classes"],
            "use_focal_loss": shared_values["use_focal_loss"],
        },
        label="postprocessor",
    )
    return RTv4(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
    )


def build_model_from_yaml(config_path: str | Path, overrides: dict[str, Any] | None = None) -> nn.Module:
    """Load one YAML config file and build the project-local RTv4 model."""
    config = load_config(str(config_path))
    return build_model_from_config(config, overrides=overrides)


def build_postprocessor_from_yaml(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Load one YAML config file and build the project-local postprocessor."""
    config = load_config(str(config_path))
    return build_postprocessor_from_config(config, overrides=overrides)


def build_criterion_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Build the project-local criterion defined by one config dictionary."""
    working_cfg = _prepare_config(config, overrides=overrides)
    shared_values = _shared_values(working_cfg)

    criterion_name, criterion_cfg = _resolve_component_entry(
        working_cfg,
        working_cfg.get("criterion"),
        label="criterion",
    )
    matcher_name, matcher_cfg = _resolve_component_entry(
        working_cfg,
        criterion_cfg.get("matcher"),
        label="matcher",
    )

    matcher = _instantiate_component(
        matcher_name,
        MATCHER_CLASSES,
        matcher_cfg,
        shared_kwargs={"use_focal_loss": shared_values["use_focal_loss"]},
        label="matcher",
    )
    return _instantiate_component(
        criterion_name,
        CRITERION_CLASSES,
        criterion_cfg,
        shared_kwargs={"num_classes": shared_values["num_classes"]},
        extra_kwargs={"matcher": matcher},
        label="criterion",
    )


def build_criterion_from_yaml(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Load one YAML config file and build the project-local criterion."""
    config = load_config(str(config_path))
    return build_criterion_from_config(config, overrides=overrides)


__all__ = [
    "RTv4",
    "build_criterion_from_config",
    "build_criterion_from_yaml",
    "build_model_from_config",
    "build_model_from_yaml",
    "build_postprocessor_from_config",
    "build_postprocessor_from_yaml",
]
