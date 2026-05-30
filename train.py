from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
# The active local package lives under src/rtdetr_v4.
# This is the project mainline package, not the removed legacy compatibility tree.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtdetr_v4.config import load_config, merge_dict
from rtdetr_v4.data import (
    build_detection_dataset,
    build_transforms,
    dataset_has_split,
    detr_collate_fn,
)
from rtdetr_v4.distill import build_teacher_from_config
from rtdetr_v4.engine import evaluate_detection
from rtdetr_v4.engine.evaluator import compute_total_loss, move_targets_to_device
from rtdetr_v4.models import build_criterion_from_config, build_model_from_config
from rtdetr_v4.runtime import load_model_weights, resolve_config_paths, resolve_dataset_classes
from rtdetr_v4.utils.misc import resolve_device, resolve_project_path, set_seed
from rtdetr_v4.utils.monitoring import load_metrics_history, write_monitoring_artifacts


LOGGER = logging.getLogger("train")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RT-DETRv4 training entrypoint.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/colab_bolt.yml",
        help="Training config path relative to the repository root.",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        help="Optional student checkpoint override used before training starts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional top-level output directory override.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override for both train and val.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="How many train steps to skip between console logs.",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=0.0,
        help="Gradient clipping max norm. Set 0 to disable.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs when a val split exists.",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients for N mini-batches before each optimizer step.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP.")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def normalize_stage_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_stages = config.get("stages")
    if isinstance(raw_stages, list) and raw_stages:
        return [copy.deepcopy(stage) for stage in raw_stages]

    train_cfg = config.get("train")
    if isinstance(train_cfg, dict):
        stage = copy.deepcopy(train_cfg)
        stage.setdefault("name", "train")
        stage.setdefault("output_dir", config.get("output_dir"))
        stage.setdefault("config_overrides", {})
        return [stage]

    raise ValueError(
        "Training config must define either a top-level 'stages' list or a top-level 'train' block."
    )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def build_class_balanced_sampler(
    dataset,
    *,
    threshold: float,
    generator: torch.Generator,
) -> WeightedRandomSampler | None:
    """Repeat-factor (LVIS-style) image sampler that up-weights rare-class images.

    Per-image weight = max class-repeat-factor over the classes present in the
    image, where repeat_factor(c) = max(1, sqrt(threshold / image_freq(c))).
    This only changes sampling order/frequency; it does not touch the model or
    the pretrained-weight loading path.
    """
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None

    num_images = len(samples)
    class_image_count: dict[int, int] = defaultdict(int)
    image_classes: list[set[int]] = []
    for sample in samples:
        classes = {int(label) for label in sample.get("labels", [])}
        image_classes.append(classes)
        for class_index in classes:
            class_image_count[class_index] += 1

    if not class_image_count:
        return None

    class_repeat = {
        class_index: max(1.0, math.sqrt(threshold / (count / num_images)))
        for class_index, count in class_image_count.items()
    }
    weights = [
        max((class_repeat[c] for c in classes), default=1.0)
        for classes in image_classes
    ]
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights_tensor,
        num_samples=num_images,
        replacement=True,
        generator=generator,
    )


def build_data_loader(
    dataset,
    *,
    loader_cfg: dict[str, Any],
    seed: int,
    num_workers_override: int | None,
    train: bool,
) -> DataLoader:
    num_workers = int(loader_cfg.get("num_workers", 0))
    if num_workers_override is not None:
        num_workers = int(num_workers_override)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    use_shuffle = bool(loader_cfg.get("shuffle", train))
    if train and bool(loader_cfg.get("class_balanced", False)):
        balance_threshold = float(loader_cfg.get("balance_threshold", 0.1))
        sampler = build_class_balanced_sampler(
            dataset,
            threshold=balance_threshold,
            generator=generator,
        )
        if sampler is not None:
            use_shuffle = False  # DataLoader forbids shuffle together with a sampler
            weights = sampler.weights
            LOGGER.info(
                "Class-balanced sampling enabled (threshold=%s, image weight min/max/mean=%.2f/%.2f/%.2f).",
                balance_threshold,
                float(weights.min()),
                float(weights.max()),
                float(weights.mean()),
            )

    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=use_shuffle,
        sampler=sampler,
        drop_last=bool(loader_cfg.get("drop_last", train)),
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        persistent_workers=bool(num_workers > 0),
        collate_fn=detr_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_datasets_and_loaders(
    config: dict[str, Any],
    *,
    seed: int,
    num_workers_override: int | None,
) -> tuple[Any, DataLoader, Any | None, DataLoader | None]:
    dataset_cfg = copy.deepcopy(config["dataset"])
    train_transforms = build_transforms(config["train_transforms"], train=True)
    train_dataset = build_detection_dataset(dataset_cfg, train_transforms, train=True)

    resolved_classes = list(getattr(train_dataset, "class_names", dataset_cfg.get("classes") or []))
    if resolved_classes:
        dataset_cfg["classes"] = resolved_classes
        config["dataset"]["classes"] = resolved_classes

    val_dataset = None
    if dataset_has_split(dataset_cfg, "val"):
        val_transforms = build_transforms(config["val_transforms"], train=False)
        val_dataset = build_detection_dataset(dataset_cfg, val_transforms, train=False)
        val_classes = list(getattr(val_dataset, "class_names", []))
        if resolved_classes and val_classes and val_classes != resolved_classes:
            raise ValueError(
                "Validation dataset class order does not match training dataset class order. "
                "Set dataset.classes explicitly to lock the label mapping."
            )

    train_loader = build_data_loader(
        train_dataset,
        loader_cfg=config["train_dataloader"],
        seed=seed,
        num_workers_override=num_workers_override,
        train=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_data_loader(
            val_dataset,
            loader_cfg=config["val_dataloader"],
            seed=seed + 1,
            num_workers_override=num_workers_override,
            train=False,
        )

    return train_dataset, train_loader, val_dataset, val_loader


def synchronize_num_classes(config: dict[str, Any], train_dataset, val_dataset) -> None:
    train_num_classes = int(getattr(train_dataset, "num_classes", 0))
    if train_num_classes <= 0:
        raise ValueError("Training dataset resolved zero classes; cannot build the detector.")

    configured_num_classes = config.get("num_classes")
    if configured_num_classes is not None and int(configured_num_classes) != train_num_classes:
        LOGGER.warning(
            "Config num_classes=%s does not match the dataset (%s). The dataset value will be used.",
            configured_num_classes,
            train_num_classes,
        )
    config["num_classes"] = train_num_classes

    if val_dataset is not None:
        val_num_classes = int(getattr(val_dataset, "num_classes", train_num_classes))
        if val_num_classes != train_num_classes:
            raise ValueError(
                f"Training dataset has {train_num_classes} classes but validation dataset has {val_num_classes}."
            )


def build_model_bundle(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, nn.Module | None]:
    model = build_model_from_config(config)
    model.to(device)

    teacher = None
    if config.get("teacher_model") is not None:
        teacher = build_teacher_from_config(config)
        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

    return model, teacher


def build_stage_runtime_config(base_config: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    runtime_config = copy.deepcopy(base_config)
    stage_overrides = stage.get("config_overrides") or {}
    if stage_overrides:
        merge_dict(runtime_config, stage_overrides, inplace=True)
    return resolve_config_paths(runtime_config, output_dir_override=None)


def resolve_stage_teacher(stage_runtime_config: dict[str, Any], teacher: nn.Module | None) -> nn.Module | None:
    teacher_config = stage_runtime_config.get("teacher_model")
    if teacher_config is None:
        return None
    return teacher


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def reset_model_trainability(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.dtype.is_floating_point or parameter.dtype.is_complex:
            parameter.requires_grad = True


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dtype.is_floating_point or parameter.dtype.is_complex:
            parameter.requires_grad = False


def apply_backbone_freeze_strategy(model: nn.Module, backbone_cfg: dict[str, Any]) -> dict[str, Any]:
    reset_model_trainability(model)

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return {"freeze_at": -1, "freeze_norm": False, "freeze_stem_only": False}

    freeze_at = int(backbone_cfg.get("freeze_at", -1))
    freeze_norm = bool(backbone_cfg.get("freeze_norm", False))
    freeze_stem_only = bool(backbone_cfg.get("freeze_stem_only", True))

    if freeze_at >= 0:
        if hasattr(backbone, "stem"):
            freeze_module(backbone.stem)
            if not freeze_stem_only and hasattr(backbone, "stages"):
                for stage_index in range(min(freeze_at + 1, len(backbone.stages))):
                    freeze_module(backbone.stages[stage_index])
        elif hasattr(backbone, "conv1"):
            freeze_module(backbone.conv1)
            if hasattr(backbone, "res_layers"):
                for stage_index in range(min(freeze_at, len(backbone.res_layers))):
                    freeze_module(backbone.res_layers[stage_index])

    if freeze_norm:
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    total_params, trainable_params = count_parameters(model)
    LOGGER.info(
        "Applied freeze strategy: freeze_at=%s freeze_stem_only=%s freeze_norm=%s | trainable=%s / total=%s",
        freeze_at,
        freeze_stem_only,
        freeze_norm,
        trainable_params,
        total_params,
    )
    return {
        "freeze_at": freeze_at,
        "freeze_norm": freeze_norm,
        "freeze_stem_only": freeze_stem_only,
    }


def enforce_frozen_norm_runtime(model: nn.Module, stage_runtime: dict[str, Any]) -> None:
    if not stage_runtime.get("freeze_norm", False):
        return
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_optimizer(model: nn.Module, optimizer_cfg: dict[str, Any]) -> Optimizer:
    optimizer_type = str(optimizer_cfg.get("type", "AdamW")).lower()
    lr = float(optimizer_cfg["lr"])
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if decay_params:
        parameter_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        parameter_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if not parameter_groups:
        raise RuntimeError("No trainable parameters remain after applying the freeze strategy.")

    if optimizer_type == "adamw":
        betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(parameter_groups, lr=lr, betas=betas)
    if optimizer_type == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return torch.optim.SGD(parameter_groups, lr=lr, momentum=momentum)
    raise ValueError(f"Unsupported optimizer type: {optimizer_cfg.get('type')}")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: dict[str, Any] | None,
    *,
    total_steps: int,
) -> LambdaLR | None:
    if not scheduler_cfg:
        return None

    scheduler_type = str(scheduler_cfg.get("type", "constant")).lower()
    total_steps = max(int(total_steps), 1)

    if scheduler_type in {"constant", "none"}:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if scheduler_type == "warmup":
        warmup_iter = max(int(scheduler_cfg.get("warmup_iter", 0)), 1)
        start_factor = float(scheduler_cfg.get("start_factor", 0.1))

        def lr_lambda(step: int) -> float:
            if step >= warmup_iter:
                return 1.0
            progress = float(step + 1) / float(warmup_iter)
            return start_factor + (1.0 - start_factor) * progress

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_type == "cosine":
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

        def lr_lambda(step: int) -> float:
            progress = min(float(step + 1) / float(total_steps), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_type == "warmup_cosine":
        warmup_iter = max(int(scheduler_cfg.get("warmup_iter", 0)), 0)
        start_factor = float(scheduler_cfg.get("start_factor", 0.1))
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
        # Cosine decays over the steps remaining after warmup so the full
        # schedule still spans total_steps end to end.
        decay_steps = max(total_steps - warmup_iter, 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_iter:
                progress = float(step + 1) / float(max(warmup_iter, 1))
                return start_factor + (1.0 - start_factor) * progress
            progress = min(float(step - warmup_iter) / float(decay_steps), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler type: {scheduler_cfg.get('type')}")


def current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_autocast_context(device: torch.device, *, amp_enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", enabled=amp_enabled)


def build_grad_scaler(*, amp_enabled: bool):
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_namespace, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


class ModelEMA:
    """Exponential moving average of model weights.

    The EMA shadow is a deepcopy of the model taken AFTER all pretrained-weight
    loading is done, so this class never participates in checkpoint loading. It
    only reads live weights during training and is used for validation/saving.
    """

    def __init__(self, model: nn.Module, *, decay: float = 0.9999, warmup: int = 2000) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)
        self.warmup = max(int(warmup), 1)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Ramp the decay in during warmup so early noisy weights fade quickly.
        decay = self.decay * (1.0 - math.exp(-self.updates / self.warmup))
        model_state = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = model_state[key]
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)


def collect_unique_trainable_parameters(*parameter_sources) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter_source in parameter_sources:
        if parameter_source is None:
            continue
        for parameter in parameter_source:
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def build_gam_gradient_monitor(model: nn.Module, criterion: nn.Module) -> dict[str, list[nn.Parameter]] | None:
    if not hasattr(criterion, "has_adaptive_distillation") or not criterion.has_adaptive_distillation():
        return None
    if "distill" not in getattr(criterion, "losses", []):
        return None

    backbone = getattr(model, "backbone", None)
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    if backbone is None or encoder is None or decoder is None:
        return None

    encoder_groups = {}
    if hasattr(encoder, "get_gam_parameter_groups"):
        encoder_groups = encoder.get_gam_parameter_groups()

    gradient_monitor = {
        "backbone": collect_unique_trainable_parameters(backbone.parameters()),
        "aifi": collect_unique_trainable_parameters(encoder_groups.get("aifi", [])),
        "ccff": collect_unique_trainable_parameters(encoder_groups.get("ccff", [])),
        "decoder": collect_unique_trainable_parameters(decoder.parameters()),
    }
    if not any(gradient_monitor.values()):
        return None
    return gradient_monitor


def compute_parameter_grad_norm(parameters: list[nn.Parameter]) -> float:
    total_squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        gradient = gradient.float()
        total_squared_norm += float(torch.sum(gradient * gradient).item())
    return math.sqrt(total_squared_norm)


def compute_gam_gradient_statistics(gradient_monitor: dict[str, list[nn.Parameter]]) -> dict[str, Any]:
    module_norms = {
        module_name: compute_parameter_grad_norm(parameters)
        for module_name, parameters in gradient_monitor.items()
    }
    total_norm = sum(module_norms.values())
    aifi_ratio = module_norms.get("aifi", 0.0) / max(total_norm, 1e-12)
    return {
        "module_norms": module_norms,
        "total_norm": float(total_norm),
        "aifi_ratio": float(aifi_ratio),
    }


def get_gam_target_band(criterion: nn.Module) -> tuple[float, float] | None:
    adaptive_params = getattr(criterion, "distill_adaptive_params", None)
    if not isinstance(adaptive_params, dict) or not adaptive_params.get("enabled", False):
        return None

    rho = float(adaptive_params.get("rho", 0.0))
    delta = float(adaptive_params.get("delta", 0.0))
    lower = max(rho - delta, 0.0)
    upper = min(rho + delta, 1.0)
    return float(lower), float(upper)


def denormalize_for_teacher(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGE_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def append_metrics_record(metrics_log_path: Path, record: dict[str, Any], metrics_history: list[dict[str, Any]]) -> None:
    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics_history.append(record)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_checkpoint(
    *,
    path: Path,
    config_path: Path,
    stage_index: int,
    stage_name: str,
    stage_epoch: int,
    global_epoch: int,
    best_map50: float,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler,
    metrics: dict[str, Any],
    criterion_distill_state: dict[str, Any] | None = None,
    ema_state_dict: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # When EMA is active its averaged weights are the deliverable, so they become
    # the primary model_state_dict (inference loads this key by default). The live
    # weights are kept under raw_model_state_dict for exact resume if needed.
    if ema_state_dict is not None:
        primary_state_dict = ema_state_dict
        raw_state_dict = model.state_dict()
    else:
        primary_state_dict = model.state_dict()
        raw_state_dict = None
    torch.save(
        {
            "config_path": str(config_path),
            "stage_index": int(stage_index),
            "stage_name": stage_name,
            "stage_epoch": int(stage_epoch),
            "global_epoch": int(global_epoch),
            "best_map50": float(best_map50),
            "model_state_dict": primary_state_dict,
            "raw_model_state_dict": raw_state_dict,
            "ema_state_dict": ema_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "metrics": metrics,
            "criterion_distill_state": criterion_distill_state,
        },
        path,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    criterion: nn.Module,
    teacher: nn.Module | None,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    stage_name: str,
    global_epoch: int,
    amp_enabled: bool,
    log_interval: int,
    clip_grad_norm: float,
    accum_steps: int,
    stage_runtime: dict[str, Any],
    gradient_monitor: dict[str, list[nn.Parameter]] | None,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train()
    enforce_frozen_norm_runtime(model, stage_runtime)
    criterion.train()
    if teacher is not None:
        teacher.eval()

    running_metrics: dict[str, float] = defaultdict(float)
    num_steps = 0
    step_start_time = time.time()
    distill_weight_used = None
    if hasattr(criterion, "get_current_distill_weight"):
        distill_weight_used = float(criterion.get_current_distill_weight())
    elif "loss_distill" in getattr(criterion, "weight_dict", {}):
        distill_weight_used = float(criterion.weight_dict.get("loss_distill", 0.0))

    accum_steps = max(int(accum_steps), 1)
    total_batches = len(data_loader)
    last_group_size = total_batches % accum_steps or accum_steps
    last_group_start = total_batches - last_group_size + 1
    optimizer.zero_grad(set_to_none=True)
    for step_index, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        teacher_features = None
        if teacher is not None:
            teacher_inputs = denormalize_for_teacher(images)
            with torch.no_grad():
                teacher_features = teacher(teacher_inputs)

        autocast_context = build_autocast_context(device, amp_enabled=amp_enabled)
        with autocast_context:
            outputs = model(images, targets=targets, teacher_encoder_output=teacher_features)
            loss_dict = criterion(outputs, targets)
            total_loss = compute_total_loss(loss_dict)

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Encountered a non-finite loss during stage '{stage_name}', epoch {global_epoch}: {total_loss.item()}"
            )

        should_step = step_index % accum_steps == 0 or step_index == total_batches
        loss_divisor = last_group_size if step_index >= last_group_start else accum_steps
        scaled_loss = total_loss / loss_divisor
        grad_norm_value = None
        gam_gradient_stats = None
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
            if should_step and (clip_grad_norm > 0 or gradient_monitor is not None):
                scaler.unscale_(optimizer)
            if should_step and gradient_monitor is not None:
                gam_gradient_stats = compute_gam_gradient_statistics(gradient_monitor)
            if should_step and clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
        else:
            scaled_loss.backward()
            if should_step and gradient_monitor is not None:
                gam_gradient_stats = compute_gam_gradient_statistics(gradient_monitor)
            if should_step and clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        if should_step and scheduler is not None:
            scheduler.step()

        num_steps += 1
        running_metrics["train_loss"] += float(total_loss.detach().item())
        for key, value in loss_dict.items():
            running_metrics[f"train_{key}"] += float(value.detach().item())
        if grad_norm_value is not None:
            running_metrics["grad_norm"] += grad_norm_value
        if gam_gradient_stats is not None:
            for module_name, module_norm in gam_gradient_stats["module_norms"].items():
                running_metrics[f"grad_norm_{module_name}"] += float(module_norm)
            running_metrics["grad_ratio_aifi"] += float(gam_gradient_stats["aifi_ratio"])

        if step_index % max(log_interval, 1) == 0 or step_index == len(data_loader):
            elapsed = time.time() - step_start_time
            LOGGER.info(
                "stage=%s epoch=%s step=%s/%s loss=%.4f lr=%.7f accum=%s time=%.1fs",
                stage_name,
                global_epoch,
                step_index,
                len(data_loader),
                float(total_loss.detach().item()),
                current_lr(optimizer),
                accum_steps,
                elapsed,
            )

    averaged = {key: value / max(num_steps, 1) for key, value in running_metrics.items()}
    if distill_weight_used is not None:
        averaged["distill_weight"] = float(distill_weight_used)
    if gradient_monitor is not None and "grad_ratio_aifi" in averaged and hasattr(criterion, "update_distillation_weight"):
        target_band = get_gam_target_band(criterion)
        avg_backbone_grad = float(averaged.get("grad_norm_backbone", 0.0))
        avg_aifi_grad = float(averaged.get("grad_norm_aifi", 0.0))
        avg_ccff_grad = float(averaged.get("grad_norm_ccff", 0.0))
        avg_decoder_grad = float(averaged.get("grad_norm_decoder", 0.0))
        averaged["grad_ratio_aifi_pct"] = float(averaged["grad_ratio_aifi"] * 100.0)
        if target_band is not None:
            averaged["gam_target_lower_pct"] = float(target_band[0] * 100.0)
            averaged["gam_target_upper_pct"] = float(target_band[1] * 100.0)
        averaged["next_distill_weight"] = float(
            criterion.update_distillation_weight(averaged["grad_ratio_aifi"])
        )
        averaged["distill_weight_next"] = averaged["next_distill_weight"]
        update_info = (
            criterion.get_last_distillation_update()
            if hasattr(criterion, "get_last_distillation_update")
            else None
        )
        LOGGER.info(
            (
                "stage=%s epoch=%s GAM avg_aifi_ratio=%.2f%% target=%s "
                "avg_grad_norms(backbone=%.4f aifi=%.4f ccff=%.4f decoder=%.4f) "
                "distill_weight=%.4f -> %.4f"
            ),
            stage_name,
            global_epoch,
            averaged["grad_ratio_aifi_pct"],
            (
                f"{averaged['gam_target_lower_pct']:.2f}%..{averaged['gam_target_upper_pct']:.2f}%"
                if target_band is not None
                else "n/a"
            ),
            avg_backbone_grad,
            avg_aifi_grad,
            avg_ccff_grad,
            avg_decoder_grad,
            averaged.get("distill_weight", 0.0),
            averaged["next_distill_weight"],
        )
        if isinstance(update_info, dict):
            averaged["gam_weight_rate_limited"] = 1.0 if update_info.get("rate_limited", False) else 0.0
            averaged["gam_weight_clamped"] = 1.0 if update_info.get("weight_clamped", False) else 0.0
            if update_info.get("rate_limited", False) or update_info.get("weight_clamped", False):
                LOGGER.warning(
                    (
                        "stage=%s epoch=%s GAM guardrail activated | status=%s avg_aifi_ratio=%.2f%% "
                        "target_ratio=%s raw_weight=%.4f adjusted_weight=%.4f rate_limited=%s clamped=%s"
                    ),
                    stage_name,
                    global_epoch,
                    update_info.get("status", "unknown"),
                    averaged["grad_ratio_aifi_pct"],
                    (
                        f"{float(update_info['target_ratio']) * 100.0:.2f}%"
                        if update_info.get("target_ratio") is not None
                        else "n/a"
                    ),
                    float(update_info.get("raw_weight", averaged["next_distill_weight"])),
                    float(update_info.get("new_weight", averaged["next_distill_weight"])),
                    bool(update_info.get("rate_limited", False)),
                    bool(update_info.get("weight_clamped", False)),
                )
    averaged["lr"] = current_lr(optimizer)
    averaged["epoch_seconds"] = time.time() - step_start_time
    return averaged


def validate_one_epoch(
    *,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    global_epoch: int,
    stage_name: str,
    amp_enabled: bool,
    eval_model: nn.Module | None = None,
) -> dict[str, Any]:
    # When EMA is active we validate the averaged weights, which are the ones we
    # ultimately ship; otherwise validate the live model.
    target_model = eval_model if eval_model is not None else model
    eval_metrics = evaluate_detection(
        target_model,
        criterion,
        data_loader,
        device,
        topk=getattr(getattr(target_model, "postprocessor", None), "num_top_queries", 300),
        amp_enabled=amp_enabled,
    )
    eval_metrics["epoch"] = int(global_epoch)
    eval_metrics["stage"] = stage_name
    return eval_metrics


def run_stage(
    *,
    base_config: dict[str, Any],
    config_path: Path,
    stage: dict[str, Any],
    stage_index: int,
    total_stages: int,
    model: nn.Module,
    teacher: nn.Module | None,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    output_dir: Path,
    metrics_log_path: Path,
    metrics_history: list[dict[str, Any]],
    checkpoint_freq: int,
    best_map50: float,
    global_epoch_start: int,
    resume_payload: dict[str, Any] | None,
    amp_enabled: bool,
    log_interval: int,
    clip_grad_norm: float,
    eval_every: int,
    accum_steps: int,
    focus_classes: list[str] | None,
    ema: ModelEMA | None = None,
) -> tuple[int, float]:
    stage_name = str(stage.get("name", f"stage{stage_index + 1}"))
    stage_epochs = int(stage["epochs"])
    stage_output_dir = Path(stage.get("output_dir") or output_dir / stage_name)
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    stage_runtime_config = build_stage_runtime_config(base_config, stage)
    active_teacher = resolve_stage_teacher(stage_runtime_config, teacher)
    criterion = build_criterion_from_config(stage_runtime_config).to(device)
    if active_teacher is None and "distill" in getattr(criterion, "losses", []):
        LOGGER.warning(
            "Stage '%s' enables distillation loss, but no teacher model is configured. "
            "Distillation terms will stay at zero.",
            stage_name,
        )

    backbone = getattr(model, "backbone", None)
    if backbone is not None and backbone.__class__.__name__ == "HGNetv2":
        stage_runtime = apply_backbone_freeze_strategy(model, stage_runtime_config.get("HGNetv2", {}))
    elif backbone is not None and backbone.__class__.__name__ == "PResNet":
        stage_runtime = apply_backbone_freeze_strategy(model, stage_runtime_config.get("PResNet", {}))
    else:
        stage_runtime = apply_backbone_freeze_strategy(model, {})

    gradient_monitor = None
    if active_teacher is not None:
        gradient_monitor = build_gam_gradient_monitor(model, criterion)
        if gradient_monitor is not None:
            target_band = get_gam_target_band(criterion)
            LOGGER.info(
                (
                    "GAM enabled for stage '%s' | monitored parameters: backbone=%s aifi=%s ccff=%s decoder=%s "
                    "| target_aifi_ratio=%s | distill_weight=%.4f"
                ),
                stage_name,
                len(gradient_monitor["backbone"]),
                len(gradient_monitor["aifi"]),
                len(gradient_monitor["ccff"]),
                len(gradient_monitor["decoder"]),
                (
                    f"{target_band[0] * 100.0:.2f}%..{target_band[1] * 100.0:.2f}%"
                    if target_band is not None
                    else "n/a"
                ),
                float(getattr(criterion, "get_current_distill_weight", lambda: 0.0)()),
            )
    elif hasattr(criterion, "has_adaptive_distillation") and criterion.has_adaptive_distillation():
        LOGGER.warning(
            "Stage '%s' enables GAM, but no teacher model is configured. Adaptive modulation will stay inactive.",
            stage_name,
        )
    LOGGER.info(
        "Stage '%s' teacher=%s distill_loss=%s",
        stage_name,
        "enabled" if active_teacher is not None else "disabled",
        "enabled" if "distill" in getattr(criterion, "losses", []) else "disabled",
    )

    optimizer = build_optimizer(model, stage["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        stage.get("scheduler"),
        total_steps=max(stage_epochs * math.ceil(len(train_loader) / max(accum_steps, 1)), 1),
    )

    start_stage_epoch = 1
    if resume_payload is not None and int(resume_payload.get("stage_index", -1)) == stage_index:
        start_stage_epoch = int(resume_payload.get("stage_epoch", 0)) + 1
        optimizer_state = resume_payload.get("optimizer_state_dict")
        scheduler_state = resume_payload.get("scheduler_state_dict")
        scaler_state = resume_payload.get("scaler_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if scaler.is_enabled() and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        criterion_distill_state = resume_payload.get("criterion_distill_state")
        if criterion_distill_state is not None and hasattr(criterion, "load_distillation_state"):
            criterion.load_distillation_state(criterion_distill_state)
        LOGGER.info("Resuming stage '%s' from local epoch %s.", stage_name, start_stage_epoch)

    if start_stage_epoch > stage_epochs:
        LOGGER.info("Stage '%s' is already complete. Moving to the next stage.", stage_name)
        return global_epoch_start, best_map50

    LOGGER.info(
        "Starting stage '%s' (%s/%s) for %s epochs.",
        stage_name,
        stage_index + 1,
        total_stages,
        stage_epochs,
    )

    global_epoch = global_epoch_start
    for stage_epoch in range(start_stage_epoch, stage_epochs + 1):
        global_epoch += 1
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            teacher=active_teacher,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            stage_name=stage_name,
            global_epoch=global_epoch,
            amp_enabled=amp_enabled,
            log_interval=log_interval,
            clip_grad_norm=clip_grad_norm,
            accum_steps=accum_steps,
            stage_runtime=stage_runtime,
            gradient_monitor=gradient_monitor,
            ema=ema,
        )

        eval_metrics: dict[str, Any] | None = None
        if val_loader is not None and global_epoch % max(eval_every, 1) == 0:
            eval_metrics = validate_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                device=device,
                global_epoch=global_epoch,
                stage_name=stage_name,
                amp_enabled=amp_enabled,
                eval_model=ema.module if ema is not None else None,
            )

        if eval_metrics is not None:
            per_class_ap = eval_metrics.get("per_class_ap50", {})
            if per_class_ap:
                ranked = sorted(per_class_ap.items(), key=lambda item: item[1])
                summary = " ".join(f"{name}={ap:.3f}" for name, ap in ranked)
                LOGGER.info("epoch %s per-class AP50 (low->high): %s", global_epoch, summary)

        metrics_record: dict[str, Any] = {
            "epoch": global_epoch,
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            **train_metrics,
        }
        if eval_metrics is not None:
            metrics_record["val_loss"] = float(eval_metrics.get("loss", 0.0))
            metrics_record["map50"] = float(eval_metrics.get("map50", 0.0))
            metrics_record["best_f1"] = float(
                eval_metrics.get("confidence_curves", {}).get("best_f1", 0.0)
            )

        append_metrics_record(metrics_log_path, metrics_record, metrics_history)
        write_monitoring_artifacts(
            output_dir=output_dir,
            metrics_history=metrics_history,
            latest_eval=eval_metrics,
            focus_classes=focus_classes,
        )

        checkpoint_metrics = {
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            "global_epoch": global_epoch,
            "train_loss": metrics_record["train_loss"],
            "map50": metrics_record.get("map50"),
        }
        ema_state_dict = ema.state_dict() if ema is not None else None

        if eval_metrics is not None:
            map50 = float(eval_metrics.get("map50", 0.0))
            if map50 >= best_map50:
                best_map50 = map50
                save_checkpoint(
                    path=output_dir / "best_map50.pt",
                    config_path=config_path,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    global_epoch=global_epoch,
                    best_map50=best_map50,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics=checkpoint_metrics,
                    criterion_distill_state=(
                        criterion.get_distillation_state()
                        if hasattr(criterion, "get_distillation_state")
                        else None
                    ),
                    ema_state_dict=ema_state_dict,
                )

        save_checkpoint(
            path=output_dir / "latest.pt",
            config_path=config_path,
            stage_index=stage_index,
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            global_epoch=global_epoch,
            best_map50=best_map50,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=checkpoint_metrics,
            criterion_distill_state=(
                criterion.get_distillation_state()
                if hasattr(criterion, "get_distillation_state")
                else None
            ),
            ema_state_dict=ema_state_dict,
        )

        if checkpoint_freq > 0 and (stage_epoch % checkpoint_freq == 0 or stage_epoch == stage_epochs):
            save_checkpoint(
                path=stage_output_dir / f"epoch_{stage_epoch:03d}.pt",
                config_path=config_path,
                stage_index=stage_index,
                stage_name=stage_name,
                stage_epoch=stage_epoch,
                global_epoch=global_epoch,
                best_map50=best_map50,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metrics=checkpoint_metrics,
                criterion_distill_state=(
                    criterion.get_distillation_state()
                    if hasattr(criterion, "get_distillation_state")
                    else None
                ),
                ema_state_dict=ema_state_dict,
            )

        LOGGER.info(
            "Finished epoch %s | stage=%s stage_epoch=%s train_loss=%.4f map50=%s",
            global_epoch,
            stage_name,
            stage_epoch,
            metrics_record["train_loss"],
            f"{metrics_record.get('map50', 'n/a')}",
        )

    return global_epoch, best_map50


def train(args: argparse.Namespace) -> None:
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    config = resolve_config_paths(config, output_dir_override=args.output_dir)
    config = resolve_dataset_classes(config)

    output_dir = Path(config["output_dir"])
    setup_logging(output_dir)
    set_seed(args.seed)
    LOGGER.info("Loaded config from %s", config_path)

    device = resolve_device(args.device, None)
    amp_enabled = bool(not args.no_amp and device.type == "cuda")
    LOGGER.info("Using device=%s amp=%s", device, amp_enabled)
    accum_steps = max(int(args.accum_steps), 1)
    LOGGER.info(
        "Gradient accumulation steps=%s effective_train_batch_size=%s",
        accum_steps,
        int(config["train_dataloader"].get("batch_size", 1)) * accum_steps,
    )

    train_dataset, train_loader, val_dataset, val_loader = build_datasets_and_loaders(
        config,
        seed=args.seed,
        num_workers_override=args.num_workers,
    )
    synchronize_num_classes(config, train_dataset, val_dataset)
    save_yaml(output_dir / "resolved_config.yml", config)

    model, teacher = build_model_bundle(config, device=device)
    total_params, trainable_params = count_parameters(model)
    LOGGER.info("Built model with %s trainable parameters out of %s total.", trainable_params, total_params)
    if teacher is not None:
        LOGGER.info("Teacher model is enabled for this run.")

    resume_payload = None
    if args.resume:
        resume_path = resolve_project_path(args.resume)
        LOGGER.info("Loading resume checkpoint from %s", resume_path)
        resume_payload = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume_payload["model_state_dict"], strict=False)
    else:
        student_checkpoint = args.student_checkpoint or config.get("student_checkpoint")
        if student_checkpoint:
            student_checkpoint_path = resolve_project_path(student_checkpoint)
            if student_checkpoint_path.exists():
                missing_keys, unexpected_keys, skipped_mismatched = load_model_weights(
                    model,
                    student_checkpoint_path,
                    strict=False,
                )
                LOGGER.info(
                    "Loaded student checkpoint from %s | missing=%s unexpected=%s skipped_shape_mismatch=%s",
                    student_checkpoint_path,
                    len(missing_keys),
                    len(unexpected_keys),
                    len(skipped_mismatched),
                )
                if skipped_mismatched:
                    preview = ", ".join(item[0] for item in skipped_mismatched[:8])
                    LOGGER.info(
                        "Skipped mismatched checkpoint tensors. This is expected when class heads do not match the dataset. "
                        "Examples: %s",
                        preview,
                    )
            else:
                LOGGER.warning(
                    "Configured student checkpoint does not exist: %s. Training will start from the current initialization.",
                    student_checkpoint_path,
                )

    metrics_log_path = output_dir / "metrics.jsonl"
    metrics_history = load_metrics_history(metrics_log_path)
    checkpoint_freq = int(config.get("checkpoint_freq", 1))
    scaler = build_grad_scaler(amp_enabled=amp_enabled)
    stages = normalize_stage_entries(config)
    focus_classes = list(getattr(train_dataset, "class_names", [])[:8]) or None

    # EMA is initialized AFTER all pretrained-weight loading above, so it never
    # interferes with checkpoint loading. It simply shadows the current weights.
    ema = None
    ema_cfg = config.get("ema") or {}
    if bool(ema_cfg.get("enabled", False)):
        ema = ModelEMA(
            model,
            decay=float(ema_cfg.get("decay", 0.9999)),
            warmup=int(ema_cfg.get("warmup", 2000)),
        )
        LOGGER.info("EMA enabled (decay=%s warmup=%s).", ema.decay, ema.warmup)
        if resume_payload is not None and resume_payload.get("ema_state_dict") is not None:
            ema.load_state_dict(resume_payload["ema_state_dict"])
            LOGGER.info("Restored EMA weights from resume checkpoint.")

    best_map50 = float(resume_payload.get("best_map50", 0.0)) if resume_payload is not None else 0.0
    global_epoch = int(resume_payload.get("global_epoch", 0)) if resume_payload is not None else 0
    start_stage_index = int(resume_payload.get("stage_index", 0)) if resume_payload is not None else 0

    for stage_index, stage in enumerate(stages):
        if resume_payload is not None and stage_index < start_stage_index:
            continue
        global_epoch, best_map50 = run_stage(
            base_config=config,
            config_path=config_path,
            stage=stage,
            stage_index=stage_index,
            total_stages=len(stages),
            model=model,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            scaler=scaler,
            output_dir=output_dir,
            metrics_log_path=metrics_log_path,
            metrics_history=metrics_history,
            checkpoint_freq=checkpoint_freq,
            best_map50=best_map50,
            global_epoch_start=global_epoch,
            resume_payload=resume_payload,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
            clip_grad_norm=args.clip_grad_norm,
            eval_every=args.eval_every,
            accum_steps=accum_steps,
            focus_classes=focus_classes,
            ema=ema,
        )
        resume_payload = None

    final_payload = {
        "completed": True,
        "global_epoch": global_epoch,
        "best_map50": best_map50,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Training complete. global_epoch=%s best_map50=%.4f", global_epoch, best_map50)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
