from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
# The active local package lives under src/rtdetr_v4.
# This is the project mainline package, not the removed legacy compatibility tree.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtdetr_v4.config import load_config, merge_dict
from rtdetr_v4.data import build_transforms
from rtdetr_v4.data.image_io import load_rgb_image
from rtdetr_v4.models import build_model_from_config
from rtdetr_v4.runtime import load_model_weights, resolve_config_paths, resolve_dataset_classes
from rtdetr_v4.utils.misc import resolve_device, resolve_project_path


LOGGER = logging.getLogger("infer")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BOX_COLORS = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (149, 165, 166),
    (243, 156, 18),
    (52, 73, 94),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RT-DETRv4 inference entrypoint.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/project.yml",
        help="Model config path relative to the repository root.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="One image path or a directory containing images.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override. Defaults to config.student_checkpoint.",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Optional stage name or 1-based stage index used to apply stage config_overrides before inference.",
    )
    parser.add_argument(
        "--allow-mismatched-head",
        action="store_true",
        help="Allow inference to continue when checkpoint class-head tensors do not match the configured class set.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/inference",
        help="Directory used for JSON predictions and visualizations.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Only keep predictions with score >= this threshold.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Maximum number of decoded predictions per image before thresholding.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input points to a directory, scan subdirectories recursively.",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="Disable annotated image output.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA AMP.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "infer.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_class_names(config: dict[str, Any]) -> list[str]:
    dataset_cfg = config.get("dataset", {})
    class_names = list(dataset_cfg.get("classes") or [])
    if class_names:
        configured_num_classes = config.get("num_classes")
        if configured_num_classes is not None and int(configured_num_classes) != len(class_names):
            LOGGER.warning(
                "Config num_classes=%s does not match dataset.classes length=%s. "
                "The class list length will be used for inference.",
                configured_num_classes,
                len(class_names),
            )
        config["num_classes"] = len(class_names)
    return class_names


def is_classification_head_tensor(name: str) -> bool:
    """Detect tensors that imply the checkpoint class space differs from the config."""
    head_markers = (
        "score_head",
        "class_embed",
        "denoising_class_embed",
    )
    return any(marker in name for marker in head_markers)


def resolve_stage_runtime_config(config: dict[str, Any], stage_selector: str | None) -> dict[str, Any]:
    if not stage_selector:
        return config

    stages = config.get("stages") or []
    if not stages:
        raise ValueError("This config does not define any stages, so --stage cannot be used.")

    if stage_selector.isdigit():
        stage_index = int(stage_selector) - 1
        if stage_index < 0 or stage_index >= len(stages):
            raise ValueError(f"Stage index out of range: {stage_selector}")
        selected = stages[stage_index]
    else:
        matches = [stage for stage in stages if stage.get("name") == stage_selector]
        if not matches:
            available = ", ".join(stage.get("name", f"stage{index + 1}") for index, stage in enumerate(stages))
            raise ValueError(f"Unknown stage '{stage_selector}'. Available stages: {available}")
        selected = matches[0]

    runtime_config = copy.deepcopy(config)
    stage_overrides = selected.get("config_overrides") or {}
    if stage_overrides:
        merge_dict(runtime_config, stage_overrides, inplace=True)
    runtime_config["_selected_stage_name"] = selected.get("name")
    return runtime_config


def resolve_image_paths(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    image_paths = [
        path
        for path in sorted(input_path.glob(pattern))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found under {input_path}. "
            f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    return image_paths


def build_empty_target(image_path: Path, image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    return {
        "image_id": torch.tensor(0, dtype=torch.long),
        "orig_size": torch.tensor([height, width], dtype=torch.long),
        "size": torch.tensor([height, width], dtype=torch.long),
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "sample_id": image_path.stem,
        "image_path": str(image_path),
    }


def map_boxes_to_original_space(boxes: torch.Tensor, target: dict[str, Any]) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()

    mapped = boxes.clone().to(dtype=torch.float32)
    orig_size = target["orig_size"].to(dtype=torch.float32)
    resized_size = target.get("resized_size", target["size"]).to(dtype=torch.float32)
    padding = target.get("padding", torch.zeros(4, dtype=torch.float32)).to(dtype=torch.float32)
    scale_factor = target.get("scale_factor")
    if scale_factor is None:
        scale_y = resized_size[0] / max(float(orig_size[0].item()), 1.0)
        scale_x = resized_size[1] / max(float(orig_size[1].item()), 1.0)
    else:
        scale_y = float(scale_factor[0].item())
        scale_x = float(scale_factor[1].item())

    pad_left = float(padding[0].item())
    pad_top = float(padding[1].item())

    mapped[:, [0, 2]] -= pad_left
    mapped[:, [1, 3]] -= pad_top
    mapped[:, [0, 2]] = mapped[:, [0, 2]].clamp(min=0.0, max=float(resized_size[1].item()))
    mapped[:, [1, 3]] = mapped[:, [1, 3]].clamp(min=0.0, max=float(resized_size[0].item()))
    mapped[:, [0, 2]] /= max(scale_x, 1e-12)
    mapped[:, [1, 3]] /= max(scale_y, 1e-12)
    mapped[:, [0, 2]] = mapped[:, [0, 2]].clamp(min=0.0, max=float(orig_size[1].item()))
    mapped[:, [1, 3]] = mapped[:, [1, 3]].clamp(min=0.0, max=float(orig_size[0].item()))
    return mapped


def format_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    score_threshold: float,
    class_names: list[str],
) -> list[dict[str, Any]]:
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask].detach().cpu()
    scores = scores[keep_mask].detach().cpu()
    labels = labels[keep_mask].detach().cpu()

    detections: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        label_index = int(label.item())
        label_name = class_names[label_index] if 0 <= label_index < len(class_names) else str(label_index)
        detections.append(
            {
                "label_id": label_index,
                "label_name": label_name,
                "score": round(float(score.item()), 6),
                "box_xyxy": [round(float(value), 3) for value in box.tolist()],
            }
        )
    return detections


def draw_detections(image, detections: list[dict[str, Any]]) -> Any:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for detection in detections:
        label_id = int(detection["label_id"])
        color = BOX_COLORS[label_id % len(BOX_COLORS)]
        x0, y0, x1, y1 = detection["box_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)

        caption = f"{detection['label_name']} {detection['score']:.2f}"
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_top = max(0, int(y0) - text_height - 4)
        draw.rectangle(
            (x0, text_top, x0 + text_width + 6, text_top + text_height + 4),
            fill=color,
        )
        draw.text((x0 + 3, text_top + 2), caption, fill=(255, 255, 255), font=font)
    return canvas


def build_output_path(output_root: Path, image_path: Path, *, input_root: Path) -> Path:
    if input_root.is_file():
        relative = Path(image_path.name)
    else:
        relative = image_path.relative_to(input_root)
    return output_root / relative


@torch.no_grad()
def infer_one_image(
    model: torch.nn.Module,
    image_path: Path,
    *,
    transforms,
    device: torch.device,
    amp_enabled: bool,
    topk: int,
    score_threshold: float,
    class_names: list[str],
) -> tuple[dict[str, Any], Any]:
    original_image = load_rgb_image(image_path)
    target = build_empty_target(image_path, original_image.size)
    image_tensor, target = transforms(original_image, target)

    inputs = image_tensor.unsqueeze(0).to(device)
    image_sizes = torch.stack([target["size"]], dim=0).to(device)
    autocast_context = (
        torch.autocast(
            device_type=device.type,
            enabled=bool(amp_enabled and device.type == "cuda"),
        )
        if device.type in {"cuda", "cpu"}
        else nullcontext()
    )

    with autocast_context:
        outputs = model(inputs)
    results = model.post_process(outputs, image_sizes=image_sizes, topk=topk)
    result = results[0]

    original_boxes = map_boxes_to_original_space(result["boxes"].detach().cpu(), target)
    detections = format_detections(
        original_boxes,
        result["scores"].detach().cpu(),
        result["labels"].detach().cpu(),
        score_threshold=score_threshold,
        class_names=class_names,
    )
    record = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_size": {
            "width": int(original_image.width),
            "height": int(original_image.height),
        },
        "num_detections": len(detections),
        "detections": detections,
    }
    return record, original_image


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    config = resolve_stage_runtime_config(config, getattr(args, "stage", None))
    config = resolve_config_paths(config, output_dir_override=args.output_dir)
    config = resolve_dataset_classes(config)
    class_names = resolve_class_names(config)

    output_dir = Path(config["output_dir"])
    setup_logging(output_dir)

    device = resolve_device(args.device, None)
    amp_enabled = bool(not args.no_amp and device.type == "cuda")

    model = build_model_from_config(config)
    model.to(device)
    model.eval()

    checkpoint_value = args.checkpoint or config.get("student_checkpoint")
    if not checkpoint_value:
        raise ValueError("No checkpoint specified. Pass --checkpoint or set student_checkpoint in the config.")
    checkpoint_path = resolve_project_path(checkpoint_value)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    missing_keys, unexpected_keys, skipped_mismatched = load_model_weights(
        model,
        checkpoint_path,
        strict=False,
    )
    LOGGER.info(
        "Loaded checkpoint %s | missing=%s unexpected=%s skipped_shape_mismatch=%s",
        checkpoint_path,
        len(missing_keys),
        len(unexpected_keys),
        len(skipped_mismatched),
    )
    if skipped_mismatched:
        preview = ", ".join(item[0] for item in skipped_mismatched[:8])
        LOGGER.warning("Skipped mismatched tensors during inference load. Examples: %s", preview)
        skipped_head_tensors = [name for name, _, _ in skipped_mismatched if is_classification_head_tensor(name)]
        if skipped_head_tensors:
            if not args.allow_mismatched_head:
                preview = ", ".join(skipped_head_tensors[:8])
                raise ValueError(
                    "Checkpoint class-head tensors do not match the configured class set. "
                    "Inference is blocked by default because predictions would likely be invalid. "
                    f"Examples: {preview}. "
                    "Use a fine-tuned checkpoint for this dataset, or pass --allow-mismatched-head to override."
                )
            LOGGER.warning(
                "Classification head tensors were skipped, but inference is continuing because "
                "--allow-mismatched-head was set."
            )

    transforms = build_transforms(config["val_transforms"], train=False)
    input_path = Path(args.input).resolve()
    image_paths = resolve_image_paths(input_path, recursive=args.recursive)
    visualizations_dir = output_dir / "visualizations"
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.json"

    summary = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_path": str(input_path),
        "num_images": len(image_paths),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "score_threshold": float(args.score_threshold),
        "topk": int(args.topk),
        "class_names": class_names,
        "selected_stage": config.get("_selected_stage_name"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    start_time = time.perf_counter()
    successful_images = 0
    failures: list[dict[str, str]] = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, image_path in enumerate(image_paths, start=1):
            try:
                record, original_image = infer_one_image(
                    model,
                    image_path,
                    transforms=transforms,
                    device=device,
                    amp_enabled=amp_enabled,
                    topk=args.topk,
                    score_threshold=args.score_threshold,
                    class_names=class_names,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                if not args.no_save_vis:
                    visualization = draw_detections(original_image, record["detections"])
                    output_path = build_output_path(visualizations_dir, image_path, input_root=input_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    visualization.save(output_path)

                successful_images += 1
                LOGGER.info(
                    "[%s/%s] %s -> %s detections",
                    index,
                    len(image_paths),
                    image_path.name,
                    record["num_detections"],
                )
            except Exception as exc:
                failure_record = {
                    "image_path": str(image_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                failures.append(failure_record)
                LOGGER.exception(
                    "[%s/%s] Failed to infer %s",
                    index,
                    len(image_paths),
                    image_path,
                )

    elapsed_seconds = time.perf_counter() - start_time
    summary["successful_images"] = int(successful_images)
    summary["failed_images"] = int(len(failures))
    summary["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["predictions_path"] = str(predictions_path)
    summary["failures_path"] = str(failures_path)
    if not args.no_save_vis:
        summary["visualizations_dir"] = str(visualizations_dir)
    save_json(output_dir / "run_summary.json", summary)
    save_json(failures_path, failures)

    if failures:
        LOGGER.warning(
            "Inference completed with %s failed images. Failure details were written to %s",
            len(failures),
            failures_path,
        )
    if successful_images == 0:
        raise RuntimeError(
            f"Inference failed for all {len(image_paths)} requested images. See {failures_path} for details."
        )
    LOGGER.info(
        "Inference finished for %s images in %.2fs. Outputs: %s",
        successful_images,
        elapsed_seconds,
        output_dir,
    )


if __name__ == "__main__":
    main()
