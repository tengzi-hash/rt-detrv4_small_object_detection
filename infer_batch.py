from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from infer import (
    build_empty_target,
    build_output_path,
    draw_detections,
    format_detections,
    is_classification_head_tensor,
    load_config,
    load_model_weights,
    load_rgb_image,
    map_boxes_to_original_space,
    resolve_class_names,
    resolve_config_paths,
    resolve_dataset_classes,
    resolve_device,
    resolve_image_paths,
    resolve_project_path,
    resolve_stage_runtime_config,
    save_json,
)
from rtdetr_v4.data import build_transforms
from rtdetr_v4.models import build_model_from_config


LOGGER = logging.getLogger("infer_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched RT-DETRv4 inference entrypoint.")
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
        default="./outputs/inference_batch",
        help="Directory used for JSON predictions and visualizations.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to run in one forward pass.",
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
    log_path = output_dir / "infer_batch.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def iter_batches(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def prepare_batch(
    image_paths: list[Path],
    *,
    transforms,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[Any]]:
    image_tensors = []
    targets = []
    original_images = []
    for image_path in image_paths:
        original_image = load_rgb_image(image_path)
        target = build_empty_target(image_path, original_image.size)
        image_tensor, target = transforms(original_image, target)
        image_tensors.append(image_tensor)
        targets.append(target)
        original_images.append(original_image)
    return torch.stack(image_tensors, dim=0), targets, original_images


def format_batch_results(
    image_paths: list[Path],
    original_images: list[Any],
    targets: list[dict[str, Any]],
    results: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
    class_names: list[str],
) -> list[tuple[dict[str, Any], Any]]:
    formatted = []
    for image_path, original_image, target, result in zip(image_paths, original_images, targets, results):
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
        formatted.append((record, original_image))
    return formatted


@torch.no_grad()
def infer_image_batch(
    model: torch.nn.Module,
    image_paths: list[Path],
    *,
    transforms,
    device: torch.device,
    amp_enabled: bool,
    topk: int,
    score_threshold: float,
    class_names: list[str],
) -> list[tuple[dict[str, Any], Any]]:
    inputs, targets, original_images = prepare_batch(image_paths, transforms=transforms)
    inputs = inputs.to(device)
    image_sizes = torch.stack([target["size"] for target in targets], dim=0).to(device)
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
    return format_batch_results(
        image_paths,
        original_images,
        targets,
        results,
        score_threshold=score_threshold,
        class_names=class_names,
    )


def load_inference_model(config: dict[str, Any], checkpoint_path: Path, *, allow_mismatched_head: bool):
    model = build_model_from_config(config)
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
        if skipped_head_tensors and not allow_mismatched_head:
            preview = ", ".join(skipped_head_tensors[:8])
            raise ValueError(
                "Checkpoint class-head tensors do not match the configured class set. "
                "Inference is blocked by default because predictions would likely be invalid. "
                f"Examples: {preview}. "
                "Use a fine-tuned checkpoint for this dataset, or pass --allow-mismatched-head to override."
            )
        if skipped_head_tensors:
            LOGGER.warning(
                "Classification head tensors were skipped, but inference is continuing because "
                "--allow-mismatched-head was set."
            )
    return model


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

    checkpoint_value = args.checkpoint or config.get("student_checkpoint")
    if not checkpoint_value:
        raise ValueError("No checkpoint specified. Pass --checkpoint or set student_checkpoint in the config.")
    checkpoint_path = resolve_project_path(checkpoint_value)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_inference_model(
        config,
        checkpoint_path,
        allow_mismatched_head=args.allow_mismatched_head,
    )
    model.to(device)
    model.eval()

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
        "batch_size": int(args.batch_size),
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
    batches = iter_batches(image_paths, args.batch_size)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for batch_index, batch_paths in enumerate(batches, start=1):
            try:
                batch_results = infer_image_batch(
                    model,
                    batch_paths,
                    transforms=transforms,
                    device=device,
                    amp_enabled=amp_enabled,
                    topk=args.topk,
                    score_threshold=args.score_threshold,
                    class_names=class_names,
                )
                for record, original_image in batch_results:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if not args.no_save_vis:
                        visualization = draw_detections(original_image, record["detections"])
                        output_path = build_output_path(
                            visualizations_dir,
                            Path(record["image_path"]),
                            input_root=input_path,
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        visualization.save(output_path)

                    successful_images += 1
                    LOGGER.info(
                        "[batch %s/%s] %s -> %s detections",
                        batch_index,
                        len(batches),
                        record["image_name"],
                        record["num_detections"],
                    )
            except RuntimeError as exc:
                if device.type == "cuda" and "out of memory" in str(exc).lower():
                    LOGGER.error(
                        "CUDA out of memory for batch_size=%s. Try a smaller --batch-size.",
                        args.batch_size,
                    )
                    torch.cuda.empty_cache()
                for image_path in batch_paths:
                    failures.append(
                        {
                            "image_path": str(image_path),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                LOGGER.exception(
                    "[batch %s/%s] Failed to infer %s images.",
                    batch_index,
                    len(batches),
                    len(batch_paths),
                )
            except Exception as exc:
                for image_path in batch_paths:
                    failures.append(
                        {
                            "image_path": str(image_path),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                LOGGER.exception(
                    "[batch %s/%s] Failed to infer %s images.",
                    batch_index,
                    len(batches),
                    len(batch_paths),
                )

    elapsed_seconds = time.perf_counter() - start_time
    summary["successful_images"] = int(successful_images)
    summary["failed_images"] = int(len(failures))
    summary["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    summary["images_per_second"] = round(float(successful_images / elapsed_seconds), 3) if elapsed_seconds > 0 else 0.0
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["predictions_path"] = str(predictions_path)
    summary["failures_path"] = str(failures_path)
    if not args.no_save_vis:
        summary["visualizations_dir"] = str(visualizations_dir)
    save_json(output_dir / "run_summary.json", summary)
    save_json(failures_path, failures)

    if failures:
        LOGGER.warning(
            "Batched inference completed with %s failed images. Failure details were written to %s",
            len(failures),
            failures_path,
        )
    if successful_images == 0:
        raise RuntimeError(
            f"Batched inference failed for all {len(image_paths)} requested images. See {failures_path} for details."
        )
    LOGGER.info(
        "Batched inference finished for %s images in %.2fs. Outputs: %s",
        successful_images,
        elapsed_seconds,
        output_dir,
    )


if __name__ == "__main__":
    main()
