"""Validation helpers for the migrated RT-DETR v4 infrastructure.

This file keeps the old project's rich validation outputs, but it is deliberately
independent from the old trainer so it can later be connected to the official
v4 training loop with minimal friction.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..utils.box_ops import box_iou


@dataclass
class Prediction:
    """One predicted box after model post-processing."""

    image_id: int
    score: float
    box: Tensor


@dataclass
class ImageRecord:
    """Ground truth and predictions for one evaluated image."""

    gt_boxes: Tensor
    gt_labels: Tensor
    pred_boxes: Tensor
    pred_scores: Tensor
    pred_labels: Tensor


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    """Move tensor values inside target dictionaries onto the evaluation device."""
    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in target.items()
            }
        )
    return moved


def compute_total_loss(loss_dict: dict[str, Tensor]) -> Tensor:
    """Sum the already-weighted loss terms returned by RTv4Criterion."""
    total_loss = None
    for value in loss_dict.values():
        total_loss = value if total_loss is None else total_loss + value
    if total_loss is None:
        raise RuntimeError("Criterion returned no loss terms; nothing can be backpropagated.")
    return total_loss


def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """Compute interpolated AP from recall/precision points."""
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    area = 0.0
    for index in range(len(mrec) - 1):
        if mrec[index + 1] != mrec[index]:
            area += (mrec[index + 1] - mrec[index]) * mpre[index + 1]
    return area


def _evaluate_class_predictions(
    predictions: list[Prediction],
    gt_by_image: dict[int, Tensor],
    *,
    iou_threshold: float,
) -> tuple[float, list[tuple[float, int, int]], int]:
    """Evaluate one class worth of predictions against one class worth of GT."""
    total_gt = sum(boxes.shape[0] for boxes in gt_by_image.values())
    predictions = sorted(predictions, key=lambda item: item.score, reverse=True)
    if total_gt == 0:
        return 0.0, [], 0

    matched = {
        image_id: torch.zeros(boxes.shape[0], dtype=torch.bool)
        for image_id, boxes in gt_by_image.items()
    }
    true_positives: list[float] = []
    false_positives: list[float] = []
    events: list[tuple[float, int, int]] = []

    # Standard greedy matching: highest-score prediction claims the best
    # available GT box above threshold.
    for prediction in predictions:
        gt_boxes = gt_by_image.get(prediction.image_id)
        if gt_boxes is None or gt_boxes.numel() == 0:
            true_positives.append(0.0)
            false_positives.append(1.0)
            events.append((float(prediction.score), 0, 1))
            continue

        ious = box_iou(prediction.box.unsqueeze(0), gt_boxes)[0].squeeze(0)
        best_iou, best_index = ious.max(dim=0)
        if best_iou.item() >= iou_threshold and not matched[prediction.image_id][best_index]:
            matched[prediction.image_id][best_index] = True
            true_positives.append(1.0)
            false_positives.append(0.0)
            events.append((float(prediction.score), 1, 0))
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)
            events.append((float(prediction.score), 0, 1))

    if not true_positives:
        return 0.0, events, total_gt

    tp_cum = torch.tensor(true_positives).cumsum(0)
    fp_cum = torch.tensor(false_positives).cumsum(0)
    recalls = (tp_cum / max(total_gt, 1)).tolist()
    precisions = (tp_cum / (tp_cum + fp_cum).clamp(min=1e-6)).tolist()
    return _compute_ap(recalls, precisions), events, total_gt


def _evaluate_ap50(
    predictions_by_class: dict[int, list[Prediction]],
    targets_by_class: dict[int, dict[int, Tensor]],
    num_classes: int,
    *,
    iou_threshold: float,
) -> tuple[float, dict[int, float], dict[int, list[tuple[float, int, int]]], dict[int, int], int]:
    """Compute mean AP50 and the per-class event streams used later."""
    per_class_ap: dict[int, float] = {}
    class_events: dict[int, list[tuple[float, int, int]]] = {}
    gt_count_by_class: dict[int, int] = {}
    total_gt_all = 0
    for class_index in range(num_classes):
        gt_by_image = {
            image_id: boxes
            for image_id, boxes in targets_by_class.get(class_index, {}).items()
        }
        ap, events, total_gt = _evaluate_class_predictions(
            predictions_by_class.get(class_index, []),
            gt_by_image,
            iou_threshold=iou_threshold,
        )
        per_class_ap[class_index] = ap
        class_events[class_index] = events
        gt_count_by_class[class_index] = total_gt
        total_gt_all += total_gt

    valid_aps = [value for class_index, value in per_class_ap.items() if targets_by_class.get(class_index)]
    mean_ap = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    return mean_ap, per_class_ap, class_events, gt_count_by_class, total_gt_all


def _build_confidence_curves(
    class_events: dict[int, list[tuple[float, int, int]]],
    *,
    total_gt: int,
) -> dict[str, Any]:
    """Build precision/recall/F1 curves over confidence thresholds."""
    thresholds = [index / 100.0 for index in range(101)]
    all_events = [event for events in class_events.values() for event in events]
    all_events.sort(key=lambda item: item[0], reverse=True)

    precision_desc: list[float] = []
    recall_desc: list[float] = []
    f1_desc: list[float] = []
    tp_running = 0
    fp_running = 0
    event_index = 0

    # We walk thresholds from high to low so running TP/FP counts can be reused.
    for threshold in reversed(thresholds):
        while event_index < len(all_events) and all_events[event_index][0] >= threshold:
            _, tp_value, fp_value = all_events[event_index]
            tp_running += tp_value
            fp_running += fp_value
            event_index += 1
        precision = tp_running / max(tp_running + fp_running, 1)
        recall = tp_running / max(total_gt, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        precision_desc.append(float(precision))
        recall_desc.append(float(recall))
        f1_desc.append(float(f1))

    precision = list(reversed(precision_desc))
    recall = list(reversed(recall_desc))
    f1 = list(reversed(f1_desc))
    best_index = max(range(len(f1)), key=lambda index: (f1[index], precision[index], -thresholds[index]))
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_threshold": float(thresholds[best_index]),
        "best_f1": float(f1[best_index]),
        "precision_at_best_f1": float(precision[best_index]),
        "recall_at_best_f1": float(recall[best_index]),
        "total_gt": int(total_gt),
    }


def _build_per_class_operating_metrics(
    class_events: dict[int, list[tuple[float, int, int]]],
    gt_count_by_class: dict[int, int],
    *,
    class_names: list[str],
    conf_threshold: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, int]]]:
    """Report per-class precision/recall at the selected confidence threshold."""
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_counts: dict[str, dict[str, int]] = {}

    for class_index, total_gt in gt_count_by_class.items():
        if total_gt <= 0:
            continue

        tp_count = 0
        fp_count = 0
        for score, tp_value, fp_value in class_events.get(class_index, []):
            if score < conf_threshold:
                continue
            tp_count += int(tp_value)
            fp_count += int(fp_value)

        label = class_names[class_index] if class_index < len(class_names) else str(class_index)
        recall = tp_count / max(total_gt, 1)
        precision = tp_count / max(tp_count + fp_count, 1)
        per_class_recall[label] = float(recall)
        per_class_precision[label] = float(precision)
        per_class_counts[label] = {
            "tp": int(tp_count),
            "fp": int(fp_count),
            "gt": int(total_gt),
            "pred": int(tp_count + fp_count),
        }

    return per_class_recall, per_class_precision, per_class_counts


def _match_pairs(
    gt_boxes: Tensor,
    pred_boxes: Tensor,
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Greedily match GT and predictions for confusion-matrix accounting."""
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return [], set(), set()

    ious = box_iou(gt_boxes, pred_boxes)[0]
    candidate_indices = torch.nonzero(ious >= iou_threshold, as_tuple=False)
    if candidate_indices.numel() == 0:
        return [], set(), set()

    candidates = sorted(
        (
            float(ious[pair[0], pair[1]].item()),
            int(pair[0].item()),
            int(pair[1].item()),
        )
        for pair in candidate_indices
    )
    candidates.reverse()

    matches: list[tuple[int, int]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index))
    return matches, matched_gt, matched_pred


def _build_confusion_matrix(
    image_records: list[ImageRecord],
    *,
    num_classes: int,
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    """Build a confusion matrix with an explicit background row/column."""
    background_index = num_classes
    labels = list(class_names) if class_names else [str(index) for index in range(num_classes)]
    labels.append("background")
    matrix = [[0 for _ in range(num_classes + 1)] for _ in range(num_classes + 1)]

    for record in image_records:
        keep_mask = record.pred_scores >= conf_threshold
        pred_boxes = record.pred_boxes[keep_mask]
        pred_labels = record.pred_labels[keep_mask]
        matches, matched_gt, matched_pred = _match_pairs(
            record.gt_boxes,
            pred_boxes,
            iou_threshold=iou_threshold,
        )

        for gt_index, pred_index in matches:
            gt_label = int(record.gt_labels[gt_index].item())
            pred_label = int(pred_labels[pred_index].item())
            matrix[gt_label][pred_label] += 1

        for gt_index, gt_label in enumerate(record.gt_labels.tolist()):
            if gt_index not in matched_gt:
                matrix[int(gt_label)][background_index] += 1

        for pred_index, pred_label in enumerate(pred_labels.tolist()):
            if pred_index not in matched_pred:
                matrix[background_index][int(pred_label)] += 1

    normalized: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        if row_total <= 0:
            normalized.append([0.0 for _ in row])
            continue
        normalized.append([float(value / row_total) for value in row])

    return {
        "labels": labels,
        "matrix": matrix,
        "normalized": normalized,
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
    }


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    criterion: nn.Module | None,
    data_loader,
    device: torch.device,
    *,
    topk: int = 300,
    amp_enabled: bool = True,
) -> dict[str, Any]:
    """Run evaluation and return the metrics consumed by monitoring artifacts.

    Expected model interface:
    - ``model(images)`` returns raw outputs
    - ``model.post_process(outputs, image_sizes=..., topk=...)`` returns boxes,
      scores, and labels per image
    - ``model.num_classes`` is available for confusion-matrix sizing
    """
    iou_threshold = 0.5
    model_was_training = model.training
    criterion_was_training = criterion.training if criterion is not None else False
    model.eval()
    if criterion is not None:
        criterion.eval()

    running_loss = 0.0
    loss_steps = 0
    predictions_by_class: dict[int, list[Prediction]] = defaultdict(list)
    targets_by_class: dict[int, dict[int, Tensor]] = defaultdict(dict)
    image_records: list[ImageRecord] = []
    num_classes = getattr(model, "num_classes", 0)

    for batch_index, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = move_targets_to_device(targets, device)

        autocast_context = (
            torch.autocast(
                device_type=device.type,
                enabled=bool(amp_enabled and device.type == "cuda"),
            )
            if device.type in {"cuda", "cpu"}
            else nullcontext()
        )

        # Validation can still compute loss when a criterion is available, but
        # metric generation does not depend on the old trainer implementation.
        with autocast_context:
            outputs = model(images)
            if criterion is not None:
                loss_dict = criterion(outputs, targets)
                total_loss = compute_total_loss(loss_dict)
                running_loss += float(total_loss.detach().item())
                loss_steps += 1

        image_sizes = torch.stack([target["size"] for target in targets], dim=0)
        results = model.post_process(outputs, image_sizes=image_sizes, topk=topk)

        for local_index, (result, target) in enumerate(zip(results, targets)):
            image_id_value = target.get("image_id")
            if isinstance(image_id_value, Tensor):
                image_id = int(image_id_value.item())
            else:
                image_id = batch_index * data_loader.batch_size + local_index

            gt_boxes = target.get("boxes_xyxy")
            if gt_boxes is None:
                gt_boxes = target["boxes"]
            gt_boxes = gt_boxes.detach().cpu()
            gt_labels = target["labels"].detach().cpu()

            for class_index in gt_labels.unique(sorted=True).tolist():
                class_mask = gt_labels == class_index
                targets_by_class[int(class_index)][image_id] = gt_boxes[class_mask]

            pred_boxes = result["boxes"].detach().cpu()
            pred_scores = result["scores"].detach().cpu()
            pred_labels = result["labels"].detach().cpu()
            image_records.append(
                ImageRecord(
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                )
            )

            for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
                predictions_by_class[int(pred_label.item())].append(
                    Prediction(
                        image_id=image_id,
                        score=float(pred_score.item()),
                        box=pred_box,
                    )
                )

    # Aggregate raw predictions into the higher-level metrics expected by the
    # dashboard and monitoring JSON outputs.
    map50, per_class_ap, class_events, gt_count_by_class, total_gt = _evaluate_ap50(
        predictions_by_class,
        targets_by_class,
        num_classes,
        iou_threshold=iou_threshold,
    )

    if model_was_training:
        model.train()
    if criterion is not None and criterion_was_training:
        criterion.train()

    class_names = getattr(getattr(data_loader, "dataset", None), "class_names", None) or []
    named_ap = {
        class_names[class_index] if class_index < len(class_names) else str(class_index): float(ap)
        for class_index, ap in per_class_ap.items()
        if targets_by_class.get(class_index)
    }
    confidence_curves = _build_confidence_curves(class_events, total_gt=total_gt)
    per_class_recall50, per_class_precision50, per_class_counts = _build_per_class_operating_metrics(
        class_events,
        gt_count_by_class,
        class_names=class_names,
        conf_threshold=float(confidence_curves["best_threshold"]),
    )
    confusion_matrix = _build_confusion_matrix(
        image_records,
        num_classes=num_classes,
        class_names=class_names,
        conf_threshold=float(confidence_curves["best_threshold"]),
        iou_threshold=iou_threshold,
    )

    return {
        "loss": running_loss / max(loss_steps, 1) if loss_steps > 0 else 0.0,
        "map50": float(map50),
        "num_eval_images": float(len(data_loader.dataset)),
        "per_class_ap50": named_ap,
        "per_class_recall50": per_class_recall50,
        "per_class_precision50": per_class_precision50,
        "per_class_counts": per_class_counts,
        "confidence_curves": confidence_curves,
        "confusion_matrix": confusion_matrix,
    }


__all__ = ["evaluate_detection"]
