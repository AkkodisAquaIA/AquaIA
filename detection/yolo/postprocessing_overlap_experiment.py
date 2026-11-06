"""Evaluate overlap-aware post-processing without tuning on the test split.

The experiment selects a post-processing rule on the validation split, then
reports that fixed rule on the test split. Predictions are first produced with
the regular YOLO NMS (IoU 0.5). Two hypotheses are compared:

* a second, more aggressive NMS suppresses duplicate boxes;
* overlapping boxes are merged into their enclosing union box.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from ultralytics import YOLO

from dataloading.datasets import JpgDetectionDataset, detection_collate_fn, parse_batch
from detection.metric import evaluate_map
from detection.yolo.predict import predict


NMS_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40)
INTERSECTION_OVER_SMALLER_THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)


def _clone_prediction(prediction):
    return {key: value.clone() for key, value in prediction.items()}


def apply_nms(prediction, iou_threshold):
    if len(prediction["boxes"]) < 2:
        return _clone_prediction(prediction)
    keep = nms(prediction["boxes"], prediction["scores"], iou_threshold)
    return {key: value[keep] for key, value in prediction.items()}


def _intersection_over_smaller(boxes):
    if len(boxes) == 0:
        return torch.empty((0, 0), device=boxes.device)
    top_left = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    areas = ((boxes[:, 2:] - boxes[:, :2]).clamp(min=0)).prod(dim=1)
    smaller = torch.minimum(areas[:, None], areas[None, :]).clamp(min=1e-12)
    return intersection / smaller


def merge_overlapping(prediction, threshold):
    """Merge connected same-class boxes whose overlap/min(area) reaches threshold."""
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    if len(boxes) < 2:
        return _clone_prediction(prediction)

    overlap = _intersection_over_smaller(boxes)
    adjacency = overlap >= threshold
    adjacency.fill_diagonal_(True)
    adjacency &= labels[:, None] == labels[None, :]

    visited = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    merged_boxes = []
    merged_scores = []
    merged_labels = []
    for start in range(len(boxes)):
        if visited[start]:
            continue
        component = torch.zeros_like(visited)
        frontier = torch.zeros_like(visited)
        frontier[start] = True
        while frontier.any():
            component |= frontier
            neighbours = adjacency[frontier].any(dim=0)
            frontier = neighbours & ~component
        visited |= component
        component_boxes = boxes[component]
        component_scores = scores[component]
        best = component_scores.argmax()
        merged_boxes.append(
            torch.cat((component_boxes[:, :2].amin(dim=0), component_boxes[:, 2:].amax(dim=0)))
        )
        merged_scores.append(component_scores[best])
        merged_labels.append(labels[component][best])

    return {
        "boxes": torch.stack(merged_boxes),
        "scores": torch.stack(merged_scores),
        "labels": torch.stack(merged_labels),
    }


@torch.no_grad()
def collect_predictions(model, dataset_root, split, imgsz, img_format, device):
    dataset = JpgDetectionDataset(
        dataset_root=dataset_root,
        data_split=split,
        img_size=imgsz,
        img_format=img_format,
    )
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    batch = next(iter(loader))
    _, targets = parse_batch(batch, device=device)
    predictions = predict(
        model=model,
        samples=batch,
        device=device,
        conf_thres=0.05,
        imgsz=imgsz,
        iou=0.5,
    )
    return predictions, targets


def evaluate(predictions, targets, imgsz, split, method, threshold):
    if method == "baseline":
        processed = [_clone_prediction(prediction) for prediction in predictions]
    elif method == "nms":
        processed = [apply_nms(prediction, threshold) for prediction in predictions]
    elif method == "merge":
        processed = [merge_overlapping(prediction, threshold) for prediction in predictions]
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = evaluate_map(processed, targets, imgsz=imgsz, split=split, device=predictions[0]["boxes"].device)
    return {
        "split": split,
        "method": method,
        "threshold": "" if threshold is None else threshold,
        "boxes_before": sum(len(prediction["boxes"]) for prediction in predictions),
        "boxes_after": sum(len(prediction["boxes"]) for prediction in processed),
        "precision": metrics[f"{split}_precision"],
        "recall": metrics[f"{split}_recall"],
        "f1": metrics[f"{split}_f1"],
        "map50": metrics[f"{split}_map_50"],
        "map50_95": metrics[f"{split}_map_50_95"],
    }


def experiment_rows(predictions, targets, imgsz, split):
    rows = [evaluate(predictions, targets, imgsz, split, "baseline", None)]
    rows.extend(evaluate(predictions, targets, imgsz, split, "nms", value) for value in NMS_THRESHOLDS)
    rows.extend(
        evaluate(predictions, targets, imgsz, split, "merge", value)
        for value in INTERSECTION_OVER_SMALLER_THRESHOLDS
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/LPL_dataset_final/dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--img-format", default="TIF")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.checkpoint)).to(device)
    args.output.mkdir(parents=True, exist_ok=True)

    validation_predictions, validation_targets = collect_predictions(
        model, args.dataset, "val", args.imgsz, args.img_format, device
    )
    validation_rows = experiment_rows(validation_predictions, validation_targets, args.imgsz, "val")
    # Keep mAP50-95 as the primary objective. If several rules have exactly the
    # same AP, prefer the one with the best F1, i.e. the fewest duplicate/missed
    # detections at the configured confidence threshold.
    best_validation = max(
        validation_rows,
        key=lambda row: (
            row["map50_95"],
            row["map50"],
            row["f1"],
            row["boxes_after"],
            float(row["threshold"]) if row["method"] == "nms" else 0.0,
        ),
    )

    test_predictions, test_targets = collect_predictions(
        model, args.dataset, "test", args.imgsz, args.img_format, device
    )
    baseline_test = evaluate(test_predictions, test_targets, args.imgsz, "test", "baseline", None)
    selected_test = evaluate(
        test_predictions,
        test_targets,
        args.imgsz,
        "test",
        best_validation["method"],
        None if best_validation["threshold"] == "" else float(best_validation["threshold"]),
    )

    fieldnames = list(validation_rows[0])
    with (args.output / "validation_sweep.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_rows)
    with (args.output / "selected_test_result.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows((baseline_test, selected_test))

    print("Best validation rule:", best_validation)
    print("Test baseline:", baseline_test)
    print("Test selected rule:", selected_test)


if __name__ == "__main__":
    main()
