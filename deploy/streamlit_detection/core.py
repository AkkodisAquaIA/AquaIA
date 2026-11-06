from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml


BOOLEAN_FILTERS = (
    "roboflow_prefinetuning",
    "lpl_finetuning",
    "yolo_augmentation",
    "copy_paste",
)


def load_catalog(path: Path, project_root: Path) -> list[dict]:
    """Load the model registry and resolve repository-relative artifact paths."""
    with path.open("r", encoding="utf-8") as catalog_file:
        payload = yaml.safe_load(catalog_file) or {}

    models = payload.get("models", [])
    for model in models:
        for key in ("checkpoint", "run_dir", "test_metrics"):
            value = model.get(key)
            if value:
                candidate = Path(value)
                model[key] = candidate if candidate.is_absolute() else project_root / candidate
    return models


def filter_models(models: list[dict], choices: dict) -> list[dict]:
    """Return models whose architecture, resolution, and experiment flags match."""
    matches = []
    for model in models:
        if model.get("architecture") != choices.get("architecture"):
            continue
        if int(model.get("imgsz", 0)) != int(choices.get("imgsz", 0)):
            continue
        if any(bool(model.get(key, False)) != bool(choices.get(key, False)) for key in BOOLEAN_FILTERS):
            continue
        matches.append(model)
    return matches


def summarize_training_run(run_dir: Path | None) -> dict | None:
    """Return validation metrics from the epoch with the largest mAP50-95."""
    if run_dir is None:
        return None
    results_path = Path(run_dir) / "results.csv"
    if not results_path.is_file():
        return None

    with results_path.open("r", encoding="utf-8", newline="") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        return None

    def number(row: dict, key: str) -> float:
        return float(row[key].strip())

    required = {
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }
    if not required.issubset(rows[0]):
        return None

    best = max(rows, key=lambda row: number(row, "metrics/mAP50-95(B)"))
    return {
        "Meilleure époque": int(number(best, "epoch")),
        "Val précision": number(best, "metrics/precision(B)"),
        "Val rappel": number(best, "metrics/recall(B)"),
        "Val mAP50": number(best, "metrics/mAP50(B)"),
        "Val mAP50-95": number(best, "metrics/mAP50-95(B)"),
    }


def load_metrics(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    with Path(path).open("r", encoding="utf-8") as metrics_file:
        return yaml.safe_load(metrics_file) or None


def parse_yolo_labels(contents: str, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized YOLO labels to class ids and pixel xyxy boxes."""
    labels = []
    boxes = []
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Ligne {line_number}: cinq valeurs YOLO sont attendues.")
        class_id, x_center, y_center, width, height = map(float, fields)
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height
        labels.append(int(class_id))
        boxes.append([x1, y1, x2, y2])
    return np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise IoU for two xyxy box arrays."""
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0, None), axis=1)
    second_area = np.prod(np.clip(second[:, 2:] - second[:, :2], 0, None), axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def image_detection_metrics(
    predicted_labels: np.ndarray,
    predicted_boxes: np.ndarray,
    predicted_scores: np.ndarray,
    target_labels: np.ndarray,
    target_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """Greedily match predictions to one image's labels and compute P/R/F1."""
    order = np.argsort(-predicted_scores)
    ious = box_iou(predicted_boxes, target_boxes)
    matched_targets: set[int] = set()
    matched_ious = []
    true_positives = 0

    for prediction_index in order:
        candidates = [
            target_index
            for target_index in range(len(target_boxes))
            if target_index not in matched_targets
            and int(target_labels[target_index]) == int(predicted_labels[prediction_index])
            and ious[prediction_index, target_index] >= iou_threshold
        ]
        if not candidates:
            continue
        target_index = max(candidates, key=lambda index: ious[prediction_index, index])
        matched_targets.add(target_index)
        matched_ious.append(float(ious[prediction_index, target_index]))
        true_positives += 1

    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(target_boxes) - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Vrais positifs": true_positives,
        "Faux positifs": false_positives,
        "Objets manqués": false_negatives,
        "Précision image": precision,
        "Rappel image": recall,
        "F1 image": f1,
        "IoU moyen des correspondances": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }
