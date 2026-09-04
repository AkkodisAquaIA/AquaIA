import torch
import yaml
from pathlib import Path
import csv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloading.datasets import parse_batch
from detection.utils.box_ops import box_cxcywh_to_xyxy

# Z: internal names --> display names
# Z: !Warning! avg not used
LOSS_DISPLAY_NAMES = {
    "avg": "loss",
    "loss_ce": "cls",
    "loss_bbox": "bbox",
    "loss_giou": "giou",
    "class_error": "class_err",
    "cardinality_error": "count_err",
}

# Z: display order for metrics
# Z: !Warning! not used
METRIC_ORDER = (
    "avg",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "class_error",
    "cardinality_error",
)


def update_metric_dict(log_dict, loss_dict, batch_loss, split, num_batches):
    """Z: accumulate per-batch losses into epoch-average metrics for this split, for training.
    Args:
        log_dict (dict): Dictionary to store accumulated metrics for one epoch.
        loss_dict (dict): Dictionary containing per-batch loss values.
        batch_loss (float): The total loss for the current batch = sum of individual losses * weights
        split (str): The data split.
        num_batches (int): Total number of batches in the epoch.
    """
    # Z: weight for each batch losses' values
    div = 1 / num_batches
    for key, value in loss_dict.items():
        if key not in log_dict[split]:
            log_dict[split][key] = value * div
        else:
            log_dict[split][key] += value * div

    log_dict[split]["loss"] += batch_loss * div


def _format_metric(metric_name, value):
    """Z: format metric names and values for printing."""
    display_name = LOSS_DISPLAY_NAMES.get(metric_name, metric_name)
    return f"{display_name}={value:.4f}"


def print_metrics(metrics):
    """Z: Print epoch summary metrics. For training and inference, metrics looks like
    { "train": {"loss": ..., "loss_ce": ..., "loss_bbox": ..., "loss_giou": ...,},
      "val": {"loss": ..., "loss_ce": ..., "loss_bbox": ..., "loss_giou": ...,},
      "epoch": 1 }"""
    print("-" * 5 + " Epoch summary " + "-" * 5)

    for split, loss_dict in metrics.items():
        # Z: skip items like "epoch" that are not loss dicts
        if not isinstance(loss_dict, dict):
            continue
        # Z: create summary string for this split
        print_summary = f" ■  {split:<5} : "
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                value = value.item()
            if not isinstance(value, (int, float)):
                continue

            # Z: format metric names and values for printing
            print_summary += f"{_format_metric(key, value)} | "

        print(print_summary[:-3])

    print()


def save_metrics(metrics, output_dir):
    """Z: Save inference metrics with splits to yaml and csv files in output_dir."""
    print_metrics(metrics)
    with (Path(output_dir) / "inference_metrics.yaml").open("w", encoding="utf-8") as f:
        # Z: write metrics dict to yaml file without sorted keys
        yaml.safe_dump(metrics, f, sort_keys=False)

    with (Path(output_dir) / "inference_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        # Z: create a CSV writer with the specified fieldnames
        writer = csv.DictWriter(f, fieldnames=["split", "map_50", "map_50_95"])
        writer.writeheader()
        for key in sorted(metrics):
            if not key.endswith("_map_50"):
                continue
            split = key.removesuffix("_map_50")
            writer.writerow(
                {
                    "split": split,
                    "map_50": metrics[key],
                    "map_50_95": metrics[f"{split}_map_50_95"],
                }
            )


def _image_size_xy(imgsz):
    """Z: formate input image size to (width, height)"""
    if isinstance(imgsz, (tuple, list)):
        return imgsz[0], imgsz[1]
    return imgsz, imgsz


def _build_refs(targets, imgsz):
    """Z: Build reference targets for mAP. Convert target boxes to xyxy and scale to pixel coordinates."""
    width, height = _image_size_xy(imgsz)
    refs = []
    for target in targets:
        # Z: gather bbox cxcywh then convert to xyxy then clip to 0~1
        target_boxes_xyxy = box_cxcywh_to_xyxy(target["boxes"]).clamp(0, 1)
        # Z: convert xs to real pixel coords
        target_boxes_xyxy[:, [0, 2]] *= width
        # Z: convert ys to real pixel coords
        target_boxes_xyxy[:, [1, 3]] *= height
        refs.append(
            {
                "boxes": target_boxes_xyxy,
                "labels": target["labels"].long(),
            }
        )
    return refs


@torch.no_grad()
def evaluate_map(predictions, targets, imgsz, split, device):
    """Z: Compute mAP50 mAP50_95 metrics for a given split using predictions and targets. For training and inference."""
    # TODO : can we reuse the object across epochs/batch ?
    # Z: create a mAP calculator
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=torch.arange(0.5, 1.0, 0.05).tolist(),
        # Z: no single class mAP, only overall
        class_metrics=False,
    ).to(device)

    refs = _build_refs(targets, imgsz)
    metric.update(predictions, refs)
    computed_metrics = metric.compute()
    return {
        f"{split}_map_50": float(computed_metrics["map_50"].item()),
        f"{split}_map_50_95": float(computed_metrics["map"].item()),
    }


@torch.no_grad()
def compute_metrics(model, dataloaders, predict_fn, device, conf_thresh):
    """Z: Compute mAP50 mAP50_95 metrics for a model on given dataloaders using a prediction function. For training and inference."""
    all_metrics = {}
    # Z: each split has its dataloader
    for loader in dataloaders:
        imgsz = loader.dataset.img_size
        predictions = []
        targets = []
        # Z: for each batch
        for batch in loader:
            if isinstance(batch, list):
                batch = batch[0]
            _, batch_targets = parse_batch(batch, device=device)
            batch_preds = predict_fn(
                model=model,
                samples=batch,
                device=device,
                conf_thres=conf_thresh,
                imgsz=imgsz,
            )
            predictions.extend(batch_preds)
            targets.extend(batch_targets)

        metrics = evaluate_map(
            predictions=predictions,
            targets=targets,
            imgsz=imgsz,
            split=loader.dataset.data_split,
            device=device,
        )
        all_metrics.update(metrics)
    return all_metrics
