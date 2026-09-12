import torch
import yaml
from pathlib import Path
import csv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloading.datasets import parse_batch
from detection.utils.box_ops import box_cxcywh_to_xyxy

# Internal names --> display names
LOSS_DISPLAY_NAMES = {
    "loss_ce": "cls",
    "loss_bbox": "bbox",
    "loss_giou": "giou",
    "class_error": "class_err",
    "cardinality_error": "count_err",
}


def update_metric_dict(log_dict, loss_dict, batch_loss, split, num_batches):
    """Accumulate per-batch losses into epoch-average metrics for this split, for training.
    Args:
        log_dict (dict): Dictionary to store accumulated metrics for one epoch.
        loss_dict (dict): Dictionary containing per-batch loss values.
        batch_loss (float): The total loss for the current batch = sum of individual losses * weights
        split (str): The data split.
        num_batches (int): Total number of batches in the epoch.
    """
    # Weight for each batch losses' values
    div = 1 / num_batches
    for key, value in loss_dict.items():
        if key not in log_dict[split]:
            log_dict[split][key] = value * div
        else:
            log_dict[split][key] += value * div
    log_dict[split]["loss"] += batch_loss * div


def print_metrics(metrics):
    """Print epoch summary metrics. For training and inference, metrics looks like
    { "train": {"loss": ..., "loss_ce": ..., "loss_bbox": ..., "loss_giou": ...,},
      "val": {"loss": ..., "loss_ce": ..., "loss_bbox": ..., "loss_giou": ...,},
      "epoch": 1 }"""
    print("-" * 5 + " Epoch summary " + "-" * 5)
    for split, loss_dict in metrics.items():
        # Skip items like "epoch" that are not loss dicts
        if not isinstance(loss_dict, dict):
            continue
        # Create summary string for this split
        print_summary = f" ■  {split:<5} : "
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                value = value.item()
            if not isinstance(value, (int, float)):
                continue
            # Format metric names and values for printing
            display_name = LOSS_DISPLAY_NAMES.get(key, key)
            print_summary += f"{display_name}={value:.4f} | "
        # Do not print latest " | "
        print(print_summary[:-3])
    print()


def save_metrics(metrics, output_dir):
    """Save inference metrics with splits to yaml and csv files in output_dir."""
    print_metrics(metrics)
    with (Path(output_dir) / "inference_metrics.yaml").open("w", encoding="utf-8") as f:
        # Write metrics dict to yaml file without sorted keys
        yaml.safe_dump(metrics, f, sort_keys=False)

    with (Path(output_dir) / "inference_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        # Create a CSV writer with the specified fieldnames
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


@torch.no_grad()
def evaluate_map(predictions, targets, imgsz, split, device):
    """Compute mAP50 and mAP50_95 for a given split during training and inference.
    predictions xyxy pixel coords. target boxes cxcywh."""
    # Create a mAP calculator
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=torch.arange(0.5, 1.0, 0.05).tolist(),
        # No single class mAP, only overall
        class_metrics=False,
    ).to(device)

    # Formate input image size to (width, height)
    if isinstance(imgsz, (tuple, list)):
        width, height = imgsz[0], imgsz[1]
    else:
        width, height = imgsz, imgsz

    # Build reference targets for mAP. Convert target boxes to xyxy and scale to pixel coordinates.
    refs = []
    for target in targets:
        # Gather bbox cxcywh then convert to xyxy then clip to 0~1
        target_boxes_xyxy = box_cxcywh_to_xyxy(target["boxes"]).clamp(0, 1)
        # Convert xs to real pixel coords
        target_boxes_xyxy[:, [0, 2]] *= width
        # Convert ys to real pixel coords
        target_boxes_xyxy[:, [1, 3]] *= height
        refs.append(
            {
                "boxes": target_boxes_xyxy,
                "labels": target["labels"].long(),
            }
        )
    # Feed calculator
    metric.update(predictions, refs)
    computed_metrics = metric.compute()
    return {
        f"{split}_map_50": float(computed_metrics["map_50"].item()),
        f"{split}_map_50_95": float(computed_metrics["map"].item()),
    }


@torch.no_grad()
def compute_metrics(model, dataloaders, predict_fn, device, conf_thresh):
    """Compute mAP50 mAP50_95 metrics for a model on given dataloaders using a prediction function. For training and inference."""
    all_metrics = {}
    # Each split has its dataloader
    for loader in dataloaders:
        imgsz = loader.dataset.img_size
        predictions = []
        targets = []
        # For each batch
        for batch in loader:
            _, batch_targets = parse_batch(batch, device=device)
            batch_preds = predict_fn(
                model=model,
                samples=batch,
                device=device,
                conf_thres=conf_thresh,
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
