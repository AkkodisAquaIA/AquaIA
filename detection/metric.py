import torch
import yaml
from pathlib import Path
import csv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from detection.utils.box_ops import box_cxcywh_to_xyxy


def update_log_dict(log_dict, loss_dict, epoch_loss):
    for key, value in loss_dict.items():
        if key not in log_dict:
            log_dict[key] = value
        else:
            log_dict[key] += value

    log_dict["avg"] += epoch_loss


def log_epoch(log_dict, num_batch):
    inv_batches = 1.0 / max(int(num_batch), 1)
    averaged = {}
    for key, value in log_dict.items():
        if torch.is_tensor(value):
            value = value.item()
        averaged[key] = float(value) * inv_batches
    return averaged


def print_metrics(metrics):
    summary = " | ".join(f"{key}={metrics[key]:.4f}" for key in sorted(metrics) if isinstance(metrics[key], (int, float)))
    print(summary)

def save_metrics(metrics, output_dir):
    print_metrics(metrics)
    with (Path(output_dir) / "inference_metrics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    with (Path(output_dir) / "inference_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "map_50", "map_50_95", "num_samples"])
        writer.writeheader()
        writer.writerow(
            {
                "map_50": metrics["map_50"],
                "map_50_95": metrics["map_50_95"],
                "num_samples": metrics["num_samples"],
            }
        )


def _build_refs(targets, shapes):
    refs = []
    for target, (height, width) in zip(targets, shapes):
        target_boxes_xyxy = box_cxcywh_to_xyxy(target["boxes"]).clamp(0, 1)
        target_boxes_xyxy[:, [0, 2]] *= width
        target_boxes_xyxy[:, [1, 3]] *= height
        refs.append(
            {
                "boxes": target_boxes_xyxy,
                "labels": target["labels"].long(),
            }
        )
    return refs


def _predict_dino(model, images, device, num_classes, conf_thres):
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        outputs = model(images)

    pred_boxes = outputs["pred_boxes"].float()
    pred_logits = outputs["pred_logits"].float()
    if pred_logits.shape[-1] > num_classes:
        pred_logits = pred_logits[..., :num_classes]

    _, _, height, width = images.shape
    scores, labels = pred_logits.sigmoid().max(dim=-1)
    preds = []
    for i in range(images.shape[0]):
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i]).clamp(0, 1)
        boxes_xyxy[:, [0, 2]] *= width
        boxes_xyxy[:, [1, 3]] *= height
        keep = scores[i] >= conf_thres
        preds.append(
            {
                "boxes": boxes_xyxy[keep],
                "scores": scores[i][keep],
                "labels": labels[i][keep].long(),
            }
        )
    return preds


def _predict_yolo(model, image_files, device, conf_thres, imgsz):
    results = model.predict(
        source=image_files, 
        conf=conf_thres, 
        device=device, 
        verbose=False, 
        imgsz=imgsz
    )
    preds = []
    shapes = []
    for result in results:
        boxes = result.boxes
        shapes.append(result.orig_shape)
        if boxes is None:
            preds.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.empty((0,), dtype=torch.float32, device=device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=device),
                }
            )
            continue
        preds.append(
            {
                "boxes": boxes.xyxy.to(device),
                "scores": boxes.conf.to(device),
                "labels": boxes.cls.to(device=device, dtype=torch.int64),
            }
        )
    return preds, shapes


@torch.no_grad()
def evaluate_map(model, dataloader, device, num_classes, conf_thresh):
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=torch.arange(0.5, 1.0, 0.05).tolist(),
        class_metrics=False,
    ).to(device)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    is_yolo_model = model.__class__.__name__.lower().startswith("yolo")

    for batch in dataloader:
        images, targets, image_files = batch
        images = images.to(device, non_blocking=True)
        _, _, height, width = images.shape
        if is_yolo_model:
            preds, shapes = _predict_yolo(model, image_files, device, conf_thresh, imgsz=height)
        else:
            preds = _predict_dino(model, images, device, num_classes, conf_thresh)
            shapes = [(height, width)] * len(targets)
        refs = _build_refs(targets, shapes)

        metric.update(preds, refs)

    computed_metrics = metric.compute()
    if was_training and hasattr(model, "train"):
        model.train()
    return {
        "map_50": float(computed_metrics["map_50"].item()),
        "map_50_95": float(computed_metrics["map"].item()),
    }
