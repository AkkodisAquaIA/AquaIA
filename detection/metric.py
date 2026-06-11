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
    if isinstance(imgsz, (tuple, list)):
        return imgsz[0], imgsz[1]
    return imgsz, imgsz


def _build_refs(targets, imgsz):
    width, height = _image_size_xy(imgsz)
    refs = []
    for target in targets:
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


@torch.no_grad()
def evaluate_map(predictions, targets, imgsz, split, device):
    # TODO : can we reuse the object across epochs/batch ?
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=torch.arange(0.5, 1.0, 0.05).tolist(),
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
    all_metrics = {}
    for loader in dataloaders:
        imgsz = loader.dataset.img_size
        predictions = []
        targets = []
        for batch in loader:
            batch_targets = loader.dataset.get_targets(batch)
            if isinstance(batch, list):
                batch = batch[0]
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
