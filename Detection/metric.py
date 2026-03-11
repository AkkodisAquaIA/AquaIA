import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from Detection.utils.box_ops import box_cxcywh_to_xyxy


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
    summary = " | ".join(f"{key}={averaged[key]:.4f}" for key in sorted(averaged))
    print(summary)
    return averaged


@torch.no_grad()
def evaluate_map(model, dataloader, device, num_classes, conf_thres=0.05):
    use_amp = device == "cuda"
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=torch.arange(0.5, 1.0, 0.05).tolist(),
        class_metrics=False,
    ).to(device)

    was_training = model.training
    model.eval()

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        _, _, h, w = images.shape

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)

        pred_boxes = outputs["pred_boxes"].float()
        pred_logits = outputs["pred_logits"].float()
        if pred_logits.shape[-1] > num_classes:
            pred_logits = pred_logits[..., :num_classes]

        scores, labels = pred_logits.sigmoid().max(dim=-1)
        preds = []
        refs = []
        for i in range(images.shape[0]):
            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i]).clamp(0, 1)
            boxes_xyxy[:, [0, 2]] *= w
            boxes_xyxy[:, [1, 3]] *= h
            keep = scores[i] >= conf_thres
            preds.append(
                {
                    "boxes": boxes_xyxy[keep],
                    "scores": scores[i][keep],
                    "labels": labels[i][keep].long(),
                }
            )

            target_boxes_xyxy = box_cxcywh_to_xyxy(targets[i]["boxes"]).clamp(0, 1)
            target_boxes_xyxy[:, [0, 2]] *= w
            target_boxes_xyxy[:, [1, 3]] *= h
            refs.append(
                {
                    "boxes": target_boxes_xyxy,
                    "labels": targets[i]["labels"].long(),
                }
            )

        metric.update(preds, refs)

    computed_metrics = metric.compute()
    if was_training:
        model.train()
    return {
        "map_50": float(computed_metrics["map_50"].item()),
        "map_50_95": float(computed_metrics["map"].item()),
    }
