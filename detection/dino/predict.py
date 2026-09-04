import torch
from detection.utils.box_ops import box_cxcywh_to_xyxy


def normalize_imgsz(config, phase):
    """Z: Ensure DINO input image size "imgsz" is a multiple of backbone patch size.
    Training or inference phase. Will update config[phase]["imgsz"] if needed."""
    # Z: get config.model.family, ex dinov3
    model_family = str(config.get("model", {}).get("family", "")).lower()
    # Z: not grid size
    patch_size = 14 if model_family == "dinov2" else 16
    # Z: read config[phase]["imgsz"], ex 640
    imgsz = int(config[phase]["imgsz"])
    rounded_imgsz = max(patch_size, round(imgsz / patch_size) * patch_size)
    if rounded_imgsz != imgsz:
        print(f"Warning: imgsz={imgsz} is not divisible by patch size {patch_size}. Using imgsz={rounded_imgsz} instead.")
        # Z: update config[phase]["imgsz"] to rounded_imgsz
        config[phase]["imgsz"] = rounded_imgsz
    return int(config[phase]["imgsz"])


def predict(model, samples, device, conf_thres, imgsz=640):
    """Z: Run inference on a batch of samples (for evaluation or visualization) and return predictions.
    For each image in the batch returns {"boxes": Tensor[N, 4], "scores": Tensor[N], "labels": Tensor[N]}.
    Boxes are in xyxy format with real pixel coords."""
    # Z: samples can be either:
    # - a dict returned by sample_dataset() for visualization
    # - a dataloader batch for metric evaluation.
    # Z: [B, 3, H, W]
    inputs = samples["inputs"]
    inputs = inputs.to(device, non_blocking=True)
    # Z: Run inference with autocast for mixed precision
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device != "cpu"):
        outputs = model(inputs)

    # Z: [B, num_queries, 4]
    pred_boxes = outputs["pred_boxes"].float()
    # Z: [B, num_queries, num_classes]
    pred_logits = outputs["pred_logits"].float()

    _, _, height, width = inputs.shape
    # Z: [B, num_queries], [B, num_queries]
    scores, labels = pred_logits.sigmoid().max(dim=-1)
    preds = []
    for i in range(inputs.shape[0]):
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i]).clamp(0, 1)
        # Z: Scale boxes to original image size
        boxes_xyxy[:, [0, 2]] *= width
        boxes_xyxy[:, [1, 3]] *= height
        # Z: Filter mask
        keep = scores[i] >= conf_thres
        preds.append(
            {
                "boxes": boxes_xyxy[keep],
                "scores": scores[i][keep],
                "labels": labels[i][keep].long(),
            }
        )
    return preds
