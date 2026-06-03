import torch
from detection.utils.box_ops import box_cxcywh_to_xyxy

def normalize_imgsz(config, phase):
	model_family = str(config.get("model", {}).get("family", "")).lower()
	patch_size = 14 if model_family == "dinov2" else 16
	imgsz = int(config[phase]["imgsz"])
	rounded_imgsz = max(patch_size, round(imgsz / patch_size) * patch_size)
	if rounded_imgsz != imgsz:
		print(f"Warning: imgsz={imgsz} is not divisible by patch size {patch_size}. Using imgsz={rounded_imgsz} instead.")
		config[phase]["imgsz"] = rounded_imgsz
	return int(config[phase]["imgsz"])

def predict(model, samples, device, conf_thres, imgsz=640):
    images = samples["inputs"]
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        outputs = model(images)

    pred_boxes = outputs["pred_boxes"].float()
    pred_logits = outputs["pred_logits"].float()

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
