import torch
from detection.utils.box_ops import box_cxcywh_to_xyxy

def predict(model, images, device, conf_thres):
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
