import torch


def predict_yolo(model, image_files, device, conf_thres, imgsz):
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