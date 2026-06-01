import torch

def normalize_imgsz(config, phase):
	# TODO: check image size for YOLO too ?
	return int(config[phase]["imgsz"])


def predict(model, samples, device, conf_thres, imgsz=640):
    image_files = samples["img_paths"]
    results = model.predict(
        source=image_files, 
        conf=conf_thres, 
        device=device, 
        verbose=False, 
        imgsz=imgsz
    )
    preds = []
    for result in results:
        boxes = result.boxes
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
    return preds