import torch


def normalize_imgsz(config, phase):
    # TODO: check image size for YOLO too ?
    return int(config[phase]["imgsz"])


def predict(model, samples, device, conf_thres, imgsz=None):
    """Return pixel xyxy boxes in the resized sample image coordinate system."""
    image_files = samples["img_paths"]
    target_height, target_width = samples["inputs"].shape[-2:]
    if imgsz is None:
        imgsz = (target_height, target_width)
    results = model.predict(source=image_files, conf=conf_thres, device=device, verbose=False, imgsz=imgsz)
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
        # Ultralytics returns coordinates on the original file, whereas the
        # dataset resizes images and evaluation targets to the sample dimensions.
        original_height, original_width = result.orig_shape
        scaled_boxes = boxes.xyxy.to(device=device, dtype=torch.float32).clone()
        scaled_boxes[:, [0, 2]] *= target_width / original_width
        scaled_boxes[:, [1, 3]] *= target_height / original_height
        preds.append(
            {
                "boxes": scaled_boxes,
                "scores": boxes.conf.to(device),
                "labels": boxes.cls.to(device=device, dtype=torch.int64),
            }
        )
    return preds
