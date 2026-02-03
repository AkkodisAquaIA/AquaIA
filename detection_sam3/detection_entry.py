from pathlib import Path
from datetime import datetime
import torch
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLOE
from ultralytics.utils.nms import TorchNMS
from dataset_dict import DATASET_DICT
from model_cfg import IMAGES_FOLDER, MODEL_NAME, MODEL_CFG

def post_nms(result, iou_threshold, INFO_NMS):
    """Per-class NMS for detection results; keeps cls/conf, optional masks.
    The terminal logs and auto-saved visualizations during inference
    are still the results before NMS."""
    # Print notice about NMS once
    if INFO_NMS:
        print("\n" + "=" * 100)
        print("The terminal logs and auto-saved visualizations during inference"
          " are still the results before NMS.")
        print("=" * 100 + "\n")
        INFO_NMS = False

    # If no boxes, return as is
    if result.boxes is None or result.boxes.shape[0] == 0:
        return result, INFO_NMS

    # Extract box coordinates, scores, and classes
    bboxes = result.boxes.xyxy
    scores = result.boxes.conf
    classes = result.boxes.cls

    keep_all = []
    # For each unique class id
    for cls_id in classes.unique():
        # idx stores indices mapping this class's boxes to the original results
        idx = (classes == cls_id).nonzero(as_tuple=False).squeeze(1)
        # keep_c contains the indices to be retained within this class subset
        keep_c = TorchNMS.fast_nms(bboxes[idx], scores[idx], iou_threshold=iou_threshold)
        # Map back to the original result indices and store
        keep_all.append(idx[keep_c])

    # Concatenate all kept indices and filter results
    keep = torch.cat(keep_all) if keep_all else torch.empty(0, dtype=torch.long, device=bboxes.device)
    result.boxes = result.boxes[keep]
    if result.masks is not None:
        result.masks = result.masks[keep]

    return result, INFO_NMS

def save_xywh_label(result, img_path: Path, labels_folder: Path, dataset_keys_sorted: list[int]) -> None:
    """Save normalized xywh labels for one image. cx, cy, w, h are normalized by image width and height.
    Args:
        result: The prediction result object containing boxes and original image.
        img_path (Path): Path to the input image.
        labels_folder (Path): Directory to save the label file.
        dataset_keys_sorted (list[int]): List of dataset keys sorted in order.    
    """
    # Always create the label file; keep it empty if no boxes detected
    label_path = labels_folder / f"{Path(img_path).stem}.txt"
    if result.boxes is None or result.boxes.shape[0] == 0:
        label_path.open("w").close()
        return

    xywh = result.boxes.xywh.cpu().numpy()
    cls_idx = result.boxes.cls.cpu().numpy().astype(int)    # text prompt n → cls_idx n
    # Map cls_idx back to dataset key
    coco_ids = [dataset_keys_sorted[i] for i in cls_idx]
    conf = result.boxes.conf.cpu().numpy()
    # Get original image size
    img_h, img_w = result.orig_img.shape[:2]

    coco_bboxes_norm = []
    for cx, cy, w, h in xywh:
        coco_bboxes_norm.append([cx / img_w, cy / img_h, w / img_w, h / img_h])

    # Write to label file
    with label_path.open("w") as f:
        for cid, bbox, score in zip(coco_ids, coco_bboxes_norm, conf):
            f.write(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {score:.6f}\n")

def print_device_info(model_instance):
    """Print the device information of the model instance."""
    global INFO_DEVICE
    if INFO_DEVICE:
        print(f"Device used: {model_instance.device}")
        INFO_DEVICE = False

if __name__ == "__main__":
    # Initialization parameters
    current_folder = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    cfg = MODEL_CFG[MODEL_NAME]
    run_name = f"{MODEL_NAME}_result_det_{timestamp}"

    # Text prompts sourced from the dataset label dictionary (keys sorted for stable order)
    text_prompts = [DATASET_DICT[idx] for idx in sorted(DATASET_DICT.keys())]

    # Map detection index back to dataset key
    dataset_keys_sorted = [idx for idx in sorted(DATASET_DICT.keys())]

    # Create output labels folder
    labels_folder = Path(current_folder) / run_name / "labels"
    labels_folder.mkdir(parents=True, exist_ok=True)

    # Save the loaded configuration as a flat text file for reference
    cfg_path = labels_folder.parent / "cfg.txt"
    cfg_content = {"IMAGES_FOLDER": IMAGES_FOLDER, "MODEL_NAME": MODEL_NAME}
    for key, value in cfg.items():
        cfg_content[key] = value
    with cfg_path.open("w", encoding="utf-8") as cfg_file:
        for key, value in cfg_content.items():
            # Format strings with quotes
            formatted = f"\"{value}\"" if isinstance(value, str) else value
            cfg_file.write(f"{key} = {formatted}\n")

    # Iterate over all images in IMAGES_FOLDER and run inference with the text prompts
    image_files = sorted(f for f in Path(IMAGES_FOLDER).glob("**/*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # Print information once
    INFO_DEVICE = True
    INFO_NMS = True

    if MODEL_NAME == "sam3":
        # Initialize SAM3
        overrides = dict(
            conf=cfg["CONF"],
            task=cfg["TASK"],
            mode=cfg["MODE"],
            model=cfg["PATH"],
            half=cfg["HALF"],
            save=cfg["SAVE"],
            imgsz=cfg["IMGSZ"],
            project=str(current_folder),
            name=run_name,
            exist_ok=True,)
        predictor = SAM3SemanticPredictor(overrides=overrides)

        for img_path in image_files:
            predictor.set_image(str(img_path))

            # Print device info
            print_device_info(predictor)

            # Run prediction
            results = predictor(text=text_prompts)

            # If no results, still create empty txt file
            if not results:
                (labels_folder / f"{img_path.stem}.txt").open("w").close()
                continue

            # Apply NMS only when enabled and keep updated result in-place
            if cfg["NMS"] != False:
                results[0], INFO_NMS = post_nms(results[0], cfg["NMS"], INFO_NMS)

            # Save labels in xywh format
            save_xywh_label(results[0], img_path, labels_folder, dataset_keys_sorted)

    if MODEL_NAME == "yoloe26":
        # Initialize YOLOE26
        model = YOLOE(cfg["PATH"])
        model.set_classes(text_prompts, model.get_text_pe(text_prompts))

        # Print device info
        print_device_info(model)

        # Run prediction
        results = model.predict(
            source=str(IMAGES_FOLDER),
            conf=cfg["CONF"],
            half=cfg["HALF"],
            save=cfg["SAVE"],
            imgsz=cfg["IMGSZ"],
            project=str(current_folder),
            name=run_name,
            exist_ok=True)

        # If no results, skip saving
        if results:
            for result in results:
                # Apply NMS only when enabled and keep updated result in-place
                if cfg["NMS"] != False:
                    result, INFO_NMS = post_nms(result, cfg["NMS"], INFO_NMS)

                # Save labels in xywh format
                img_path = Path(result.path)
                save_xywh_label(result, img_path, labels_folder, dataset_keys_sorted)