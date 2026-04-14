from pathlib import Path
from datetime import datetime
import shutil
import yaml
import torch
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLOE
from ultralytics.utils.nms import TorchNMS
from utils import to_long_path, collect_image_files

PARENT_FOLDER = Path(__file__).resolve().parent # Folder containing this script
CFG_PATH = PARENT_FOLDER / "model_cfg.yaml"
CFG_DATA = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
IMAGES_FOLDER = CFG_DATA["IMAGES_FOLDER"]
MODEL_NAME = CFG_DATA["MODEL_NAME"]
MODEL_CFG = CFG_DATA["MODEL_CFG"]
DATASET_DICT_PATH = PARENT_FOLDER / "dataset_dict.yaml"
DATASET_DICT_RAW = yaml.safe_load(DATASET_DICT_PATH.read_text(encoding="utf-8"))
DATASET_DICT = {int(key): value for key, value in DATASET_DICT_RAW.items()}

def post_nms_cls(result, iou_threshold):
    """Per-class NMS for detection results; keeps cls/conf, optional masks.
    Args:
        result: The prediction result object (1 image) containing boxes, scores, classes, and optional masks.
        iou_threshold: IoU threshold for NMS.
    """
    # If no boxes, return as is
    if result.boxes is None or result.boxes.shape[0] == 0:
        return result

    # Extract box coordinates, scores, and classes
    bboxes = result.boxes.xyxy
    scores = result.boxes.conf
    classes = result.boxes.cls

    keep_all = []
    # For each unique class id
    for cls_id in classes.unique():
        # Find indices (orginal) of all predicted boxes where class == cls_id.
        idx = (classes == cls_id).nonzero(as_tuple=False).squeeze(1)
        # keep_c contains the indices (interior of idx) to be retained within this class subset
        keep_c = TorchNMS.fast_nms(bboxes[idx], scores[idx], iou_threshold=iou_threshold)
        # Map back to the original result indices and store
        keep_all.append(idx[keep_c])

    # Concatenate all kept indices and filter results
    keep = torch.cat(keep_all) if keep_all else torch.empty(0, dtype=torch.long, device=bboxes.device)
    result.boxes = result.boxes[keep]
    if result.masks is not None:
        result.masks = result.masks[keep]

    return result

def post_nms_glb(result, iou_threshold):
    """Global NMS for detection results; keeps cls/conf, optional masks.
    Args:
        result: The prediction result object (1 image) containing boxes, scores, classes, and optional masks.
        iou_threshold: IoU threshold for NMS.
    """
    # If no boxes, return as is
    if result.boxes is None or result.boxes.shape[0] == 0:
        return result

    # Extract box coordinates and scores
    bboxes = result.boxes.xyxy
    scores = result.boxes.conf

    # Run NMS on all boxes regardless of class
    keep = TorchNMS.fast_nms(bboxes, scores, iou_threshold=iou_threshold)
    result.boxes = result.boxes[keep]
    if result.masks is not None:
        result.masks = result.masks[keep]

    return result

def post_unic(result):
    """Keep only the highest-confidence bbox for one image."""
    # If no boxes, return as is
    if result.boxes is None or result.boxes.shape[0] == 0:
        return result

    # Find the index of the box with the highest confidence score
    keep = result.boxes.conf.argmax().reshape(1)

    # Filter results to keep only the highest-confidence box
    result.boxes = result.boxes[keep]

    # If masks exist, keep the corresponding mask
    if result.masks is not None:
        result.masks = result.masks[keep]

    return result

def save_xywh_label(result, img_path: Path, labels_folder: Path, dataset_keys_sorted: list[int]) -> None:
    """Save normalized xywh labels for 1 image. cx, cy, w, h are normalized by image width and height.
    Args:
        result: The prediction result object (1 image) containing boxes and original image.
        img_path (Path): Path to the input image.
        labels_folder (Path): Directory to save the label file.
        dataset_keys_sorted (list[int]): List of dataset keys sorted in order.
    Note:
        Class values in result.boxes.cls are prompt-order indices,
        which may be different from the actual keys from dataset_dict.yaml,
        dataset_keys_sorted provides this index-to-key mapping.
    """
    # Always create the label file; keep it empty if no boxes detected
    label_path = labels_folder / f"{Path(img_path).stem}.txt"
    if result.boxes is None or result.boxes.shape[0] == 0:
        open(to_long_path(label_path), "w").close()
        return

    xywh = result.boxes.xywh.cpu().numpy()
    cls_idx = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()

    # Map cls_idx back to dataset key
    dataset_ids = [dataset_keys_sorted[i] for i in cls_idx]

    # Get original image size
    img_h, img_w = result.orig_img.shape[:2]

    dataset_bboxes_norm = []
    for cx, cy, w, h in xywh:
        dataset_bboxes_norm.append([cx / img_w, cy / img_h, w / img_w, h / img_h])

    # Write to label file
    with open(to_long_path(label_path), "w") as f:
        for cid, bbox, score in zip(dataset_ids, dataset_bboxes_norm, conf):
            f.write(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {score:.6f}\n")

def print_device_info(model_instance, info_device: bool) -> bool:
    """Print the device information of the model instance. Only prints once."""
    if info_device:
        print(f"\nDevice used: {model_instance.device}")
        return False
    return info_device

if __name__ == "__main__":
    # Initialization parameters
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    cfg = MODEL_CFG[MODEL_NAME]
    run_name = f"{MODEL_NAME}_result_det_{timestamp}"
    run_dir = Path(PARENT_FOLDER) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Folder to save non-visualisation files
    (run_dir / "docs_run").mkdir(parents=True, exist_ok=True)

    # Copy config files to run_dir for record-keeping
    shutil.copy2(CFG_PATH, run_dir / "docs_run" / CFG_PATH.name)
    shutil.copy2(DATASET_DICT_PATH, run_dir / "docs_run" / DATASET_DICT_PATH.name)

    # Text prompts (string) sourced from the dataset dictionary (sorted for stable order)
    text_prompts = [DATASET_DICT[idx] for idx in sorted(DATASET_DICT.keys())]

    # Sorted dataset keys used for mapping detection index back to dataset key
    dataset_keys_sorted = [idx for idx in sorted(DATASET_DICT.keys())]

    # All images in IMAGES_FOLDER (recursive)
    image_files, total_images = collect_image_files(IMAGES_FOLDER, stage="detection")

    # Print information once
    info_device = True
    if cfg["NMS_CLS"] != False or cfg["NMS_GLB"] != False or cfg["UNIC"] == True:
        print("\n" + "=" * 100)
        print("The terminal logs during inference are still the results before custom post-processing (NMS_CLS, NMS_GLB, or UNIC).")
        print("=" * 100 + "\n")

    if MODEL_NAME == "sam3":
        # Initialize SAM3
        overrides = dict(
            conf=cfg["CONF"],
            task=cfg["TASK"],
            mode=cfg["MODE"],
            model=cfg["PATH"],
            half=cfg["HALF"],
            save=False,
            imgsz=cfg["IMGSZ"],
            project=str(PARENT_FOLDER),
            name=run_name,
            exist_ok=True,)
        predictor = SAM3SemanticPredictor(overrides=overrides)

        # For each image
        for index, img_path in enumerate(image_files, start=1):
            rel_dir = img_path.parent.relative_to(Path(IMAGES_FOLDER))   # Image folder name
            vis_dir = run_dir / "detection_result" / rel_dir
            label_dir = vis_dir / "labels"
            Path(to_long_path(label_dir)).mkdir(parents=True, exist_ok=True)
            predictor.set_image(str(img_path))

            # Print device info (only once)
            info_device = print_device_info(predictor, info_device)

            # Run prediction
            results = predictor(text=text_prompts)

            # Apply per-class NMS only when enabled and keep updated result in-place
            if cfg["NMS_CLS"] != False:
                results[0] = post_nms_cls(results[0], cfg["NMS_CLS"])

            # Apply global NMS only when enabled and keep updated result in-place
            if cfg["NMS_GLB"] != False:
                results[0] = post_nms_glb(results[0], cfg["NMS_GLB"])

            # Keep only the highest-confidence bbox when UNIC is enabled
            if cfg["UNIC"] == True:
                results[0] = post_unic(results[0])

            # Save labels in xywh format
            save_xywh_label(results[0], img_path, label_dir, dataset_keys_sorted)

            # Save visualization 
            if cfg["SAVE"]:
                results[0].save(filename=str(vis_dir / img_path.name))

            # Print progress every 100 images (and at the end)
            if index % 20 == 0 or index == total_images:
                print(f"\nSAM3 progress: {index}/{total_images} images processed.\n")

    if MODEL_NAME == "yoloe26":
        # Initialize YOLOE26
        model = YOLOE(cfg["PATH"])
        model.set_classes(text_prompts, model.get_text_pe(text_prompts))

        # Write all image paths to a txt file for efficient processing
        imgpath_dir = run_dir / "docs_run" / "yolo_imgpath.txt"
        with imgpath_dir.open("w", encoding="utf-8") as f:
            for img_path in image_files:
                f.write(f"{img_path}\n")

        results = model.predict(
            source=str(imgpath_dir),
            stream=True,    # Stream results for memory efficiency with large datasets; process one by one
            conf=cfg["CONF"],
            half=cfg["HALF"],
            save=False,
            imgsz=cfg["IMGSZ"],
            batch=cfg["BATCH"],
            project=str(PARENT_FOLDER),
            name=run_name,
            exist_ok=True)

        # Run prediction
        for result in results:
            img_path = Path(result.path)
            rel_dir = img_path.parent.relative_to(Path(IMAGES_FOLDER))
            vis_dir = run_dir / "detection_result" / rel_dir
            label_dir = vis_dir / "labels"
            Path(to_long_path(label_dir)).mkdir(parents=True, exist_ok=True)

            # Print device info (only once)
            info_device = print_device_info(model, info_device)

            # Apply per-class NMS only when enabled and keep updated result in-place
            if cfg["NMS_CLS"] != False:
                result = post_nms_cls(result, cfg["NMS_CLS"])

            # Apply global NMS only when enabled and keep updated result in-place
            if cfg["NMS_GLB"] != False:
                result = post_nms_glb(result, cfg["NMS_GLB"])

            # Keep only the highest-confidence bbox when UNIC is enabled
            if cfg["UNIC"] == True:
                result = post_unic(result)

            # Save labels in xywh format
            save_xywh_label(result, img_path, label_dir, dataset_keys_sorted)

            # Save visualization
            if cfg["SAVE"]:
                result.save(filename=str(vis_dir / img_path.name))