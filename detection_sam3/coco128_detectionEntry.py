from pathlib import Path
from datetime import datetime
from ultralytics.models.sam import SAM3SemanticPredictor
from coco128_dict import COCO128_DICT
from coco128_cfg import IMAGES_FOLDER, SAM3_CONF, SAM3_TASK, SAM3_MODE, SAM3_PATH, SAM3_HALF, SAM3_SAVE, SAM3_IMGSZ

# Initialize SAM3 predictor with configuration
current_folder = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y%m%d%H%M")

overrides = dict(
    conf=SAM3_CONF,
    task=SAM3_TASK,
    mode=SAM3_MODE,
    model=SAM3_PATH,
    half=SAM3_HALF,
    save=SAM3_SAVE,
    imgsz=SAM3_IMGSZ,
    project=str(current_folder),
    name=f"result_det_{timestamp}",)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Text prompts sourced from the COCO128 label dictionary (keys sorted for stable order)
text_prompts = [COCO128_DICT[idx] for idx in sorted(COCO128_DICT.keys())]

# Map detection index back to COCO128 key
coco128_keys_sorted = [idx for idx in sorted(COCO128_DICT.keys())]

# Create output labels folder
labels_folder = Path(current_folder) / f"result_det_{timestamp}" / "labels"
labels_folder.mkdir(parents=True, exist_ok=True)

# Save the loaded configuration as a flat text file for reference
cfg_path = labels_folder.parent / "cfg.txt"
cfg_content = {
    "IMAGES_FOLDER": IMAGES_FOLDER,
    "SAM3_CONF": SAM3_CONF,
    "SAM3_TASK": SAM3_TASK,
    "SAM3_MODE": SAM3_MODE,
    "SAM3_PATH": SAM3_PATH,
    "SAM3_HALF": SAM3_HALF,
    "SAM3_SAVE": SAM3_SAVE,
    "SAM3_IMGSZ": SAM3_IMGSZ,
}
with cfg_path.open("w", encoding="utf-8") as cfg_file:
    for key, value in cfg_content.items():
        formatted = f"\"{value}\"" if isinstance(value, str) else value
        cfg_file.write(f"{key} = {formatted}\n")

def save_xywh_label(result, img_path: Path, labels_folder: Path, coco128_keys_sorted: list[int]) -> None:
    """Save normalized xywh labels for one image. cx, cy, w, h are normalized by image width and height.
    Args:
        result: The prediction result object containing boxes and original image.
        img_path (Path): Path to the input image.
        labels_folder (Path): Directory to save the label file.
        coco128_keys_sorted (list[int]): List of COCO128 keys sorted in order.    
    """
    # Return if no boxes detected
    if result.boxes is None:
        return

    xywh = result.boxes.xywh.cpu().numpy()
    cls_idx = result.boxes.cls.cpu().numpy().astype(int)    # text prompt n → cls_idx n
    # Map cls_idx back to COCO128 key
    coco_ids = [coco128_keys_sorted[i] for i in cls_idx]
    conf = result.boxes.conf.cpu().numpy()
    # Get original image size
    img_h, img_w = result.orig_img.shape[:2]

    coco_bboxes_norm = []
    for cx, cy, w, h in xywh:
        coco_bboxes_norm.append([cx / img_w, cy / img_h, w / img_w, h / img_h])

    # Write to label file
    label_path = labels_folder / f"{Path(img_path).stem}.txt"
    with label_path.open("w") as f:
        for cid, bbox, score in zip(coco_ids, coco_bboxes_norm, conf):
            f.write(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {score:.6f}\n")

# Iterate over all images in IMAGES_FOLDER and run inference with the text prompts
image_files = sorted(f for f in Path(IMAGES_FOLDER).glob("**/*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

# Print device information once
INFO_DEVICE = True

for img_path in image_files:
    predictor.set_image(str(img_path))

    if INFO_DEVICE:
        print(f"Device used: {predictor.device}")
        INFO_DEVICE = False
    
    results = predictor(text=text_prompts)

    # If no results, skip saving
    if not results:
        continue

    save_xywh_label(results[0], img_path, labels_folder, coco128_keys_sorted)