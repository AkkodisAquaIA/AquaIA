import cv2
from pathlib import Path
import yaml
from utils import select_or_latest, collect_image_files, load_label_txt, xywh_norm_to_xyxy_norm

PARENT_FOLDER = Path(__file__).resolve().parent # Folder containing this script
CFG_PATH = PARENT_FOLDER / "model_cfg.yaml"
CFG_DATA = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
IMAGES_FOLDER = CFG_DATA["IMAGES_FOLDER"]
DATASET_DICT_PATH = PARENT_FOLDER / "dataset_dict.yaml"
DATASET_DICT_RAW = yaml.safe_load(DATASET_DICT_PATH.read_text(encoding="utf-8"))
DATASET_DICT = {int(key): value for key, value in DATASET_DICT_RAW.items()}

if __name__ == "__main__":
    # Select result_det folder or automatically use the latest one
    det_dir = select_or_latest(base_dir=PARENT_FOLDER, title="Select result_det folder (Cancel to use latest)")

    # All images in IMAGES_FOLDER (recursive)
    image_files, _ = collect_image_files(IMAGES_FOLDER, stage="cropping")

    crop_count = 0

    # For each image
    for img_path in image_files:
        # Get relative directory to maintain the original folder structure
        rel_dir = img_path.parent.relative_to(Path(IMAGES_FOLDER))   # Image folder name

        # In detection_entry.py, labels are saved in: run_dir (here det_dir) / rel_dir / "labels"
        label_dir = det_dir / rel_dir / "labels"
        label_path = label_dir / f"{img_path.stem}.txt"

        # Skip if label file does not exist
        if not label_path.exists():
            print(f"Warning: Label file not found for image {img_path}")
            continue

        # Load labels from txt without confidence
        labels = load_label_txt(label_path, with_conf=False, conf_threshold=None)

        # If no boxes detected, skip
        if labels.shape[0] == 0:
            continue

        # Read the original image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to read image {img_path}")
            continue

        # Get image dimensions, convert to xyxyn, then to absolute pixel coordinates
        img_h, img_w = img.shape[:2]
        xywhn = labels[:, 1:5]
        xyxyn = xywh_norm_to_xyxy_norm(xywhn)
        x1 = (xyxyn[:, 0] * img_w).astype(int)
        y1 = (xyxyn[:, 1] * img_h).astype(int)
        x2 = (xyxyn[:, 2] * img_w).astype(int)
        y2 = (xyxyn[:, 3] * img_h).astype(int)

        # Get class indices
        classes = labels[:, 0].astype(int)

        # Define and create the specific crop directory maintaining recursive structure
        crop_dir = det_dir / "00crop" / rel_dir
        crop_dir.mkdir(parents=True, exist_ok=True)

        # For each detected box
        for i in range(labels.shape[0]):
            # Clip coordinates to image boundaries to prevent out-of-bounds slicing
            x1_i = max(0, x1[i])
            y1_i = max(0, y1[i])
            x2_i = min(img_w, x2[i])
            y2_i = min(img_h, y2[i])

            # Skip invalid bounding boxes
            if x2_i <= x1_i or y2_i <= y1_i:
                print(f"Warning: Invalid bounding box for image {img_path}, box index {i}")
                continue

            # Map class ID to string name
            cls_name = DATASET_DICT[classes[i]]

            # Crop the image using numpy slicing
            crop_img = img[y1_i:y2_i, x1_i:x2_i]

            # Format: [original]_[class]_[num]
            crop_filename = f"{img_path.stem}_{cls_name}_{i + 1:03d}{img_path.suffix}"
            crop_filepath = crop_dir / crop_filename

            # Save the cropped image
            cv2.imwrite(str(crop_filepath), crop_img)
            crop_count += 1

    print(f"\nCropping completed successfully!")
    print(f"Total {crop_count} bounding boxes cropped and saved in: {det_dir / '00crop'}")