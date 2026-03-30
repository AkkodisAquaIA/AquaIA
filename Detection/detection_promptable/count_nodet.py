import yaml
import csv
from pathlib import Path
from collections import defaultdict
from utils import select_or_latest, collect_image_files, load_label_txt

PARENT_FOLDER = Path(__file__).resolve().parent # Folder containing this script

if __name__ == "__main__":
    # Select result_det folder or automatically use the latest one
    det_dir = select_or_latest(base_dir=PARENT_FOLDER, title="Select result_det folder (Cancel to use latest)")

    # Read config file
    cfg_path = det_dir / "docs_run" / "model_cfg.yaml"
    cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    images_folder = cfg_data["IMAGES_FOLDER"]

    # All images in IMAGES_FOLDER (recursive)
    image_files, _ = collect_image_files(images_folder, stage="checking no detection images")

    # Use dictionaries to count no detection images, totals, and predicted boxes per subfolder
    subfolder_nodet_counts = defaultdict(int)
    subfolder_total_counts = defaultdict(int)
    subfolder_box_counts = defaultdict(int)

    # For each image
    for img_path in image_files:
        # Get relative directory and subfolder name
        rel_dir = img_path.parent.relative_to(Path(images_folder))   # Image folder name
        subfolder_name = str(rel_dir)

        # Count total images for subfolder
        subfolder_total_counts[subfolder_name] += 1

        # In detection.py, labels are saved in: run_dir (here det_dir) / "detection_result" / rel_dir / "labels"
        label_dir = det_dir / "detection_result" / rel_dir / "labels"
        label_path = label_dir / f"{img_path.stem}.txt"

        # Skip if label file does not exist
        if not label_path.exists():
            print(f"Warning: Label file not found for image {img_path}")
            continue

        # Load labels and count the number of boxes (lines) in the label file
        labels = load_label_txt(label_path, with_conf=False)
        subfolder_box_counts[subfolder_name] += labels.shape[0]

        # File is empty or contains only whitespace = no detections
        if labels.shape[0] == 0:
            subfolder_nodet_counts[subfolder_name] += 1

    # Write the results to a CSV file
    output_dir = det_dir / "docs_run" / "results_statistics_detection.csv"
    with output_dir.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subfolder", "No detection", "Total", "Crop boxes"])

        total_all_nodets = 0
        total_all_images = 0
        total_all_boxes = 0

        # Sort subfolders by name
        for subfolder in sorted(subfolder_total_counts.keys()):
            total = subfolder_total_counts[subfolder]
            nodet = subfolder_nodet_counts[subfolder]
            box_count = subfolder_box_counts[subfolder]
            writer.writerow([subfolder, nodet, total, box_count])
            total_all_images += total
            total_all_nodets += nodet
            total_all_boxes += box_count

        writer.writerow(["TOTAL", total_all_nodets, total_all_images, total_all_boxes])

    print(f"\nNo detection images: {total_all_nodets}, Total images: {total_all_images}, Total crop boxes: {total_all_boxes}")
    print(f"\nDone! Saved summary to: {output_dir}")