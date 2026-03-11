import yaml
from pathlib import Path
from collections import defaultdict
from utils import select_or_latest, collect_image_files, load_label_txt

PARENT_FOLDER = Path(__file__).resolve().parent # Folder containing this script
CFG_PATH = PARENT_FOLDER / "model_cfg.yaml"
CFG_DATA = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
IMAGES_FOLDER = CFG_DATA["IMAGES_FOLDER"]

if __name__ == "__main__":
    # Select result_det folder or automatically use the latest one
    det_dir = select_or_latest(base_dir=PARENT_FOLDER, title="Select result_det folder (Cancel to use latest)")

    # All images in IMAGES_FOLDER (recursive)
    image_files, _ = collect_image_files(IMAGES_FOLDER, stage="checking empty detection images")

    # Use dictionaries to count empties and totals per subfolder
    subfolder_empty_counts = defaultdict(int)
    subfolder_total_counts = defaultdict(int)

    # For each image
    for img_path in image_files:
        # Get relative directory and subfolder name
        rel_dir = img_path.parent.relative_to(Path(IMAGES_FOLDER))   # Image folder name
        subfolder_name = str(rel_dir)

        # Count total images for subfolder
        subfolder_total_counts[subfolder_name] += 1

        # In detection_entry.py, labels are saved in: run_dir (here det_dir) / rel_dir / "labels"
        label_dir = det_dir / rel_dir / "labels"
        label_path = label_dir / f"{img_path.stem}.txt"

        # Skip if label file does not exist
        if not label_path.exists():
            print(f"Warning: Label file not found for image {img_path}")
            continue

        # File is empty or contains only whitespace
        labels = load_label_txt(label_path, with_conf=False)
        if labels.shape[0] == 0:
            subfolder_empty_counts[subfolder_name] += 1

    # Write the results to a file
    output_dir = det_dir / "empty_detection.txt"
    with output_dir.open("w", encoding="utf-8") as f:
        f.write("Empty detections:"
        "\n(Subfolder, Empty, Total)")

        total_all_empties = 0
        total_all_images = 0

        # Sort subfolders by name
        for subfolder in sorted(subfolder_total_counts.keys()):
            total = subfolder_total_counts[subfolder]
            empties = subfolder_empty_counts[subfolder]
            line = f"{subfolder}, {empties}, {total}\n"
            f.write(line)
            total_all_images += total
            total_all_empties += empties

        f.write("-" * 30)
        f.write(f"\nTOTAL, {total_all_empties}, {total_all_images}\n")

    print(f"\nEmpty images: {total_all_empties}, Total images: {total_all_images}")
    print(f"\nDone! Saved summary to: {output_dir}")