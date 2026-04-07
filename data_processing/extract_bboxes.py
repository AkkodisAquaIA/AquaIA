from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for bbox extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ROIs from YOLO bounding box labels and save them by class."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing YOLO label files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where extracted ROIs will be saved.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for image sampling.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Minimum ROI width and height in pixels.",
    )
    return parser.parse_args()


def list_image_files(images_dir: Path) -> list[Path]:
    """Return supported image files sorted by name."""
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def yolo_bbox_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Convert a normalized YOLO bbox to a clipped xyxy bounding box."""
    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    width_px = width * image_width
    height_px = height * image_height

    x_min = max(0, int(x_center_px - width_px / 2))
    y_min = max(0, int(y_center_px - height_px / 2))
    x_max = min(image_width, int(x_center_px + width_px / 2))
    y_max = min(image_height, int(y_center_px + height_px / 2))

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def extract_bboxes_from_image(
    image_path: Path,
    labels_dir: Path,
    output_dir: Path,
    min_size: int,
) -> int:
    """Extract and save all valid bbox crops from one image."""
    label_path = labels_dir / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return 0

    if not label_path.is_file():
        print(f"[WARNING] Missing label file: {label_path}")
        return 0

    image_height, image_width = image.shape[:2]
    saved_count = 0

    # Read all bbox annotations associated with this image.
    with label_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    # Extract one ROI per annotation line.
    for obj_idx, line in enumerate(lines, start=1):
        values = line.strip().split()
        if len(values) != 5:
            continue

        try:
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError:
            continue

        bbox = yolo_bbox_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        roi = image[y_min:y_max, x_min:x_max]

        # Skip empty or very small crops.
        if roi.size == 0 or roi.shape[0] < min_size or roi.shape[1] < min_size:
            continue

        # Save crops into one folder per class.
        class_dir = output_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        roi_name = f"{image_path.stem}_cls{class_id}_obj{obj_idx:03d}.jpg"
        roi_path = class_dir / roi_name

        success = cv2.imwrite(str(roi_path), roi)
        if success:
            saved_count += 1

    return saved_count


def main() -> None:
    """Run bbox extraction on a sampled subset of images."""
    args = parse_args()

    # Resolve and validate all input/output directories.
    images_dir = Path(args.images_dir).expanduser().resolve()
    labels_dir = Path(args.labels_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels directory not found: {labels_dir}")

    all_images = list_image_files(images_dir)
    print(f"[INFO] Total images available: {len(all_images)}")

    if not all_images:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    # Sample images deterministically for reproducible runs.
    rng = random.Random(args.seed)
    num_images = min(args.max_images, len(all_images))
    selected_images = rng.sample(all_images, num_images)
    print(f"[INFO] Selected images: {len(selected_images)}")

    total_saved = 0

    # Process each selected image independently.
    for image_path in selected_images:
        saved_count = extract_bboxes_from_image(
            image_path=image_path,
            labels_dir=labels_dir,
            output_dir=output_dir,
            min_size=args.min_size,
        )
        total_saved += saved_count

    print(f"[INFO] Bbox extraction complete. Saved ROIs: {total_saved}")


if __name__ == "__main__":
    main()