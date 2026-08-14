# Computes mean and std matching the original DALI pipeline.
# Streamlines 1 image at a time to prevent OOM (no batch_size).
# Aggregates 'train', 'val', and 'test' under 'images'.
# Matches DALI RGB, Bilinear resize, (C, H, W), and 1/255.0 scaling.
# Saves 'stats.npy' in the <dataset_name> folder.

import os
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_sorted_jpg_files(dataset_dir):
    """Get all jpg files in dataset_dir, sort them by numeric order if possible, otherwise by name."""

    def _numeric_sort_key(path):
        stem = Path(path).stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    jpg_files = []
    sub_dirs = ["train", "val", "test"]
    for sub_dir in sub_dirs:
        img_dir = os.path.join(dataset_dir, "images", sub_dir)
        if os.path.exists(img_dir):
            # Only jpg in img_dir not deeper
            files = [str(p) for p in Path(img_dir).glob("*.jpg")]
            jpg_files.extend(files)
    if not jpg_files:
        images_dir = Path(dataset_dir) / "images"
        for img_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
            files = [str(p) for p in img_dir.glob("*.jpg")]
            jpg_files.extend(files)
    return sorted(jpg_files, key=_numeric_sort_key)


def compute_and_save_stats(dataset_name="coco_custom_match", image_size=640):
    dataset_dir = os.path.join(BASE_DIR, "datasets", dataset_name)
    sorted_files = _get_sorted_jpg_files(dataset_dir)
    n = len(sorted_files)
    if n == 0:
        raise FileNotFoundError(f"No jpg files found under {os.path.join(dataset_dir, 'images')}")
    print(f"Found {n} images")

    # To avoid memory overflow, compute mean and std incrementally
    pixel_count_per_img = image_size * image_size

    # Initialize accumulators for sum and sum of squares for each channel
    channels = 3
    sum_ = np.zeros(channels, dtype=np.float64)
    sum_sq_ = np.zeros(channels, dtype=np.float64)

    for idx, file_path in enumerate(sorted_files):
        try:
            # Read image, transform to RGB, resize and normalize
            with Image.open(file_path).convert("RGB") as img:
                # Image.BILINEAR similar to DALI's TRIANGULAR filter
                img_resized = img.resize((image_size, image_size), Image.BILINEAR)
                # Convert to numpy array (H, W, C) and normalize to [0, 1]
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                # Transform to (C, H, W)
                arr = np.transpose(arr, (2, 0, 1))
                # Accumulate the sum and sum of squares for each channel
                sum_ += np.sum(arr, axis=(1, 2))
                sum_sq_ += np.sum(arr**2, axis=(1, 2))

        except Exception as e:
            print(f"Failed to read image {file_path}: {e}")
            continue

        if (idx + 1) % 500 == 0 or (idx + 1) == n:
            print(f"Processed: {idx + 1}/{n}")

    # Calculate final channel wise mean and std
    total_pixels = n * pixel_count_per_img
    mean = sum_ / total_pixels
    # Variance = E[X^2] - (E[X])^2
    var = (sum_sq_ / total_pixels) - (mean**2)
    # Ensure no small negative values due to floating point errors
    var = np.clip(var, a_min=0, a_max=None)
    std = np.sqrt(var)
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    # Save stats.npy
    output_path = os.path.join(dataset_dir, "stats.npy")
    np.save(output_path, stats)
    print(f"Statistics computation completed! Saved to: {output_path}")
    print(f"Mean: {stats['mean']}, Std: {stats['std']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute dataset stats.npy for mean and std")
    parser.add_argument("--dataset", type=str, default="coco_custom_match", help="Name of the dataset")
    parser.add_argument("--image-size", type=int, default=640, help="Output image size")
    args = parser.parse_args()
    compute_and_save_stats(dataset_name=args.dataset, image_size=args.image_size)
