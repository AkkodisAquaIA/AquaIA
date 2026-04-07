from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for padded image resizing."""
    parser = argparse.ArgumentParser(
        description="Resize images while preserving aspect ratio and adding padding."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing class subfolders with input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where resized images will be saved.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Target output width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Target output height in pixels.",
    )
    parser.add_argument(
        "--pad-value",
        type=int,
        default=0,
        help="Padding color value for all channels (0-255).",
    )
    return parser.parse_args()


def list_class_directories(input_dir: Path) -> list[Path]:
    """Return all class subdirectories sorted by name."""
    return sorted([path for path in input_dir.iterdir() if path.is_dir()])


def list_image_files(class_dir: Path) -> list[Path]:
    """Return supported image files sorted by name."""
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def resize_with_padding(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize an image while keeping aspect ratio, then pad to target size."""
    image_height, image_width = image.shape[:2]

    scale = min(target_width / image_width, target_height / image_height)
    new_width = max(1, int(round(image_width * scale)))
    new_height = max(1, int(round(image_height * scale)))

    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (target_height, target_width, 3),
        pad_value,
        dtype=np.uint8,
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas


def process_image(
    image_path: Path,
    output_path: Path,
    width: int,
    height: int,
    pad_value: int,
) -> bool:
    """Read, resize with padding, and save one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return False

    output_image = resize_with_padding(
        image=image,
        target_width=width,
        target_height=height,
        pad_value=pad_value,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), output_image)

    if not success:
        print(f"[WARNING] Could not write image: {output_path}")
        return False

    return True


def main() -> None:
    """Resize all images with aspect-ratio preservation and padding."""
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    class_dirs = list_class_directories(input_dir)
    print(f"[INFO] Class folders found: {len(class_dirs)}")

    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in: {input_dir}")

    total_processed = 0
    total_saved = 0

    for class_dir in class_dirs:
        output_class_dir = output_dir / class_dir.name
        image_files = list_image_files(class_dir)

        print(f"[INFO] Processing class '{class_dir.name}' with {len(image_files)} images")

        for image_path in image_files:
            output_path = output_class_dir / image_path.name
            total_processed += 1

            success = process_image(
                image_path=image_path,
                output_path=output_path,
                width=args.width,
                height=args.height,
                pad_value=args.pad_value,
            )
            if success:
                total_saved += 1

    print(f"[INFO] Resize with padding complete. Processed: {total_processed}, Saved: {total_saved}")


if __name__ == "__main__":
    main()