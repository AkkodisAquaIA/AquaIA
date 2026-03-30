from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for image resizing."""
    parser = argparse.ArgumentParser(
        description="Resize extracted ROI images and preserve the class folder structure."
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
        help="Target image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Target image height in pixels.",
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


def resize_image(
    image_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> bool:
    """Resize one image and save it to the output path."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return False

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), resized_image)

    if not success:
        print(f"[WARNING] Could not write image: {output_path}")
        return False

    return True


def main() -> None:
    """Resize all images while preserving the class folder structure."""
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

            success = resize_image(
                image_path=image_path,
                output_path=output_path,
                width=args.width,
                height=args.height,
            )
            if success:
                total_saved += 1

    print(f"[INFO] Resize complete. Processed: {total_processed}, Saved: {total_saved}")


if __name__ == "__main__":
    main()