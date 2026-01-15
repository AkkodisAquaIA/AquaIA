"""
Train a YOLO11n object detection model from scratch on the COCO128 subset.

This script:
- Loads a COCO128-style dataset config (train/val image paths and class names).
- Uses Ultralytics YOLO label convention automatically:
    images/<split>/...  ->  labels/<split>/... (same filename stem, .txt)
- Instantiates an untrained YOLO11n model from its YAML definition (random init, no pretrained weights).
- Trains the model from scratch with hyperparameters adapted to a Quadro RTX 5000 (16 GB VRAM class).
- Saves training artifacts (weights, plots, logs) in the `models/` directory.

Important:
- Training is supervised: it REQUIRES label files (.txt) in the expected labels/ folder.
- If labels are missing, training degenerates to "background-only" (no meaningful learning).
"""

from pathlib import Path

from ultralytics import YOLO
import torch
import yaml


def assert_labels_exist(dataset_yaml: str) -> None:
    p = Path(dataset_yaml).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"[DATA ERROR] Dataset YAML not found: {p}")

    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = Path(d.get("path") or p.parent).expanduser().resolve()

    def check(split: str) -> None:
        rel = d.get(split)
        if not rel:
            return
        img = (root / rel).resolve()
        if not img.exists():
            raise FileNotFoundError(f"[DATA ERROR] {split} images dir not found: {img}")

        parts = list(img.parts)
        if "images" not in parts:
            raise ValueError(f"[DATA ERROR] Cannot infer labels (no 'images' in path): {img}")
        parts[parts.index("images")] = "labels"
        lab = Path(*parts)

        if not lab.exists() or not any(lab.glob("*.txt")):
            raise FileNotFoundError(f"[DATA ERROR] Missing/empty labels dir for {split}: {lab}")

    check("train")
    check("val")


def main() -> None:
    """Main training entry point."""
    # Path to the dataset configuration (COCO128-style YAML with train/val paths and class names).
    # Inside Docker, the project root is /app, so the dataset is mounted at /app/datasets/...
    DATASET_YAML = "/app/datasets/coco128/coco128_custom.yaml"

    # ---------- DATA SANITY CHECK ----------
    # Make sure labels exist where Ultralytics expects them:
    # images/<split> -> labels/<split>
    assert_labels_exist(DATASET_YAML)

    # ---------- MODEL ----------
    # Create an untrained YOLO11n model from its architecture definition.
    # Using the `.yaml` file means the model starts with random weights (no pretraining).
    model = YOLO("yolo11n.yaml")

    # Select computation device: first CUDA GPU if available, otherwise CPU.
    device = 0 if torch.cuda.is_available() else "cpu"

    # Launch training with hyperparameters tuned for:
    # - A very small dataset (COCO128).
    # - A Quadro RTX 5000-class GPU (16 GB VRAM).
    results = model.train(
        # ---------- DATA & INPUT SIZE ----------
        data=DATASET_YAML,     # Dataset YAML file (train/val paths, number of classes, class names).
        imgsz=1024,            # RTX5000 often holds 1024. If OOM -> 960 then 832.
        batch=24,              # Aggressive start at 1024. If OOM -> 16. If margin -> 32.
        rect=False,            # Off: better augmentation diversity for from-scratch (often).

        # ---------- TRAINING SCHEDULE ----------
        epochs=800,            # From scratch can need more epochs; small dataset -> early stop helps.
        patience=80,          # Stop if validation metrics do not improve for 100 epochs.

        # ---------- OPTIMIZER & LEARNING RATE ----------
        optimizer="AdamW",     # More stable than SGD for from-scratch on small datasets.
        lr0=0.0020,            # Initial learning rate (AdamW). If unstable -> 0.0015.
        lrf=0.05,              # Final LR as a fraction of lr0 (cosine schedule).
        cos_lr=True,           # Enable cosine LR schedule.
        momentum=0.9,          # Not critical for AdamW, but acceptable.
        weight_decay=0.01,    # Regularization (helpful for small datasets).

        warmup_epochs=8,       # Longer warmup for large imgsz + bigger batches.
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ---------- REGULARIZATION ----------
        label_smoothing=0.03,  # Reduce overfitting and stabilize classification.

        # ---------- DATA AUGMENTATION ----------
        augment=True,          # Enable built-in data augmentation pipeline.
        hsv_h=0.015,           # Maximum change in hue for HSV color augmentation.
        hsv_s=0.7,             # Maximum change in saturation.
        hsv_v=0.4,             # Maximum change in value/brightness.
        degrees=10.0,          # A bit more geometry helps from scratch.
        translate=0.10,        # Maximum translation as a fraction of image size.
        scale=0.5,            # More aggressive scaling (small dataset).
        shear=2.0,             # Shear angle in degrees.
        perspective=0.0005,    # Perspective distortion factor.
        flipud=0.0,            # Probability to flip images vertically.
        fliplr=0.5,            # Probability to flip images horizontally.
        mosaic=0.8,            # Strong mosaic early: big diversity boost for COCO128.
        mixup=0.05,            # Light-to-moderate mixup.
        close_mosaic=40,       # Disable mosaic for the last epochs to refine.

        # ---------- PERFORMANCE ----------
        amp=True,              # Mixed precision: faster / lower VRAM.
        cache=False,           # Docker: safe default. Set True only if you are sure about RAM/disk.
        workers=4,             # 4 vCPU: 2-4 typically optimal.

        # ---------- OUTPUT / LOGGING ----------
        project="models",      # Root folder for training runs (inside the container).
        name="yolo11n_coco128_scratch_RTX5000_max_2",  # Name of this specific run.
        exist_ok=True,         # Allow reusing the same run name (do not raise an error).
        device=device,         # Device used for training ("cpu" or GPU index).
        save=True,             # Save final weights and checkpoints.
        save_period=25,        # Save intermediate weights every N epochs.
        plots=True,            # Save training plots (loss curves, metrics).
    )

    # `results` contains training metrics and paths to saved artifacts.
    return results


if __name__ == "__main__":
    # Inform the user when no GPU is detected, since training on CPU will be very slow.
    if not torch.cuda.is_available():
        print("Warning: no GPU detected. Training will run on CPU and be very slow.")
    main()