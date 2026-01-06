"""
Train a YOLO11n object detection model from scratch on the COCO128 subset.

This script:
- Loads a custom COCO128-style dataset.
- Instantiates an untrained YOLO11n model from its YAML definition.
- Trains the model from scratch with hyperparameters adapted to a small GPU (4 GB VRAM).
- Saves training artifacts (weights, plots, logs) in the `models/` directory.
"""

from ultralytics import YOLO
import torch


def main() -> None:
    """Main training entry point."""
    # Path to the dataset configuration (COCO128-style YAML with train/val paths and class names).
    # Inside Docker, the project root is /app, so the dataset is mounted at /app/datasets/...
    DATASET_YAML = "../datasets/coco128/coco128_custom.yaml"

    # Create an untrained YOLO11n model from its architecture definition.
    # Using the `.yaml` file means the model starts with random weights (no pretraining).
    model = YOLO("yolo11n.yaml")

    # Select computation device: first CUDA GPU if available, otherwise CPU.
    device = 0 if torch.cuda.is_available() else "cpu"

    # Launch training with hyperparameters tuned for:
    # - A very small dataset (COCO128).
    # - A small GPU (NVIDIA Quadro M3000M, 4 GB VRAM).
    results = model.train(
        # ---------- DATA & INPUT SIZE ----------
        data=DATASET_YAML,   # Dataset YAML file (train/val paths, number of classes, class names).
        imgsz=640,           # Input image size (square). Larger = more detail but more memory.
        batch=8,             # Number of images per batch. Limited by GPU memory.

        # ---------- TRAINING SCHEDULE ----------
        epochs=800,          # Maximum number of training epochs.
        patience=50,         # Early stopping: stop if validation metrics do not improve for 50 epochs.

        # ---------- OPTIMIZER & LEARNING RATE ----------
        optimizer="SGD",     # Optimizer type: "SGD" or "AdamW" (SGD is standard for YOLO).
        lr0=0.005,           # Initial learning rate at the start of training.
        lrf=0.01,            # Final learning rate as a fraction of lr0 (with cosine schedule).
        momentum=0.937,      # Momentum for SGD; helps smooth updates.
        weight_decay=0.0005, # L2 regularization strength; combats overfitting.
        warmup_epochs=3,     # Number of warmup epochs with gradually increasing LR.

        # ---------- DATA AUGMENTATION ----------
        augment=True,        # Enable built-in data augmentation pipeline.
        hsv_h=0.015,         # Maximum change in hue for HSV color augmentation.
        hsv_s=0.7,           # Maximum change in saturation.
        hsv_v=0.4,           # Maximum change in value/brightness.
        degrees=5.0,         # Maximum rotation in degrees.
        translate=0.1,       # Maximum translation as a fraction of image size.
        scale=0.2,           # Random scaling factor range.
        shear=2.0,           # Shear angle in degrees.
        perspective=0.0005,  # Perspective distortion factor.
        flipud=0.0,          # Probability to flip images vertically.
        fliplr=0.5,          # Probability to flip images horizontally.
        mosaic=0.1,          # Probability to use mosaic augmentation (4 images combined).
        mixup=0.05,          # Probability to use mixup (linear combination of 2 images).

        # ---------- OUTPUT / LOGGING ----------
        # In Docker, /app is the working directory. We mount the local "models" folder to /app/models.
        project="models",                # Root folder for training runs (inside the container).
        name="yolo11n_coco128_scratch_M3000M",  # Name of this specific run.
        exist_ok=True,        # Allow reusing the same run name (do not raise an error).
        device=device,        # Device used for training ("cpu" or GPU index).
        workers=2,            # Number of DataLoader worker processes.
        save=True,            # Save final weights and checkpoints.
        save_period=50,       # Save intermediate weights every N epochs.
        plots=True,           # Save training plots (loss curves, metrics).
    )

    # `results` contains training metrics and paths to saved artifacts.
    return results


if __name__ == "__main__":
    # Inform the user when no GPU is detected, since training on CPU will be very slow.
    if not torch.cuda.is_available():
        print("Warning: no GPU detected. Training will run on CPU and be very slow.")

    main()
