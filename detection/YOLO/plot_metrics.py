"""
Custom plotting for AquaIA YOLO runs.

Usage (after a run exists):
    python plot_custom_metrics.py --run-dir models/yolo11n_coco128_custom_e800_bs4_img640_pretrained
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_context(run_dir: Path) -> dict:
    """Load resolved_config.after.yaml to get model/data/hyperparameter context if available."""
    cfg_after = run_dir / "resolved_config.after.yaml"
    if not cfg_after.is_file():
        return {}

    data = yaml.safe_load(cfg_after.read_text(encoding="utf-8")) or {}
    return data


def build_title_prefix(ctx: dict) -> str:
    """Build a human-readable prefix with model, dataset, and key hyperparameters."""
    model_cfg = ctx.get("model", {})
    data_cfg = ctx.get("data", {})
    train = ctx.get("training", {})

    family = model_cfg.get("family", "yolo")
    size = model_cfg.get("size", "")
    init = model_cfg.get("init", "")
    dataset_yaml = data_cfg.get("dataset_yaml", "unknown_dataset.yaml")
    dataset_name = Path(dataset_yaml).stem

    epochs = train.get("epochs", "?")
    batch = train.get("batch", "?")
    imgsz = train.get("imgsz", "?")

    return (
        f"AquaIA – {family}{size} ({init}) – {dataset_name} "
        f"[epochs={epochs}, batch={batch}, imgsz={imgsz}]"
    )


def plot_loss_curves(df: pd.DataFrame, run_dir: Path, title_prefix: str) -> None:
    """Plot train/val loss curves with clear titles, axis labels and legend."""
    plt.figure(figsize=(8, 5))

    has_any = False
    if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
        plt.plot(df["epoch"], df["train/box_loss"], label="train box loss")
        plt.plot(df["epoch"], df["val/box_loss"], label="val box loss")
        has_any = True
    if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
        plt.plot(df["epoch"], df["train/cls_loss"], label="train cls loss", linestyle="--", alpha=0.7)
        plt.plot(df["epoch"], df["val/cls_loss"], label="val cls loss", linestyle="--", alpha=0.7)
        has_any = True
    if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
        plt.plot(df["epoch"], df["train/dfl_loss"], label="train dfl loss", linestyle="-.")
        plt.plot(df["epoch"], df["val/dfl_loss"], label="val dfl loss", linestyle="-.")
        has_any = True

    if not has_any:
        plt.close()
        return

    # Axes labels and title
    plt.xlabel("Epoch")  # x-axis
    plt.ylabel("Loss")   # y-axis
    plt.title(f"{title_prefix}\nTraining and validation losses")  # main title
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    out = run_dir / "custom_loss_curves.png"
    plt.savefig(out, dpi=200)
    plt.close()


def plot_map_curves(df: pd.DataFrame, run_dir: Path, title_prefix: str) -> None:
    """Plot mAP metrics vs. epochs with clear titles, axis labels and legend."""
    plt.figure(figsize=(8, 5))

    has_any = False
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95(B)")
        has_any = True
    if "metrics/mAP50(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50(B)")
        has_any = True

    if not has_any:
        plt.close()
        return

    # Axes labels and title
    plt.xlabel("Epoch")  # x-axis
    plt.ylabel("mAP")    # y-axis
    plt.title(f"{title_prefix}\nValidation mAP vs. epochs")  # main title
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    out = run_dir / "custom_map_curves.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot custom YOLO training metrics for AquaIA.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a YOLO run directory (contains results.csv and resolved_config.after.yaml).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    results_csv = run_dir / "results.csv"

    if not results_csv.is_file():
        raise FileNotFoundError(f"results.csv not found in {run_dir}")

    df = pd.read_csv(results_csv)

    # Build a descriptive title prefix from the resolved config if available
    ctx = load_context(run_dir)
    if ctx:
        title_prefix = build_title_prefix(ctx)
    else:
        # Fallback if no resolved config is found
        title_prefix = f"AquaIA – {run_dir.name}"

    plot_loss_curves(df, run_dir, title_prefix=title_prefix)
    plot_map_curves(df, run_dir, title_prefix=title_prefix)

    print(f"Custom plots saved in {run_dir}")


if __name__ == "__main__":
    main()
