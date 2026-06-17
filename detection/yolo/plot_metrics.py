#!/usr/bin/env python3
"""
AquaIA – YOLO Training Metrics Plotter.

Two modes:
  - Single run : unified 2x2 panel (losses / mAP / precision+recall / summary).
  - Multi-run  : overlay all runs on shared axes for direct comparison.

Usage – single run:
    python plot_metrics.py --run-dir yolo_models/20260318-...

Usage – multi-run comparison:
    python plot_metrics.py --runs-dir yolo_models --output-dir runs/plots

Usage – filtered multi-run:
    python plot_metrics.py --runs-dir yolo_models --run-filter pretrained
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_FILE = "resolved_config.after.yaml"
_RESULTS_CSV = "results.csv"

# Color cycle – enough for up to 8 runs side by side.
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load results.csv from a run directory. Returns None if not found."""
    csv_path = run_dir / _RESULTS_CSV
    if not csv_path.is_file():
        print(f"[WARNING] results.csv not found in {run_dir}")
        return None
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (Ultralytics sometimes adds leading spaces).
    df.columns = [c.strip() for c in df.columns]
    return df


def load_run_context(run_dir: Path) -> dict:
    """Load resolved_config.after.yaml to get model/training context."""
    cfg_path = run_dir / _CONFIG_FILE
    if not cfg_path.is_file():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def build_run_label(run_dir: Path, ctx: dict) -> str:
    """
    Build a short human-readable label for a run, used in legends.
    Falls back to the run folder name if no config is found.
    """
    model = ctx.get("model", {})
    training = ctx.get("training", {})
    if model:
        family = model.get("family", "yolo")
        size = model.get("size", "")
        init = model.get("init", "")
        epochs = training.get("epochs", "?")
        return f"{family}{size} e{epochs} ({init})"
    return run_dir.name


def build_title_prefix(run_dir: Path, ctx: dict) -> str:
    """Build a descriptive title prefix from the resolved config."""
    model = ctx.get("model", {})
    data = ctx.get("data", {})
    training = ctx.get("training", {})
    if not model:
        return f"AquaIA – {run_dir.name}"
    family = model.get("family", "yolo")
    size = model.get("size", "")
    init = model.get("init", "")
    dataset = Path(data.get("dataset_yaml", "unknown.yaml")).stem
    epochs = training.get("epochs", "?")
    batch = training.get("batch", "?")
    imgsz = training.get("imgsz", "?")
    return f"AquaIA – {family}{size} ({init}) – {dataset} [epochs={epochs}, batch={batch}, imgsz={imgsz}]"


# ---------------------------------------------------------------------------
# Best-epoch detection
# ---------------------------------------------------------------------------


def find_best_epoch(df: pd.DataFrame) -> Optional[Tuple[int, float]]:
    """
    Return (epoch, mAP50_value) for the epoch with the highest mAP50.
    Returns None if the required column is missing.
    """
    col = "metrics/mAP50(B)"
    if col not in df.columns or "epoch" not in df.columns:
        return None
    idx = df[col].idxmax()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, col])


# ---------------------------------------------------------------------------
# Single-run: unified 2x2 panel
# ---------------------------------------------------------------------------


def plot_single_run(run_dir: Path, output_dir: Optional[Path] = None) -> None:
    """
    Plot a unified panel for one training run:
        [0,0] Train losses      |  [0,1] Val losses
        [1,0] Validation mAP    |  [1,1] Precision & Recall
        [2,  ] Final metrics summary (full width)
    """
    df = load_results_csv(run_dir)
    if df is None:
        return

    ctx = load_run_context(run_dir)
    title_prefix = build_title_prefix(run_dir, ctx)
    best = find_best_epoch(df)

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.2], hspace=0.45, wspace=0.3)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_val = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_pr = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])  # full width

    fig.suptitle(title_prefix, fontsize=9, y=1.01)

    loss_defs = [
        ("train/box_loss", "val/box_loss", "box loss", "#1f77b4"),
        ("train/cls_loss", "val/cls_loss", "cls loss", "#ff7f0e"),
        ("train/dfl_loss", "val/dfl_loss", "dfl loss", "#2ca02c"),
    ]

    # ---- [0,0] Train losses ----
    for train_col, _, label, color in loss_defs:
        if train_col in df.columns:
            ax_train.plot(df["epoch"], df[train_col], color=color, linewidth=1.2, label=label)
    if best:
        ax_train.axvline(best[0], color="gray", linestyle=":", linewidth=1, label=f"best ({best[0]})")
    ax_train.set_title("Train Losses")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_train.legend(fontsize=7)
    ax_train.grid(True, linestyle=":")

    # ---- [0,1] Val losses ----
    for _, val_col, label, color in loss_defs:
        if val_col in df.columns:
            ax_val.plot(df["epoch"], df[val_col], color=color, linewidth=1.2, label=label)
    if best:
        ax_val.axvline(best[0], color="gray", linestyle=":", linewidth=1, label=f"best ({best[0]})")
    ax_val.set_title("Val Losses")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Loss")
    ax_val.legend(fontsize=7)
    ax_val.grid(True, linestyle=":")

    # ---- [1,0] Validation mAP ----
    map_cols = [
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    has_map = False
    for col, label in map_cols:
        if col in df.columns:
            ax_map.plot(df["epoch"], df[col], label=label)
            has_map = True
    if best and has_map:
        ax_map.axvline(best[0], color="gray", linestyle=":", linewidth=1)
        ax_map.annotate(
            f"best epoch {best[0]}\nmAP50={best[1]:.3f}",
            xy=(best[0], best[1]),
            xytext=(best[0] + max(1, len(df) * 0.06), best[1] - 0.06),
            fontsize=7,
            color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
    ax_map.set_title("Validation mAP")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP")
    ax_map.legend(fontsize=8)
    ax_map.grid(True, linestyle=":")

    # ---- [1,1] Precision & Recall ----
    pr_cols = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ]
    has_pr = False
    for col, label in pr_cols:
        if col in df.columns:
            ax_pr.plot(df["epoch"], df[col], label=label)
            has_pr = True
    if not has_pr:
        ax_pr.text(
            0.5,
            0.5,
            "No precision/recall data",
            ha="center",
            va="center",
            transform=ax_pr.transAxes,
            color="gray",
        )
    if best and has_pr:
        ax_pr.axvline(best[0], color="gray", linestyle=":", linewidth=1)
    ax_pr.set_title("Precision & Recall")
    ax_pr.set_xlabel("Epoch")
    ax_pr.set_ylabel("Score")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(True, linestyle=":")

    # ---- [2, :] Final metrics summary table (full width) ----
    ax_table.axis("off")
    resolved = ctx.get("resolved", {})
    metrics = resolved.get("metrics", {}) or {}
    training = ctx.get("training", {})

    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    summary_rows = [
        [
            _fmt(metrics.get("map50")),
            _fmt(metrics.get("map50_95")),
            _fmt(metrics.get("precision")),
            _fmt(metrics.get("recall")),
            str(best[0]) if best else "N/A",
            str(training.get("epochs", "N/A")),
            str(training.get("batch", "N/A")),
            str(training.get("imgsz", "N/A")),
        ]
    ]
    col_labels = [
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "Best epoch",
        "Epochs",
        "Batch",
        "Imgsz",
    ]
    table = ax_table.table(
        cellText=summary_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    ax_table.set_title("Final Metrics", pad=8)

    save_dir = output_dir if output_dir is not None else run_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"panel_{run_dir.name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Panel saved → {out_path}")


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------


def plot_multi_run(run_dirs: List[Path], output_dir: Path) -> None:
    """
    Overlay multiple runs on shared axes for direct visual comparison.

    Layout (2x2):
        [0,0] mAP50      |  [0,1] mAP50-95
        [1,0] Precision   |  [1,1] Recall
    """
    if not run_dirs:
        print("[WARNING] No runs to compare.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("AquaIA – Multi-Run Comparison", fontsize=11)

    plot_specs = [
        (axes[0, 0], "metrics/mAP50(B)", "mAP50"),
        (axes[0, 1], "metrics/mAP50-95(B)", "mAP50-95"),
        (axes[1, 0], "metrics/precision(B)", "Precision"),
        (axes[1, 1], "metrics/recall(B)", "Recall"),
    ]
    for ax, _, title in plot_specs:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":")

    for i, run_dir in enumerate(run_dirs):
        df = load_results_csv(run_dir)
        if df is None:
            continue
        ctx = load_run_context(run_dir)
        label = build_run_label(run_dir, ctx)
        color = _COLORS[i % len(_COLORS)]
        best = find_best_epoch(df)

        for ax, col, _ in plot_specs:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=label, color=color)
                if best:
                    ax.axvline(best[0], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    # Single shared legend at the bottom of the figure.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(run_dirs), 4),
            fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout()
    out_path = output_dir / "comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Comparison plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def scan_valid_runs(runs_dir: Path, run_filter: Optional[str] = None) -> List[Path]:
    """Return sorted run directories that contain results.csv."""
    candidates = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        if run_filter and run_filter not in p.name:
            continue
        if (p / _RESULTS_CSV).is_file():
            candidates.append(p)
    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AquaIA – YOLO Training Metrics Plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Single run directory (must contain results.csv). Produces a unified 2x2 panel.",
    )
    group.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directory of run folders. Produces individual panels + a combined comparison.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Where to save the PNG(s). Default: run folder (single mode) or runs/plots (multi mode)."),
    )
    parser.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="Substring filter on run folder names (multi mode only, e.g. 'pretrained').",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=8,
        help="Maximum number of runs included in the comparison (multi mode).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise NotADirectoryError(f"run-dir not found: {run_dir}")
        plot_single_run(run_dir, output_dir=output_dir)

    else:
        runs_dir = Path(args.runs_dir).expanduser().resolve()
        if not runs_dir.is_dir():
            raise NotADirectoryError(f"runs-dir not found: {runs_dir}")

        run_dirs = scan_valid_runs(runs_dir, run_filter=args.run_filter)
        run_dirs = run_dirs[: args.max_runs]

        if not run_dirs:
            print("No valid runs found.")
            return

        print(f"  {len(run_dirs)} run(s) found.")

        if output_dir is None:
            output_dir = runs_dir.parent / "runs" / "plots"

        for run_dir in run_dirs:
            plot_single_run(run_dir, output_dir=output_dir)

        plot_multi_run(run_dirs, output_dir=output_dir)


if __name__ == "__main__":
    main()
