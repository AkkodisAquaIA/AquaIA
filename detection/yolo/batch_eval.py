#!/usr/bin/env python3
"""
AquaIA – Batch YOLO Run Evaluator.

Scans a directory of YOLO training runs and evaluates each one by calling
yolo_run_diagnostics.py as a subprocess. Results are aggregated into a
timestamped CSV and a Markdown report.

Usage:
    python batch_eval.py \
        --runs-dir yolo_models \
        --dataset-yaml-override app/datasets/coco2017/coco2017.yaml \
        --output-dir runs/batch_eval \
        --pred-conf 0.10 \
        --pred-iou 0.90 \
        --match-iou 0.5
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_FILE = "resolved_config.after.yaml"

_DISPLAY_COLS = [
    "run_name",
    "model_family",
    "model_size",
    "model_init",
    "epochs",
    "train_mAP50",
    "train_mAP50_95",
    "train_precision",
    "train_recall",
    "iou_mean",
    "iou_samples",
    "good",
    "missed",
    "fp_only",
    "low_iou_examples",
    "pred_conf",
    "pred_iou",
    "match_iou",
    "eval_time_s",
    "status",
]

_MD_COLS = [
    "run_name",
    "model_init",
    "epochs",
    "train_mAP50",
    "train_mAP50_95",
    "train_precision",
    "train_recall",
    "iou_mean",
    "good",
    "missed",
    "fp_only",
    "status",
]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def find_run_config(run_dir: Path) -> Optional[Path]:
    """Return the resolved config file path if it exists inside a run directory."""
    p = run_dir / _CONFIG_FILE
    return p if p.is_file() else None


def scan_valid_runs(runs_dir: Path) -> List[Path]:
    """Return sorted list of run directories that contain a resolved config."""
    return [p for p in sorted(runs_dir.iterdir()) if p.is_dir() and (p / _CONFIG_FILE).is_file()]


# ---------------------------------------------------------------------------
# Metadata extraction from resolved_config.after.yaml
# ---------------------------------------------------------------------------


def extract_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """
    Extract model info and training metrics from resolved_config.after.yaml.

    Returns a flat dict: run_name, model_family, model_size, model_init,
    epochs, batch, imgsz, train_mAP50, train_mAP50_95, train_precision,
    train_recall.
    """
    cfg_path = find_run_config(run_dir)
    if cfg_path is None:
        return {"run_name": run_dir.name}

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"run_name": run_dir.name}

    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    resolved = cfg.get("resolved", {}) if isinstance(cfg.get("resolved"), dict) else {}
    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    metrics = resolved.get("metrics", {}) or {}

    return {
        "run_name": run_dir.name,
        "model_family": model.get("family", "unknown"),
        "model_size": model.get("size", "unknown"),
        "model_init": model.get("init", "unknown"),
        "epochs": training.get("epochs"),
        "batch": training.get("batch"),
        "imgsz": training.get("imgsz"),
        # Training-time metrics stored by train_yolo.py after the run.
        "train_mAP50": metrics.get("map50", None),
        "train_mAP50_95": metrics.get("map50_95", None),
        "train_precision": metrics.get("precision", None),
        "train_recall": metrics.get("recall", None),
    }


# ---------------------------------------------------------------------------
# stdout parsing – aligned with yolo_run_diagnostics.py log format
# ---------------------------------------------------------------------------


def parse_diagnostics_stdout(stdout: str) -> Dict[str, Any]:
    """
    Extract diagnostic metrics from the stdout of yolo_run_diagnostics.py.

    Matched log patterns emitted by yolo_run_diagnostics.py:
        [INFO] Mean IoU: 0.612 over 192 matched predictions
        [INFO] Categorized images: good=45, missed=12, fp_only=3
        [INFO] Low-IoU examples logged: 8
        [INFO] TensorBoard export complete: iou_samples=192, ...
    """
    diag: Dict[str, Any] = {}

    # Mean IoU and matched sample count.
    m = re.search(
        r"\[INFO\]\s+Mean IoU:\s*([\d.]+)\s+over\s+(\d+)\s+matched",
        stdout,
        re.IGNORECASE,
    )
    if m:
        diag["iou_mean"] = float(m.group(1))
        diag["iou_samples"] = int(m.group(2))

    # Per-category image counts: good / missed / fp_only.
    m = re.search(
        r"Categorized images:\s*good=(\d+),\s*missed=(\d+),\s*fp_only=(\d+)",
        stdout,
        re.IGNORECASE,
    )
    if m:
        diag["good"] = int(m.group(1))
        diag["missed"] = int(m.group(2))
        diag["fp_only"] = int(m.group(3))

    # Number of low-IoU examples sent to TensorBoard.
    m = re.search(r"Low-IoU examples logged:\s*(\d+)", stdout, re.IGNORECASE)
    if m:
        diag["low_iou_examples"] = int(m.group(1))

    # Confirm the final TensorBoard export line is present.
    diag["tb_exported"] = bool(re.search(r"TensorBoard export complete", stdout, re.IGNORECASE))

    return diag


# ---------------------------------------------------------------------------
# Command builder for yolo_run_diagnostics.py
# ---------------------------------------------------------------------------


def build_eval_command(
    run_dir: Path,
    dataset_yaml: Path,
    tb_dir: Path,
    pred_conf: float,
    pred_iou: float,
    match_iou: float,
    conf_thres_iou: float,
    max_images: int,
    workers: int,
    subset_seed: int,
    subset_mode: str,
    fixed_compare_count: int,
    category_grid_count: int,
    log_images: bool,
    skip_val: bool,
    split: str,
) -> List[str]:
    """Build the subprocess command that calls yolo_run_diagnostics.py."""
    script_dir = Path(__file__).parent.resolve()
    diag_script = script_dir / "yolo_run_diagnostics.py"

    if not diag_script.is_file():
        raise FileNotFoundError(f"yolo_run_diagnostics.py not found next to batch_eval.py: {diag_script}")

    cmd = [
        sys.executable,
        str(diag_script),
        f"--train-run-dir={run_dir}",
        f"--dataset-yaml-override={dataset_yaml}",
        f"--log-dir={tb_dir}",
        f"--pred-conf={pred_conf}",
        f"--pred-iou={pred_iou}",
        f"--match-iou={match_iou}",
        f"--conf-thres-iou={conf_thres_iou}",
        f"--max-images={max_images}",
        f"--workers={workers}",
        f"--subset-seed={subset_seed}",
        f"--subset-mode={subset_mode}",
        f"--fixed-compare-count={fixed_compare_count}",
        f"--category-grid-count={category_grid_count}",
        f"--split={split}",
    ]

    cmd.append("--log-images" if log_images else "--no-log-images")
    if skip_val:
        cmd.append("--skip-val")

    return cmd


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------


def eval_single_run(
    run_dir: Path,
    dataset_yaml: Path,
    tb_dir: Path,
    pred_conf: float,
    pred_iou: float,
    match_iou: float,
    conf_thres_iou: float,
    max_images: int,
    workers: int,
    subset_seed: int,
    subset_mode: str,
    fixed_compare_count: int,
    category_grid_count: int,
    log_images: bool,
    skip_val: bool,
    split: str,
    timeout: int = 900,
) -> Dict[str, Any]:
    """
    Call yolo_run_diagnostics.py on one run as a subprocess.

    Returns a dict containing diagnostic metrics (parsed from stdout),
    status, optional error message, and wall-clock eval_time_s.
    """
    base: Dict[str, Any] = {
        "run_name": run_dir.name,
        "pred_conf": pred_conf,
        "pred_iou": pred_iou,
        "match_iou": match_iou,
    }

    try:
        cmd = build_eval_command(
            run_dir=run_dir,
            dataset_yaml=dataset_yaml,
            tb_dir=tb_dir,
            pred_conf=pred_conf,
            pred_iou=pred_iou,
            match_iou=match_iou,
            conf_thres_iou=conf_thres_iou,
            max_images=max_images,
            workers=workers,
            subset_seed=subset_seed,
            subset_mode=subset_mode,
            fixed_compare_count=fixed_compare_count,
            category_grid_count=category_grid_count,
            log_images=log_images,
            skip_val=skip_val,
            split=split,
        )
    except FileNotFoundError as exc:
        return {**base, "status": "FAILED", "error": str(exc), "eval_time_s": 0.0}

    print(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            **base,
            "status": "TIMEOUT",
            "error": f"Exceeded {timeout}s",
            "eval_time_s": round(time.time() - t0, 2),
        }

    elapsed = round(time.time() - t0, 2)

    if proc.returncode != 0:
        print(f"  [ERROR] returncode={proc.returncode}")
        print(proc.stderr[-2000:])
        return {
            **base,
            "status": "FAILED",
            "error": proc.stderr.strip()[-500:],
            "eval_time_s": elapsed,
        }

    diag = parse_diagnostics_stdout(proc.stdout)
    return {**base, "status": "OK", "eval_time_s": elapsed, **diag}


# ---------------------------------------------------------------------------
# DataFrame ranking
# ---------------------------------------------------------------------------


def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort runs with OK status first, then by:
        train_mAP50 descending, iou_mean descending,
        missed ascending, fp_only ascending.
    """
    df = df.copy()
    df["_ok"] = (df["status"] == "OK").astype(int)

    sort_cols = ["_ok"]
    ascending = [False]

    for col in ("train_mAP50", "iou_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            sort_cols.append(col)
            ascending.append(False)

    for col in ("missed", "fp_only"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            sort_cols.append(col)
            ascending.append(True)

    return df.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_ok"])


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(df: pd.DataFrame, md_path: Path, batch_id: str) -> None:
    """Write a Markdown comparison table and best-run summary to md_path."""
    available = [c for c in _MD_COLS if c in df.columns]

    lines = [
        "# AquaIA – YOLO Batch Evaluation",
        "",
        f"**Batch ID**: `{batch_id}`  ",
        f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Runs evaluated**: {len(df)}",
        "",
        "## Results (ranked by train_mAP50 ↓)",
        "",
        df[available].round(4).to_markdown(index=False),
        "",
    ]

    ok_df = df[df["status"] == "OK"]
    if not ok_df.empty:
        best = ok_df.iloc[0]
        lines += [
            "## Best run",
            "",
            f"**{best['run_name']}**",
            f"- `train_mAP50`    = {best.get('train_mAP50', 'N/A')}",
            f"- `train_mAP50_95` = {best.get('train_mAP50_95', 'N/A')}",
            f"- `iou_mean`       = {best.get('iou_mean', 'N/A')}",
            (f"- `good / missed / fp_only` = {best.get('good', '?')} / {best.get('missed', '?')} / {best.get('fp_only', '?')}"),
        ]

    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AquaIA Batch YOLO Run Evaluator → CSV + Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--runs-dir", required=True, help="Directory containing YOLO run folders.")
    parser.add_argument(
        "--dataset-yaml-override",
        required=True,
        help="Path to the dataset YAML used for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/batch_eval",
        help="Root output folder for CSV, Markdown and TensorBoard logs.",
    )
    parser.add_argument("--max-runs", type=int, default=20, help="Maximum number of runs to evaluate.")
    parser.add_argument(
        "--run-filter",
        default=None,
        help="Optional substring filter on run folder names (e.g. 'pretrained').",
    )

    parser.add_argument("--pred-conf", type=float, default=0.10)
    parser.add_argument("--pred-iou", type=float, default=0.90)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--conf-thres-iou", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--subset-mode", choices=["random", "head"], default="random")
    parser.add_argument("--fixed-compare-count", type=int, default=32)
    parser.add_argument("--category-grid-count", type=int, default=32)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Subprocess timeout in seconds per run.",
    )

    parser.add_argument(
        "--no-log-images",
        dest="log_images",
        action="store_false",
        help="Disable TensorBoard image grids (faster batch runs).",
    )
    parser.set_defaults(log_images=True)

    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip model.val() and run only custom per-image diagnostics.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    dataset_yaml = Path(args.dataset_yaml_override).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.is_dir():
        raise NotADirectoryError(f"runs-dir not found: {runs_dir}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    batch_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    tb_dir = output_dir / f"tb_{batch_id}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"results_{batch_id}.csv"
    md_path = output_dir / f"report_{batch_id}.md"

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AquaIA Batch Evaluator  –  {batch_id}")
    print(sep)
    print(f"  runs-dir   : {runs_dir}")
    print(f"  dataset    : {dataset_yaml}")
    print(f"  output-dir : {output_dir}")
    print(f"  tb-dir     : {tb_dir}")
    print(f"  pred_conf={args.pred_conf}  pred_iou={args.pred_iou}  match_iou={args.match_iou}")
    print(sep)

    valid_runs = scan_valid_runs(runs_dir)
    if args.run_filter:
        valid_runs = [r for r in valid_runs if args.run_filter in r.name]
    valid_runs = valid_runs[: args.max_runs]

    print(f"\n  {len(valid_runs)} valid run(s) found.\n")
    if not valid_runs:
        print("  Nothing to evaluate. Exiting.")
        return

    results: List[Dict[str, Any]] = []
    for i, run_dir in enumerate(valid_runs, 1):
        print(f"\n[{i}/{len(valid_runs)}] {run_dir.name}")
        metadata = extract_run_metadata(run_dir)
        eval_result = eval_single_run(
            run_dir=run_dir,
            dataset_yaml=dataset_yaml,
            tb_dir=tb_dir,
            pred_conf=args.pred_conf,
            pred_iou=args.pred_iou,
            match_iou=args.match_iou,
            conf_thres_iou=args.conf_thres_iou,
            max_images=args.max_images,
            workers=args.workers,
            subset_seed=args.subset_seed,
            subset_mode=args.subset_mode,
            fixed_compare_count=args.fixed_compare_count,
            category_grid_count=args.category_grid_count,
            log_images=args.log_images,
            skip_val=args.skip_val,
            split=args.split,
            timeout=args.timeout,
        )
        results.append({**metadata, **eval_result})
        print(f"  → status={eval_result.get('status', '?'):8s}  time={eval_result.get('eval_time_s', '?'):>6}s")

    df = pd.DataFrame(results)
    df = rank_dataframe(df)

    available_display = [c for c in _DISPLAY_COLS if c in df.columns]
    print(f"\n{sep}")
    print("  RANKING  (train_mAP50 ↓  ·  iou_mean ↓  ·  missed ↑  ·  fp_only ↑)")
    print(sep)
    print(df[available_display].round(4).to_string(index=False))

    df.to_csv(csv_path, index=False)
    write_markdown_report(df, md_path, batch_id)

    print(f"\n  CSV    → {csv_path}")
    print(f"  Report → {md_path}")
    print(f'  TB     : tensorboard --logdir "{tb_dir}"')

    ok_df = df[df["status"] == "OK"]
    if not ok_df.empty and "train_mAP50" in ok_df.columns:
        best = ok_df.iloc[0]
        print(f"\n  WINNER : {best['run_name']}  (train_mAP50={best.get('train_mAP50', 'N/A')}  iou_mean={best.get('iou_mean', 'N/A')})")
    print()


if __name__ == "__main__":
    main()
