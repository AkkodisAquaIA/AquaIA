"""
Generic Ultralytics YOLO training entrypoint for AquaIA (Docker/OVH ready).

Highlights:
- Model selection is configurable via family/size/init.
- All training hyperparameters are defined under the "training" section in YAML.
- No hardcoding of GPU model/capacity; works on any CUDA GPU or CPU.
- Docker-safe paths: relative paths are resolved from the YAML directory.
- Environment variables in YAML are supported (e.g., ${DATASET_YAML}).
- Prints a compact JSON of key validation metrics at the end.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from ultralytics import YOLO


# -----------------------------
# Helpers: env expansion & path resolution
# -----------------------------
def _expand_env(value: Any) -> Any:
    """Expand environment variables in strings; leave other types unchanged."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _expand_env_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in a nested dict."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _expand_env_in_dict(v)
        elif isinstance(v, list):
            out[k] = [_expand_env(x) for x in v]
        else:
            out[k] = _expand_env(v)
    return out


def _resolve_path(base_dir: Path, maybe_path: str) -> str:
    """
    Resolve a path in a container-friendly way:
    - Expands env vars and "~"
    - If absolute -> return resolved absolute path
    - If relative -> resolve relative to the YAML directory (base_dir)
    """
    s = os.path.expandvars(maybe_path)
    p = Path(s).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((base_dir / p).expanduser().resolve())


# -----------------------------
# Config loading
# -----------------------------
def load_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path]:
    """
    Load the YAML configuration and return (config, config_dir).

    config_dir is used to resolve relative paths robustly.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"[CONFIG ERROR] YAML config not found: {cfg_path}")

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"[CONFIG ERROR] Failed to parse YAML: {cfg_path}") from e

    if not isinstance(cfg, dict):
        raise ValueError(f"[CONFIG ERROR] YAML root must be a dict: {cfg_path}")

    cfg = _expand_env_in_dict(cfg)

    # Required sections
    for key in ("model", "data", "training"):
        if key not in cfg:
            raise KeyError(f"[CONFIG ERROR] Missing top-level key '{key}' in {cfg_path}")

    return cfg, cfg_path.parent


# -----------------------------
# Dataset sanity checks
# -----------------------------
def assert_labels_exist(dataset_yaml: str | Path) -> None:
    """
    Ensure train/val label folders exist and contain label .txt files.

    Expected convention:
    - .../images/... for images
    - .../labels/... for labels
    """
    p = Path(dataset_yaml).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"[DATA ERROR] Dataset YAML not found: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"[DATA ERROR] Dataset YAML root must be a dict: {p}")

    # Ultralytics dataset YAML convention: "path" is the dataset root.
    # If missing, fallback to the dataset YAML directory.
    root = Path(data.get("path") or p.parent).expanduser().resolve()

    def check_split(split: str) -> None:
        rel = data.get(split)
        if not rel:
            return  # split is optional

        img_dir = (root / rel).expanduser().resolve()
        if not img_dir.exists():
            raise FileNotFoundError(f"[DATA ERROR] {split} images dir not found: {img_dir}")

        parts = list(img_dir.parts)
        if "images" not in parts:
            raise ValueError(f"[DATA ERROR] Cannot infer labels dir (no 'images' in path): {img_dir}")

        parts[parts.index("images")] = "labels"
        lab_dir = Path(*parts)

        if not lab_dir.exists() or not any(lab_dir.glob("*.txt")):
            raise FileNotFoundError(f"[DATA ERROR] Missing/empty labels dir for {split}: {lab_dir}")

    check_split("train")
    check_split("val")


# -----------------------------
# Model resolution (family/size/init)
# -----------------------------
def resolve_model_identifier(model_cfg: Dict[str, Any]) -> str:
    """
    Resolve the model identifier passed to Ultralytics YOLO().
    
    - model.family + model.size + model.init
      - init="pretrained" -> "{family}{size}.pt"
      - init="random"     -> "{family}{size}.yaml"
    """
    family = str(model_cfg.get("family", "yolo11")).strip().lower()
    size = str(model_cfg.get("size", "n")).strip().lower()
    init = str(model_cfg.get("init", "pretrained")).strip().lower()

    if size not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"[CONFIG ERROR] Unsupported size '{size}'. Expected one of n/s/m/l/x.")

    base = f"{family}{size}"
    if init == "random":
        return f"{base}.yaml"
    if init == "pretrained":
        return f"{base}.pt"

    raise ValueError(f"[CONFIG ERROR] Unsupported model.init '{init}'. Use 'pretrained' or 'random'.")


# -----------------------------
# Device selection (portable)
# -----------------------------
def resolve_device(training_cfg: Dict[str, Any]) -> str | int:
    """
    Decide device without hardcoding GPU name/capacity.

    - If training.device is set in YAML -> pass-through
    - Else: use GPU 0 if CUDA available, otherwise 'cpu'
    """
    if "device" in training_cfg and training_cfg["device"] is not None:
        return training_cfg["device"]
    return 0 if torch.cuda.is_available() else "cpu"


# -----------------------------
# Main entrypoint
# -----------------------------
def main(config_path: str) -> Any:
    """Load config, validate dataset, instantiate model, and launch training."""
    cfg, cfg_dir = load_config(config_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    output_cfg = cfg.get("output", {})

    dataset_yaml = data_cfg.get("dataset_yaml")
    if not dataset_yaml:
        raise KeyError("[CONFIG ERROR] 'data.datase t_yaml' must be set.")

    # Resolve dataset YAML relative to the config file location
    dataset_yaml = _resolve_path(cfg_dir, str(dataset_yaml))

    # Fail fast if dataset structure is broken
    assert_labels_exist(dataset_yaml)

    # Instantiate YOLO model (keep variable name: model)
    model_id = resolve_model_identifier(model_cfg)
    model = YOLO(model_id)

    # Device selection (portable)
    device = resolve_device(training_cfg)

    # Build training args (pass-through from YAML)
    train_args: Dict[str, Any] = dict(training_cfg)
    train_args["data"] = dataset_yaml
    train_args["device"] = device

    # Output controls (Docker-friendly)
    project = output_cfg.get("project")
    name = output_cfg.get("name")

    if project:
        project_resolved = _resolve_path(cfg_dir, str(project))
        Path(project_resolved).mkdir(parents=True, exist_ok=True)
        train_args["project"] = project_resolved

    if name:
        train_args["name"] = str(name)

    # Merge a few output flags into train args
    for k in ("exist_ok", "save", "save_period", "plots"):
        if k in output_cfg:
            train_args[k] = output_cfg[k]

    results = model.train(**train_args)
    return results


if __name__ == "__main__":

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "yolo_train_config.yaml"

    # main() returns the Ultralytics results object (DetMetrics for detection tasks)
    out = main(cfg_path)

    # Prefer reading save_dir from the returned results object (robust across Ultralytics versions)
    save_dir = getattr(out, "save_dir", None)
    if save_dir is not None:
        print(f"Training finished. Results saved to: {Path(save_dir)}")
    else:
        print("Training finished.")

    # Print compact KPIs as JSON
    metrics = getattr(out, "results_dict", {}) or {}
    pretty = {
        "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
    }
    print(json.dumps(pretty, indent=2))
