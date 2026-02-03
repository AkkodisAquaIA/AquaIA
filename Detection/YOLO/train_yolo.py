"""
Generic Ultralytics YOLO training entrypoint for AquaIA.

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
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from ultralytics import YOLO
import argparse
from datetime import datetime, timezone

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

def write_resolved_config_yaml(save_dir: Path, resolved_cfg: Dict[str, Any], filename: str) -> Path:
    """Write the enriched configuration used for the run as a YAML file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    out_path.write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_path

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
# Device selection
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
# Run documentation (README)
# -----------------------------
def write_run_readme(save_dir: Path, cfg: Dict[str, Any], metrics: Dict[str, float]) -> None:
    """Generate a README_run.md file describing the run and the main plots."""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train = cfg.get("training", {})

    lines = [
        "# AquaIA Training Run",
        "",
        "## Run overview",
        f"- Date (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Model: {model_cfg.get('family', 'yolo')}{model_cfg.get('size', '')} ({model_cfg.get('init', '')})",
        f"- Dataset YAML: {data_cfg.get('dataset_yaml')}",
        f"- Task / mode: {train.get('task', 'detect')} / {train.get('mode', 'train')}",
        f"- Epochs: {train.get('epochs')}",
        f"- Batch size: {train.get('batch')}",
        f"- Image size: {train.get('imgsz')}",
        "",
        "All training hyperparameters, paths and runtime details are stored in "
        "the `resolved_config.after.yaml` file in this folder. It contains the "
        "exact model used, the dataset YAML path, the device, and the final "
        "save directory.",
        "- **results.png**: summaries of the whole training; the left columns "
        "show how the training losses (box / cls / dfl) decrease over epochs, "
        "the right columns show how validation metrics (precision, recall, "
        "mAP) improve.",
        "- **confusion_matrix.png**: matrix of predicted vs true classes; the "
        "diagonal corresponds to correct detections, strong off-diagonal "
        "values reveal which classes the model confuses.",
        "- **F1_curve.png**: F1 score as a function of the confidence "
        "threshold; the peak indicates the best compromise between missing "
        "objects (low recall) and producing false positives (low precision).",
        "- **PR_curve.png**: precision–recall curves for each class; curves "
        "closer to the top-right corner indicate that the model keeps high "
        "precision while covering most objects.",
        "- **P_curve.png**: precision as a function of confidence threshold; "
        "it shows how many predicted boxes are correct when you increase the "
        "confidence cut-off.",
        "- **R_curve.png**: recall as a function of confidence threshold; it "
        "shows how many true objects are still detected when you raise the "
        "confidence cut-off.",
        
        "## Final validation metrics",
        f"- mAP50-95: {metrics.get('map50_95', 0.0):.4f}",
        f"- mAP50:    {metrics.get('map50', 0.0):.4f}",
        f"- Precision: {metrics.get('precision', 0.0):.4f}",
        f"- Recall:    {metrics.get('recall', 0.0):.4f}",
    ]

    (save_dir / "README_run.md").write_text("\n".join(lines), encoding="utf-8")

# -----------------------------
# Main entrypoint
# -----------------------------

def main(config_path: str) -> Any:
    """Main: load config, prepare run, train YOLO, and save metadata."""
    # ---------- 1. Load and unpack config ----------
    cfg, cfg_dir = load_config(config_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    output_cfg = cfg.get("output", {})

    # ---------- 2. Resolve dataset path ----------
    dataset_yaml = data_cfg.get("dataset_yaml")
    if not dataset_yaml:
        raise KeyError("[CONFIG ERROR] 'data.dataset_yaml' must be set.")

    # Resolve dataset YAML relative to the config file location
    dataset_yaml = _resolve_path(cfg_dir, str(dataset_yaml))

    # ---------- 3. Build model and device ----------
    # Instantiate YOLO model (keep variable name: model)
    model_id = resolve_model_identifier(model_cfg)
    model = YOLO(model_id)

    # Device selection (portable)
    device = resolve_device(training_cfg)

    # ---------- 4. Build train arguments ----------
    # Start from training section and inject data/device
    train_args: Dict[str, Any] = dict(training_cfg)
    train_args["data"] = dataset_yaml
    train_args["device"] = device

    # ---------- 5. Resolve output project/name ----------
    project = output_cfg.get("project")

    if project:
        project_resolved = _resolve_path(cfg_dir, str(project))
        Path(project_resolved).mkdir(parents=True, exist_ok=True)
        train_args["project"] = project_resolved
    else:
        project_resolved = str((cfg_dir / "runs").resolve())
        Path(project_resolved).mkdir(parents=True, exist_ok=True)
        train_args["project"] = project_resolved

    # Auto-generate informative run name if not provided in YAML
    family = str(model_cfg.get("family", "yolo")).lower()
    size = str(model_cfg.get("size", "n")).lower()
    init = str(model_cfg.get("init", "pretrained")).lower()

    dataset_yaml_name = Path(data_cfg.get("dataset_yaml", "data.yaml")).stem
    epochs = training_cfg.get("epochs")
    batch = training_cfg.get("batch")
    imgsz = training_cfg.get("imgsz")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    default_name = (
        f"{run_timestamp}_{family}{size}_{dataset_yaml_name}"
        f"_e{epochs}_bs{batch}_img{imgsz}_{init}"
    )

    # If output.name is set in YAML, it overrides the default
    name = output_cfg.get("name") or default_name
    train_args["name"] = str(name)

    # Merge a few output flags into train args
    for k in ("exist_ok", "save", "save_period", "plots"):
        if k in output_cfg:
            train_args[k] = output_cfg[k]

    # ---------- 6. Prepare resolved config snapshot ----------
    resolved_cfg = {
        "model": model_cfg,
        "data": {
            "dataset_yaml": dataset_yaml,
        },
        "training": training_cfg,
        "output": output_cfg,
        "resolved": {
            # Time & reproducibility
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),

            # Effective runtime values
            "model_id": model_id,
            "device": str(device),

            # Where Ultralytics is instructed to write
            "project": train_args.get("project"),
            "name": train_args.get("name"),

            # Filled after training
            "save_dir": None,
            "weights": {
                "best": None,
                "last": None,
            },
            "metrics": None,
        },
    }
    
    write_cfg = output_cfg.get("write_config", True)
    cfg_after_name = str(output_cfg.get("config_filename_after", "resolved_config.after.yaml"))
    run_dir_guess = Path(str(train_args.get("project"))) / str(train_args.get("name"))

    # ---------- 7. Launch training ----------
    results = model.train(**train_args)

    # ---------- 8. Collect outputs and metrics ----------
    final_dir = getattr(results, "save_dir", None)
    if final_dir is not None:
        final_dir = Path(final_dir)
        resolved_cfg["resolved"]["save_dir"] = str(final_dir)

        best = final_dir / "weights" / "best.pt"
        last = final_dir / "weights" / "last.pt"
        resolved_cfg["resolved"]["weights"]["best"] = str(best) if best.exists() else None
        resolved_cfg["resolved"]["weights"]["last"] = str(last) if last.exists() else None

        metrics = getattr(results, "results_dict", {}) or {}
        resolved_cfg["resolved"]["metrics"] = {
            "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
            "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
            "precision": float(metrics.get("metrics/precision(B)", 0.0)),
            "recall": float(metrics.get("metrics/recall(B)", 0.0)),
        }

        # Write human-readable summary README in the run folder
        write_run_readme(final_dir, cfg, resolved_cfg["resolved"]["metrics"])

        # Snapshot of final resolved config after training
        if write_cfg:
            target_dir = Path(final_dir) if final_dir is not None else run_dir_guess
            write_resolved_config_yaml(target_dir, resolved_cfg, cfg_after_name)

    return results

# -----------------------------
# CLI entrypoint
# -----------------------------

def parse_args() -> str:
    """Parse command-line arguments and return the config path."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO model from a YAML config."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="yolo_train_config.yaml",
        help="Path to the training config YAML file.",
    )
    args = parser.parse_args()
    return args.config

if __name__ == "__main__":
    # Parse CLI args and run training
    cfg_path = parse_args()
    out = main(cfg_path)

    # Print where results were saved
    savedir = getattr(out, "save_dir", None)
    if savedir is not None:
        print(f"Training finished. Results saved to {Path(savedir)}")
    else:
        print("Training finished.")

    # Print compact KPIs as JSON for easy logging/automation
    metrics = getattr(out, "results_dict", {}) or {}
    pretty = {
        "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
    }
    print(json.dumps(pretty, indent=2))