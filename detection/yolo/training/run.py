from __future__ import annotations
import os
from typing import Any, Dict

import torch
from ultralytics import YOLO
from datetime import datetime
from detection.utils.config_utils import save_resolved_config


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
# Main entrypoint
# -----------------------------


def train_yolo(config) -> Any:
    model_config = config["model"]
    training_config = config["training"]
    output_config = config.get("output", {})

    # ---------- 2. Resolve dataset path ----------
    dataset_config = config["data"]["dataset_yaml"]

    # ---------- 3. Build model and device ----------
    # Instantiate YOLO model (keep variable name: model)
    model_id = resolve_model_identifier(model_config)
    model = YOLO(model_id)

    configured_device = training_config.get("device", "auto")
    if configured_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = configured_device
    use_amp = device == "cuda"

    # ---------- 4. Build train arguments ----------
    # Start from training section and inject data/device
    train_args: Dict[str, Any] = dict(training_config)
    train_args["data"] = dataset_config
    train_args["device"] = device
    train_args["amp"] = use_amp

    # ---------- 5. Resolve output project/name ----------
    run_dir = os.path.join(output_config["project"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=output_config.get("exist_ok", True))
    train_args["save_dir"] = run_dir

    # Merge a few output flags into train args
    for k in ("exist_ok", "save_period", "plots"):
        if k in output_config:
            train_args[k] = output_config[k]

    resolved_config_path = os.path.join(run_dir, "resolved_config.yaml")
    save_resolved_config(
        path=resolved_config_path,
        config=config,
        device=device,
        use_amp=use_amp,
        run_dir=run_dir,
    )

    return model.train(**train_args)
