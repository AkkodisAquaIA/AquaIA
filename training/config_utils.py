import copy
from pathlib import Path
import yaml


def infer_output_project(config):
    task = str(config["training"]["task"]).strip().lower()
    model_config = config["model"]
    family = str(model_config["family"]).strip().lower()
    size = str(model_config.get("size", "")).strip().lower()
    init = str(model_config.get("init", "")).strip().lower()

    model_parts = [family]
    if size:
        model_parts.append(size)
    if init:
        model_parts.append(init)

    return str(Path("results") / task / "_".join(model_parts))


def load_train_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("output", {})
    resolved_config["output"]["project"] = infer_output_project(resolved_config)
    return resolved_config


def save_resolved_config(path, config, device, use_amp, run_dir):
    resolved_config = copy.deepcopy(config)
    resolved_config["training"]["device"] = device
    resolved_config["training"]["amp_enabled"] = use_amp
    resolved_config["output"]["run_dir"] = run_dir

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)
