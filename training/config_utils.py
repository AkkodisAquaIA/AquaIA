import copy
from pathlib import Path

import yaml


def load_train_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    resolved_config = copy.deepcopy(config)
    return resolved_config


def save_resolved_config(path, config, device, use_amp, run_dir, img_size, num_classes):
    resolved_config = copy.deepcopy(config)
    resolved_config["training"]["device"] = device
    resolved_config["training"]["amp_enabled"] = use_amp
    resolved_config["output"]["run_dir"] = run_dir
    resolved_config["data"]["img_size"] = int(img_size)
    resolved_config["data"]["num_classes"] = int(num_classes)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)
