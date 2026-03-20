import ast
import re
from pathlib import Path

import yaml


def load_infer_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_run_dir(runs_root):
    run_dirs = [path for path in Path(runs_root).iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    valid_run_dirs = [
        path for path in run_dirs
        if (path / "best_model.pt").exists() or (path / "weights" / "best.pt").exists()
    ]
    if not valid_run_dirs:
        raise FileNotFoundError(f"No completed run directories with a supported best checkpoint found under {runs_root}")
    return max(valid_run_dirs, key=lambda path: path.name)


def load_class_names(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        dataset_path = dataset_path / f"{dataset_path.name}.yaml"

    yaml_text = dataset_path.read_text(encoding="utf-8")
    match = re.search(r"names:\s*(\[[\s\S]*?\])", yaml_text)
    if match is None:
        raise ValueError(f"Could not parse class names from {dataset_path}")
    return ast.literal_eval(match.group(1))


def load_run_config(run_dir):
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
