import copy
from pathlib import Path
import yaml


def load_class_names(dataset_path):
    """Read class names and class nb from dataset YAML. For training and inference."""
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        dataset_path = dataset_path / f"{dataset_path.name}.yaml"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Class names
    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, list):
        raise ValueError(f"Expected 'names' to be a list in {dataset_path}, got {type(names).__name__}")
    # Class nb
    num_class = data.get("nc") if isinstance(data, dict) else None
    if num_class is not None and num_class != len(names):
        raise ValueError(f"Number of classes 'nc' ({num_class}) does not match length of 'names' ({len(names)}) in {dataset_path}")
    return names, len(names)


def infer_output_project(config):
    """Infer an output directory path based on config. ex. 'results/task/family_size_init'. For training."""
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
    """Load train_config.yaml, complete with output dir in 'output.project'. For training."""
    # Return an absolute path
    config_path = Path(config_path).resolve()
    # Load YAML config, transform to a python dict
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    resolved_config = copy.deepcopy(config)
    # Ensure "output" in config, if not, create one with empty dict
    resolved_config.setdefault("output", {})
    # Infer an output dir only when "project" is not explicitly configured
    if not resolved_config["output"].get("project"):
        resolved_config["output"]["project"] = infer_output_project(resolved_config)
    return resolved_config


def print_train_config(config):
    """Prints config when training."""
    print("\n=== Training Config ===")
    for section_name in ("model", "data", "training", "output"):
        if section_name not in config:
            continue
        print(f"\n[{section_name}]")
        # yaml.safe_dump() transforms dict to YAML str
        # sort_key=False do not reorder alphabetically; default_flow_style=False: use multiline YAML
        print(yaml.safe_dump(config[section_name], sort_keys=False, default_flow_style=False).strip())
    print("=======================\n")


def save_resolved_config(path, config, device, use_amp, run_dir):
    """Save training config as YAML with added device, use_amp, run_dir. For training."""
    resolved_config = copy.deepcopy(config)
    resolved_config["training"]["device"] = device
    resolved_config["training"]["amp_enabled"] = use_amp
    # Ex. run_dir: results/detect/dinov3_small_pretrained/20260623_143012
    resolved_config["output"]["run_dir"] = run_dir

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)


def load_infer_config(config_path):
    """Load infer_config.yaml. For inference."""
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_run_dir(runs_root):
    """Return lastest run_dir with saved best checkpoint. For inference."""
    # Gather all dirs (not files) below runs_root
    run_dirs = [path for path in Path(runs_root).iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    valid_run_dirs = [path for path in run_dirs if (path / "best_model.pt").exists() or (path / "weights" / "best.pt").exists()]
    if not valid_run_dirs:
        raise FileNotFoundError(f"No completed run directories with a supported best checkpoint found under {runs_root}")
    return max(valid_run_dirs, key=lambda path: path.name)


def load_run_config(run_dir):
    """Load a resolved_config.yaml (with added output.project, device, use_amp, run_dir) file. For inference."""
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
