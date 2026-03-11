from pathlib import Path

from test.config_utils import find_latest_run_dir, load_infer_config, load_run_config

from test.plot_utils import plot_metrics
from test.test_dino import test_dino


def test(config):
    run_cfg = config["run"]
    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    run_config = load_run_config(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to determine which test backend to use.")

    model_config = run_config.get("model", {})
    model_family = str(model_config.get("family", "")).lower()
    if model_family.startswith("dino"):
        return test_dino(config)
    if model_family.startswith("yolo"):
        print("Pas encore refactored")
        raise NotImplementedError
    raise ValueError(f"Unsupported test backend for model config: {model_config}")


def test_from_config(config_path):
    config = load_infer_config(config_path)
    run_dir = test(config)
    plot_metrics(run_dir)
    return run_dir
