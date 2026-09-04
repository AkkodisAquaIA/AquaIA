from datetime import datetime
from pathlib import Path
import csv

import torch
import yaml

from detection.utils.config_utils import find_latest_run_dir, load_run_config


def get_run_context(config):
    """Z: gather info from infer_config.yaml, with resolved_config.yaml to build a context dict for inference."""
    # Z: runs_root, run_dir
    run_cfg = config["run"]
    # Z: output_dir
    output_cfg = config["output"]
    # Z: test_data_root
    data_cfg = config["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    # Z: get run_dir from infer_config.yaml or find the latest one under runs_root
    # Z: run_dir is the directory of the training run to evaluate
    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    # Z: load resolved_config.yaml saved during training
    run_config = load_run_config(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to run inference.")

    test_data_root = Path(data_cfg["test_data_root"])
    # Z: get output_root from infer_config.yaml or use run_dir / "inference"
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Z: output_dir is the directory where inference results will be saved
    output_dir = output_root / f"{test_data_root.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "run_config": run_config,
        "train_data_root": run_config["data"]["dataset_yaml"],
        "test_data_root": str(test_data_root),
        "output_dir": output_dir,
        "device": device,
        "use_amp": use_amp,
    }


def print_test_header(ctx):
    """Z: Not used. print a header with key info about the inference run."""
    print(f"Evaluating run: {ctx['run_dir']}")
    print(f"Device: {ctx['device']} | AMP: {ctx['use_amp']}")
    print(f"Train dataset: {ctx['train_data_root']}")
    print(f"Test dataset: {ctx['test_data_root']}")
    print(f"Saving predictions under: {ctx['output_dir']}")


def build_splits(train_source, test_source, train_class_names, test_class_names, seed):
    """Z: Not used. Build train/test splits info for metrics saving."""
    return [
        ("train", train_source, train_class_names, seed),
        ("test", test_source, test_class_names, seed + 1),
    ]


def save_metrics(metrics, output_dir):
    """Z: Not used. Save inference metrics with splits to yaml and csv files in output_dir."""
    with (Path(output_dir) / "inference_metrics.yaml").open("w", encoding="utf-8") as f:
        # Z: write metrics dict to yaml file without sorted keys
        yaml.safe_dump(metrics, f, sort_keys=False)

    with (Path(output_dir) / "inference_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        # Z: create a CSV writer with the specified fieldnames
        writer = csv.DictWriter(f, fieldnames=["split", "map_50", "map_50_95", "num_samples"])
        writer.writeheader()
        # Z: write each split's (train/test) metrics to a row in the csv file
        for split_name, split_metrics in metrics.items():
            writer.writerow(
                {
                    "split": split_name,
                    "map_50": split_metrics["map_50"],
                    "map_50_95": split_metrics["map_50_95"],
                    "num_samples": split_metrics["num_samples"],
                }
            )
