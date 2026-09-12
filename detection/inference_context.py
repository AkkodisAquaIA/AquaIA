from datetime import datetime
from pathlib import Path
import torch
from detection.utils.config_utils import find_latest_run_dir, load_run_config


def get_run_context(config):
    """Gather info from infer_config.yaml, with resolved_config.yaml, to build a context dict for inference."""
    # runs_root, run_dir
    run_cfg = config["run"]
    # output_dir
    output_cfg = config["output"]
    # test_data_root, split
    data_cfg = config["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    # Get run_dir from infer_config.yaml or find the latest one under runs_root
    # run_dir is the directory of the training run to evaluate
    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    # Load resolved_config.yaml saved during training
    run_config = load_run_config(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to run inference.")

    test_data_root = Path(data_cfg["test_data_root"])
    # Get output_root from infer_config.yaml or use run_dir / "inference"
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir is the directory where inference results will be saved
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
    """Print a header with key info about the inference run."""
    print(f"Evaluating run: {ctx['run_dir']}")
    print(f"Device: {ctx['device']} | AMP: {ctx['use_amp']}")
    print(f"Train dataset: {ctx['train_data_root']}")
    print(f"Test dataset: {ctx['test_data_root']}")
    print(f"Saving predictions under: {ctx['output_dir']}")
