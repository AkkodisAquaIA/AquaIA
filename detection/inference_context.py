from datetime import datetime
import hashlib
from pathlib import Path
import platform
import sys
import torch
import yaml
from detection.utils.config_utils import find_latest_run_dir, load_run_config


def get_run_context(config):
    """Gather info from infer_config.yaml, with resolved_config.yaml, to build a context dict for inference."""
    # runs_root, run_dir
    run_cfg = config["run"]
    # output_dir
    output_cfg = config["output"]
    # infer_data_root, split
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

    infer_data_root = Path(data_cfg["infer_data_root"])
    configured_checkpoint = Path(run_cfg.get("checkpoint", "weights/best.pt"))
    checkpoint_path = configured_checkpoint if configured_checkpoint.is_absolute() else run_dir / configured_checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Get output_root from infer_config.yaml or use run_dir / "inference"
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir is the directory where inference results will be saved
    output_dir = output_root / f"{infer_data_root.name}_{checkpoint_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "run_config": run_config,
        "checkpoint_path": checkpoint_path,
        "train_data_root": run_config["data"]["dataset_yaml"],
        "infer_data_root": str(infer_data_root),
        "output_dir": output_dir,
        "device": device,
        "use_amp": use_amp,
    }


def print_infer_header(ctx):
    """Print a header with key info about the inference run."""
    print(f"Evaluating run: {ctx['run_dir']}")
    print(f"Device: {ctx['device']} | AMP: {ctx['use_amp']}")
    print(f"Checkpoint: {ctx['checkpoint_path']}")
    print(f"Train dataset: {ctx['train_data_root']}")
    print(f"Inference dataset: {ctx['infer_data_root']}")
    print(f"Saving predictions under: {ctx['output_dir']}")
    if ctx.get("metadata_path"):
        print(f"Inference metadata: {ctx['metadata_path']}")


def _sha256(path: Path) -> str:
    """Return a stable fingerprint for the exact checkpoint used by inference."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_yaml(path: Path):
    """Load a YAML artifact when present, otherwise return None."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def save_inference_metadata(config, context, config_path=None) -> Path:
    """Save a self-contained record of the model, checkpoint, data, and settings used for inference."""
    run_dir = Path(context["run_dir"])
    checkpoint_path = Path(context["checkpoint_path"])
    output_dir = Path(context["output_dir"])
    resolved_config_path = Path(config_path).resolve() if config_path is not None else None

    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except (ImportError, AttributeError):
        ultralytics_version = None

    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "configuration": context["run_config"].get("model", {}),
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "name": checkpoint_path.name,
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            },
        },
        "training": {
            "run_dir": str(run_dir.resolve()),
            "resolved_config": context["run_config"],
            "ultralytics_args": _load_optional_yaml(run_dir / "args.yaml"),
        },
        "inference": {
            "config_path": str(resolved_config_path) if resolved_config_path is not None else None,
            "config": config,
            "dataset_root": context["infer_data_root"],
            "split": config.get("data", {}).get("split", "test"),
            "output_dir": str(output_dir.resolve()),
            "device": context["device"],
            "amp_enabled": context["use_amp"],
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "ultralytics": ultralytics_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device": cuda_device,
        },
        "command": (
            f"{Path(sys.executable).name} main.py infer --config {resolved_config_path}"
            if resolved_config_path is not None
            else f"{Path(sys.executable).name} main.py infer"
        ),
    }

    metadata_path = output_dir / "inference_metadata.yaml"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        yaml.safe_dump(metadata, metadata_file, sort_keys=False, allow_unicode=True)
    return metadata_path
