from detection.config_printer import print_train_config
from detection.dino.training.run import train_dino
from detection.dino.inference.run import infer_dino
from detection.inference_context import get_run_context, print_infer_header
from detection.utils.config_utils import load_infer_config, load_train_config
from detection.yolo.training.run import train_yolo
from detection.yolo.inference.run import infer_yolo


def train_from_config(config_path, resume_dir=None):
    """Load the training config and dispatch training to the selected backend."""
    config = load_train_config(config_path)
    print_train_config(config)
    model_config = config.get("model", {})
    model_family = str(model_config.get("family", "")).lower()
    if model_family.startswith("dino"):
        return train_dino(config, resume_dir=resume_dir)
    if model_family.startswith("yolo"):
        return train_yolo(config)
    raise ValueError(f"Unsupported training backend for model config: {model_config}")


def infer_from_config(config_path):
    """Load the inference config and dispatch inference to the selected backend."""
    config = load_infer_config(config_path)
    context = get_run_context(config)
    print_infer_header(context)
    model_config = context["run_config"].get("model", {})
    model_family = str(model_config.get("family", "")).lower()
    if model_family.startswith("dino"):
        return infer_dino(config, context)
    if model_family.startswith("yolo"):
        return infer_yolo(config, context)
    raise ValueError(f"Unsupported inference backend for model config: {model_config}")
