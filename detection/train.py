from detection.utils.config_utils import load_train_config
from detection.config_printer import print_train_config
from detection.dino.training.run import train_dino
from detection.yolo.training.run import train_yolo


def train(config, resume_dir=None):
    model_config = config.get("model", {})
    model_family = str(model_config.get("family", "")).lower()
    if model_family.startswith("dino"):
        return train_dino(config, resume_dir=resume_dir)
    if model_family.startswith("yolo"):
        return train_yolo(config)


def train_from_config(config_path, resume_dir=None):
    """Z: main.py calls this function for training."""
    config = load_train_config(config_path)
    print_train_config(config)
    return train(config, resume_dir=resume_dir)
