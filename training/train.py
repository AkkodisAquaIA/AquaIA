from training.config_utils import load_train_config
from training.dino.run import train_dino


def train_yolo(config):
    print("Pas encore refactored")
    raise NotImplementedError


def train(config):
    model_config = config.get("model", {})
    model_family = str(model_config.get("family", "")).lower()
    if model_family.startswith("dino"):
        return train_dino(config)
    if model_family.startswith("yolo"):
        return train_yolo(config)


def train_from_config(config_path):
    config = load_train_config(config_path)
    return train(config)


