from detection.utils.config_utils import load_infer_config
from detection.dino.inference.run import test_dino
from detection.yolo.inference.run import test_yolo
from detection.inference_context import get_run_context, print_test_header


def test(config):
    # config from infer_config.yaml, ctx derived from resolved_config.yaml
    ctx = get_run_context(config)
    print_test_header(ctx)

    model_config = ctx["run_config"].get("model", {})
    model_family = str(model_config.get("family", "")).lower()
    if model_family.startswith("dino"):
        return test_dino(config, ctx)
    if model_family.startswith("yolo"):
        return test_yolo(config, ctx)
    raise ValueError(f"Unsupported test backend for model config: {model_config}")


def test_from_config(config_path):
    """Z: main.py calls this function for inference."""
    config = load_infer_config(config_path)
    output_dir = test(config)
    return output_dir
