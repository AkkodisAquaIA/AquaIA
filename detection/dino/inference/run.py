from datetime import datetime
from pathlib import Path

import torch
import tqdm

from dataloading.datasets import YOLOFormatDataset, DALIDetectionDataLoader
from detection.dino.dino_detector import DINODetector
from detection.metric import compute_metrics, save_metrics
from detection.utils.config_utils import find_latest_run_dir, load_run_config, load_class_names
from detection.utils.plot_utils import save_sample_predictions
from detection.dino.predict import predict, normalize_imgsz

def load_model(run_dir, backbone_id, img_size, num_classes, device):
    checkpoint = torch.load(Path(run_dir) / "weights" / "last.pt", map_location=device)
    model = DINODetector(
        backbone_id=backbone_id,
        img_size=int(img_size),
        device=device,
        num_classes=int(num_classes),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def test_dino(config):
    inference_config = config["inference"]
    run_cfg = config["run"]
    output_cfg = config["output"]
    data_cfg = config["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    run_config = load_run_config(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to run inference.")

    test_data_root = data_cfg["test_data_root"]
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    output_dir = output_root / f"{Path(test_data_root).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, num_classes = load_class_names(test_data_root)
    model = load_model(
        run_dir=run_dir,
        backbone_id=f"{run_config['model']['family']}_{run_config['model']['size']}",
        img_size=run_config["training"]["imgsz"],
        num_classes=num_classes,
        device=device,
    )
    test_dataset = YOLOFormatDataset(
        dataset_root=test_data_root,
        data_split="test",
        batch_size=inference_config["batch"],
    )
    imgsz = normalize_imgsz(config, "inference")
    test_loader = DALIDetectionDataLoader(test_dataset, device="gpu", img_size=imgsz)
    save_sample_predictions(
        model=model,
        subset=test_dataset,
        predict_fn=predict,
        output_dir=output_dir / "inference_predictions",
        conf=inference_config.get("conf", 0.3),
        seed=inference_config["seed"],
        device=device,
    )
    metrics = compute_metrics(
        model=model,
        dataloaders=[test_loader],
        predict_fn=predict,
        conf_thresh=inference_config.get("conf", 0.3),
        device=device,
    )
    save_metrics(metrics, output_dir)
    
    return output_dir
