from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader
from ultralytics import YOLO
import torch

from dataloading.datasets import JpgDALIDataset, collate_dali
from detection.utils.config_utils import find_latest_run_dir, load_run_config
from detection.metric import compute_metrics, save_metrics
from detection.utils.plot_utils import save_sample_predictions
from detection.yolo.predict import predict, normalize_imgsz


def load_model(run_dir, device):
    return YOLO(str(Path(run_dir) / "weights" / "best.pt")).to(device)


def test_yolo(config):
    inference_config = dict(config["inference"])
    run_cfg = config["run"]
    output_cfg = config["output"]
    data_cfg = config["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    run_config = load_run_config(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to run inference.")

    test_data_root = str(Path(data_cfg["test_data_root"]))
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    output_dir = output_root / f"{Path(test_data_root).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(run_dir, device)
    test_dataset = JpgDALIDataset(
        dataset_root=test_data_root,
        data_split="test",
        img_size=normalize_imgsz(config, "inference"),
        batch_size=inference_config["batch"],
    )
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.batch_size, shuffle=False, num_workers=0, collate_fn=collate_dali)

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
