from pathlib import Path
from torch.utils.data import DataLoader
from ultralytics import YOLO
from dataloading.datasets import JpgDetectionDataset, detection_collate_fn
from detection.metric import compute_metrics, save_metrics
from detection.utils.plot_utils import save_sample_predictions
from detection.yolo.predict import predict, normalize_imgsz


def load_model(run_dir, device):
    return YOLO(str(Path(run_dir) / "weights" / "best.pt")).to(device)


def test_yolo(config, ctx):
    # config from infer_config.yaml, ctx derived from resolved_config.yaml
    inference_config = dict(config["inference"])
    data_cfg = config["data"]

    device = ctx["device"]
    run_dir = ctx["run_dir"]
    test_data_root = ctx["test_data_root"]
    output_dir = ctx["output_dir"]

    model = load_model(run_dir, device)
    test_dataset = JpgDetectionDataset(
        dataset_root=test_data_root,
        data_split=data_cfg.get("split", "test"),
        img_size=normalize_imgsz(config, "inference"),
        device=device,
    )
    test_loader = DataLoader(test_dataset, batch_size=inference_config["batch"], shuffle=False, num_workers=3, collate_fn=detection_collate_fn)

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
