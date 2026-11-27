from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from ultralytics import YOLO
from dataloading.datasets import JpgDetectionDataset, detection_collate_fn
from detection.metric import compute_metrics, save_metrics
from detection.utils.plot_utils import save_sample_predictions
from detection.yolo.predict import predict, normalize_imgsz

# For consistency, the same inference pipeline as DINO/DETR is used here


def infer_yolo(config, context):
    # config from infer_config_yolo.yaml, context derived from resolved_config.yaml
    inference_config = dict(config["inference"])
    data_cfg = config["data"]
    # Bind the configured NMS IoU to predict so shared evaluation and visualization
    # functions can call it without accepting a YOLO-specific iou argument
    predict_fn = partial(predict, iou=float(inference_config.get("iou", 0.5)))

    device = context["device"]
    run_dir = context["run_dir"]
    infer_data_root = context["infer_data_root"]
    output_dir = context["output_dir"]

    model = YOLO(str(Path(run_dir) / "weights" / "best.pt")).to(device)
    infer_dataset = JpgDetectionDataset(
        dataset_root=infer_data_root,
        data_split=data_cfg.get("split", "test"),
        img_size=normalize_imgsz(config, "inference"),
    )
    num_workers = max(int(inference_config.get("workers", 0)), 0)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=inference_config["batch"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
    num_samples = int(inference_config.get("num_samples", 20))
    if num_samples <= 0:
        raise ValueError("inference.num_samples must be greater than 0")

    # num_samples visualization
    save_sample_predictions(
        model=model,
        subset=infer_dataset,
        predict_fn=predict_fn,
        output_dir=output_dir / "inference_predictions",
        num_samples=num_samples,
        conf=inference_config.get("conf", 0.3),
        seed=inference_config["seed"],
        device=device,
    )

    # Inference full size
    metrics = compute_metrics(
        model=model,
        dataloaders=[infer_loader],
        predict_fn=predict_fn,
        conf_thresh=inference_config.get("conf_thresh", 0.05),
        device=device,
    )
    print(metrics)
    save_metrics(metrics, output_dir)
    print("Inference complete!")

    return output_dir
