from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataloading.datasets import JpgDetectionDataset, detection_collate_fn
from detection.dino.dino_detector import DINODetector
from detection.metric import compute_metrics, save_metrics
from detection.utils.config_utils import load_class_names
from detection.utils.plot_utils import save_sample_predictions
from detection.dino.predict import predict, normalize_imgsz


def load_model(run_dir, backbone_id, img_size, num_classes, device):
    """Z: load best model weights, initialize model, load weights to model, set to eval mode."""
    checkpoint = torch.load(Path(run_dir) / "weights" / "best.pt", map_location=device)
    model = DINODetector(
        backbone_id=backbone_id,
        img_size=int(img_size),
        device=device,
        num_classes=int(num_classes),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def test_dino(config, ctx):
    """Z: Read inference parameters from configuration, load best trained model,
    create test dataset and dataloader, run prediction and metric evaluation,
    save inference visualizations and metrics."""
    # config from infer_config.yaml, ctx derived from resolved_config.yaml
    inference_config = config["inference"]
    data_cfg = config["data"]

    device = ctx["device"]
    run_dir = ctx["run_dir"]
    run_config = ctx["run_config"]
    test_data_root = ctx["test_data_root"]
    output_dir = ctx["output_dir"]

    _, num_classes = load_class_names(test_data_root)
    model = load_model(
        run_dir=run_dir,
        backbone_id=f"{run_config['model']['family']}_{run_config['model']['size']}",
        img_size=run_config["training"]["imgsz"],
        num_classes=num_classes,
        device=device,
    )
    imgsz = normalize_imgsz(config, "inference")
    data_split = data_cfg.get("split", "test")
    test_dataset = JpgDetectionDataset(
        dataset_root=test_data_root,
        data_split=data_split,
        img_size=imgsz,
        device=device,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_config["batch"],
        shuffle=False,
        num_workers=3,
        collate_fn=detection_collate_fn,
    )
    save_sample_predictions(
        model=model,
        subset=test_dataset,
        predict_fn=predict,
        output_dir=output_dir / "inference_predictions",
        conf=inference_config.get("conf", 0.3),
        seed=inference_config["seed"],
        device=device,
    )
    model.eval()
    metrics = compute_metrics(
        model=model,
        dataloaders=[test_loader],
        predict_fn=predict,
        conf_thresh=inference_config.get("conf_thresh", 0.05),
        device=device,
    )
    print(metrics)
    save_metrics(metrics, output_dir)

    return output_dir
