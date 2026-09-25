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
    """Load best model weights, initialize model, load weights to model, set to eval mode."""
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


def infer_dino(config, context):
    """Read inference params from config, load best trained model,
    create inference dataset and dataloader, run prediction and metric evaluation,
    save inference visualizations and metrics."""
    # config from infer_config.yaml, context derived from resolved_config.yaml
    inference_config = config["inference"]
    data_cfg = config["data"]

    device = context["device"]
    run_dir = context["run_dir"]
    run_config = context["run_config"]
    infer_data_root = context["infer_data_root"]
    output_dir = context["output_dir"]

    # Reuse the mapping saved at training time. This is required because the
    # checkpoint's DETR head has one output when single_cls=True.
    single_cls = bool(run_config["training"].get("single_cls", False))
    class_name = str(run_config["data"].get("single_class_name", "specimen")).strip()
    if single_cls:
        num_classes = 1
    else:
        _, num_classes = load_class_names(infer_data_root)
    model = load_model(
        run_dir=run_dir,
        backbone_id=f"{run_config['model']['family']}_{run_config['model']['size']}",
        img_size=run_config["training"]["imgsz"],
        num_classes=num_classes,
        device=device,
    )
    imgsz = normalize_imgsz(config, "inference")
    data_split = data_cfg.get("split", "test")
    infer_dataset = JpgDetectionDataset(
        dataset_root=infer_data_root,
        data_split=data_split,
        img_size=imgsz,
        # Test targets are also converted to class 0, otherwise mAP would
        # compare mono-class predictions with the original taxon IDs.
        single_cls=single_cls,
        class_name=class_name,
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
        predict_fn=predict,
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
        predict_fn=predict,
        conf_thresh=inference_config.get("conf_thresh", 0.05),
        device=device,
    )
    print(metrics)
    save_metrics(metrics, output_dir)
    print("Inference complete!")

    return output_dir
