from datetime import datetime
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

from dataloading.datasets import NpyDetectionDataset, detection_collate_fn, sample_dataset
from detection.dino.dino_detector import DINODetector
from detection.metric import evaluate_map, save_metrics
from detection.utils.config_utils import find_latest_run_dir, load_run_config, load_class_names
from detection.utils.plot_utils import annotate_images_with_predictions


def load_model(run_dir, backbone_id, img_size, num_classes, device):
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


@torch.no_grad()
def predict(model, images, device, use_amp):
    images = images.to(device, non_blocking=device == "cuda")
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        outputs = model(images)
    outputs["pred_boxes"] = outputs["pred_boxes"].float()
    outputs["pred_logits"] = outputs["pred_logits"].float()
    return outputs


def save_sample_predictions(model, dataset, class_names, inference_config, output_dir, seed, device, use_amp):
    images, image_files = sample_dataset(dataset=dataset, num_samples=inference_config["num_samples"], seed=seed)
    pred_output_dir = output_dir / "predictions"

    print(f"Sampled {len(image_files)} images from {dataset.dataset_root}")
    for start in tqdm.tqdm(range(0, len(image_files), inference_config["batch"]), desc="Testing"):
        end = min(start + inference_config["batch"], len(image_files))
        batch_images = images[start:end]
        batch_files = image_files[start:end]
        outputs = predict(model, batch_images, device, use_amp)
        annotate_images_with_predictions(
            images=batch_images,
            outputs=outputs,
            class_names=class_names,
            conf_thres=inference_config["conf"],
            output_dir=pred_output_dir,
            image_files=batch_files,
        )


def evaluate_dataset(model, output_dir, dataset, inference_config, device):
    dataloader = DataLoader(
        dataset,
        batch_size=inference_config["batch"],
        shuffle=False,
        collate_fn=detection_collate_fn,
    )
    metrics = evaluate_map(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=int(model.num_classes),
        conf_thresh=inference_config["conf"],
    )
    metrics["num_samples"] = len(dataset)
    save_metrics(metrics, output_dir)
    return metrics


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

    test_data_root = str(Path(data_cfg["test_data_root"]))
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    output_dir = output_root / f"{Path(test_data_root).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_class_names = load_class_names(test_data_root)
    model = load_model(
        run_dir=run_dir,
        backbone_id=f"{run_config['model']['family']}_{run_config['model']['size']}",
        img_size=run_config["training"]["imgsz"],
        num_classes=len(test_class_names),
        device=device,
    )
    test_dataset = NpyDetectionDataset(dataset_root=str(Path(test_data_root)), device=device)

    # print_test_header(run_dir, device, use_amp, train_data_root, test_data_root, output_dir)

    save_sample_predictions(
        model=model,
        dataset=test_dataset,
        class_names=test_class_names,
        inference_config=inference_config,
        output_dir=output_dir,
        seed=inference_config["seed"],
        device=device,
        use_amp=use_amp,
    )
    evaluate_dataset(
        model=model,
        output_dir=output_dir,
        dataset=test_dataset,
        inference_config=inference_config,
        device=device,
    )

    return output_dir
