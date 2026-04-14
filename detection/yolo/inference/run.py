from datetime import datetime
from pathlib import Path

import tqdm
from torch.utils.data import DataLoader
from ultralytics import YOLO

from dataloading.datasets import NpyDetectionDataset, detection_collate_fn, sample_dataset
from detection.metric import evaluate_map, print_metrics
from detection.utils.config_utils import find_latest_run_dir, load_run_config
from test.config_utils import load_class_names
from test.plot_utils import annotate_yolo_predictions
from test.run_utils import build_splits, print_test_header, save_metrics


def load_model(run_dir, device):
    return YOLO(str(Path(run_dir) / "weights" / "best.pt")).to(device)


def predict(model, image_files, device, inference_config):
    return model.predict(
        source=image_files,
        imgsz=inference_config["imgsz"],
        conf=inference_config["conf"],
        device=device,
        verbose=False,
        batch=len(image_files),
    )


def save_sample_predictions(model, dataset, split_name, class_names, inference_config, output_dir, seed, device):
    _, image_files = sample_dataset(dataset=dataset, num_samples=inference_config["num_samples"], seed=seed)
    split_output_dir = output_dir / f"{split_name}_predictions"

    print(f"{split_name}: sampled {len(image_files)} images from {dataset.dataset_root}")
    for start in tqdm.tqdm(range(0, len(image_files), inference_config["batch"]), desc=f"Testing {split_name}"):
        end = min(start + inference_config["batch"], len(image_files))
        batch_files = image_files[start:end]
        results = predict(model, batch_files, device, inference_config)
        annotate_yolo_predictions(
            results=results,
            class_names=class_names,
            conf_thres=inference_config["conf"],
            output_dir=split_output_dir,
            image_files=batch_files,
        )


def evaluate_dataset(model, dataset, split_name, inference_config, device, num_classes, metric_conf):
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
        num_classes=int(num_classes),
        conf_thresh=metric_conf,
    )
    metrics["num_samples"] = len(dataset)
    print(f"{split_name} metrics:")
    print_metrics(metrics)
    return metrics


def test_yolo(config):
    inference_config = dict(config["inference"])
    run_cfg = config["run"]
    output_cfg = config["output"]
    data_cfg = config["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    run_config = load_run_config(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to run inference.")

    train_data_root = run_config["data"]["dataset_yaml"]
    test_data_root = str(Path(data_cfg["test_data_root"]))
    output_root = Path(output_cfg["output_dir"]) if output_cfg.get("output_dir") else run_dir / "inference"
    output_dir = output_root / f"{Path(test_data_root).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_conf = float(run_config.get("training", {}).get("conf", 0.001))

    train_class_names = load_class_names(train_data_root)
    test_class_names = load_class_names(test_data_root)
    model = load_model(run_dir, device)
    train_dataset = NpyDetectionDataset(dataset_root=str(Path(train_data_root)), device=device)
    test_dataset = NpyDetectionDataset(dataset_root=str(Path(test_data_root)), device=device)

    print_test_header(run_dir, device, use_amp, train_data_root, test_data_root, output_dir)

    metrics = {}
    for split_name, dataset, class_names, seed in build_splits(
        train_dataset,
        test_dataset,
        train_class_names,
        test_class_names,
        inference_config["seed"],
    ):
        save_sample_predictions(
            model=model,
            dataset=dataset,
            split_name=split_name,
            class_names=class_names,
            inference_config=inference_config,
            output_dir=output_dir,
            seed=seed,
            device=device,
        )
        metrics[split_name] = evaluate_dataset(
            model=model,
            dataset=dataset,
            split_name=split_name,
            inference_config=inference_config,
            device=device,
            num_classes=len(class_names),
            metric_conf=metric_conf,
        )

    save_metrics(metrics, output_dir)
    return output_dir
