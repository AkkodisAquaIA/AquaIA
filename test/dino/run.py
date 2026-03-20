from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

from dataloading.datasets import NpyDetectionDataset, detection_collate_fn, sample_dataset
from Detection.DINO.dino_detector import DINODetector
from Detection.metric import evaluate_map, print_metrics
from test.config_utils import load_class_names
from test.plot_utils import annotate_images_with_predictions
from test.run_utils import build_splits, get_run_context, print_test_header, save_metrics


def load_model(run_dir, img_size, num_classes, device):
    checkpoint = torch.load(Path(run_dir) / "best_model.pt", map_location=device)
    model = DINODetector(
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


def save_sample_predictions(model, dataset, split_name, class_names, inference_config, output_dir, seed, device, use_amp):
    images, image_files = sample_dataset(dataset=dataset, num_samples=inference_config["num_samples"], seed=seed)
    split_output_dir = output_dir / f"{split_name}_predictions"

    print(f"{split_name}: sampled {len(image_files)} images from {dataset.dataset_root}")
    for start in tqdm.tqdm(range(0, len(image_files), inference_config["batch"]), desc=f"Testing {split_name}"):
        end = min(start + inference_config["batch"], len(image_files))
        batch_images = images[start:end]
        batch_files = image_files[start:end]
        outputs = predict(model, batch_images, device, use_amp)
        annotate_images_with_predictions(
            images=batch_images,
            outputs=outputs,
            class_names=class_names,
            conf_thres=inference_config["conf"],
            output_dir=split_output_dir,
            image_files=batch_files,
        )


def evaluate_dataset(model, dataset, split_name, inference_config, device):
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
    print(f"{split_name} metrics:")
    print_metrics(metrics)
    return metrics


def test_dino(config):
    inference_config = config["inference"]
    ctx = get_run_context(config)

    train_class_names = load_class_names(ctx["train_data_root"])
    test_class_names = load_class_names(ctx["test_data_root"])
    model = load_model(
        run_dir=ctx["run_dir"],
        img_size=ctx["run_config"]["training"]["imgsz"],
        num_classes=len(train_class_names),
        device=ctx["device"],
    )
    train_dataset = NpyDetectionDataset(dataset_root=str(Path(ctx["train_data_root"])), device=ctx["device"])
    test_dataset = NpyDetectionDataset(dataset_root=str(Path(ctx["test_data_root"])), device=ctx["device"])

    print_test_header(ctx)

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
            output_dir=ctx["output_dir"],
            seed=seed,
            device=ctx["device"],
            use_amp=ctx["use_amp"],
        )
        metrics[split_name] = evaluate_dataset(
            model=model,
            dataset=dataset,
            split_name=split_name,
            inference_config=inference_config,
            device=ctx["device"],
        )

    save_metrics(metrics, ctx["output_dir"])
    return ctx["run_dir"]
