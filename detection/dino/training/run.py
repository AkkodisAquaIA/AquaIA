import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import get_scheduler

from dataloading.datasets import NpyDetectionDataset, detection_collate_fn, sample_dataset
from detection.dino.dino_detector import DINODetector
from detection.dino.loss import SetCriterion
from detection.metric import evaluate_map, log_epoch, print_metrics, update_log_dict
from detection.dino.utils.matcher import HungarianMatcher
from detection.checkpoint import save_model_checkpoint, save_training_state_checkpoint
from detection.utils.config_utils import save_resolved_config
from detection.utils.plot_utils import plot_metrics, annotate_images_with_predictions


@torch.no_grad()
def save_sample_predictions(model, subset, output_dir, num_samples=20, conf=0.3, seed=0, device="cuda", use_amp=True):
    images, image_files = sample_dataset(dataset=subset, num_samples=num_samples, seed=seed)
    print(image_files)
    print(f"Sampled {len(image_files)} images from {subset.dataset.dataset_root}")
    images = images.to(device, non_blocking=device == "cuda")
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        outputs = model(images)
    outputs["pred_boxes"] = outputs["pred_boxes"].float()
    outputs["pred_logits"] = outputs["pred_logits"].float()
    annotate_images_with_predictions(
        images=images,
        outputs=outputs,
        class_names=subset.dataset.class_names,
        conf_thres=conf,
        output_dir=output_dir,
        image_files=image_files,
    )


def normalize_imgsz(config):
    model_family = str(config.get("model", {}).get("family", "")).lower()
    patch_size = 14 if model_family == "dinov2" else 16
    imgsz = int(config["training"]["imgsz"])
    rounded_imgsz = max(patch_size, round(imgsz / patch_size) * patch_size)
    if rounded_imgsz != imgsz:
        print(f"Warning: imgsz={imgsz} is not divisible by patch size {patch_size}. Using imgsz={rounded_imgsz} instead.")
        config["training"]["imgsz"] = rounded_imgsz
    return int(config["training"]["imgsz"])


def get_datasets(data_yaml_path, device, train_fraction=0.9):
    # Compute random split for train and eval set
    dataset = NpyDetectionDataset(
        dataset_root=data_yaml_path,
        device=device,
    )
    train_size = int(train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    num_classes = dataset.num_classes
    return train_dataset, test_dataset, num_classes


def build_scheduler(training_config, optimizer):
    if not training_config.get("cos_lr", False):
        return None
    warmup_ratio = float(training_config.get("warmup_ratio", 0.0))
    warmup_steps = int(training_config["epochs"] * warmup_ratio)
    if warmup_ratio > 0.0:
        warmup_steps = max(warmup_steps, 1)
    scheduler = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_config["epochs"],
        scheduler_specific_kwargs={"min_lr_rate": training_config.get("lrf", 0.01)},
    )
    return scheduler


def compute_metrics(model, train_dataloader, eval_dataloader, metrics_dict, device, num_classes, conf_thresh):
    train_metrics = evaluate_map(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_classes=num_classes,
        conf_thresh=conf_thresh,
    )
    metrics_dict["train_map_50"] = train_metrics["map_50"]
    metrics_dict["train_map_50_95"] = train_metrics["map_50_95"]
    eval_metrics = evaluate_map(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        num_classes=num_classes,
        conf_thresh=conf_thresh,
    )
    metrics_dict.update(eval_metrics)


# TODO : the image size specified inside the trainin_conig.yaml and the npy is not the same
def train_dino(config):
    training_config = config["training"]
    output_config = config["output"]

    configured_device = training_config.get("device", "auto")
    if configured_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = configured_device

    use_amp = device == "cuda"

    train_set, test_set, num_classes = get_datasets(config["data"]["dataset_yaml"], device)

    # === Setup dataloaders ===
    dataloader = DataLoader(
        train_set,
        batch_size=training_config["batch"],
        shuffle=True,
        num_workers=training_config["workers"],
        collate_fn=detection_collate_fn,
    )
    eval_dataloader = DataLoader(
        test_set,
        batch_size=training_config["batch"],
        shuffle=False,
        num_workers=training_config["workers"],
        collate_fn=detection_collate_fn,
    )

    # === Config, save path and fun ===
    # root folder for training outputs (weights, logs, resolved config)
    run_dir = os.path.join(output_config["project"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=output_config.get("exist_ok", True))
    # folder to save model weights (best and last)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # === DINO related stuff ===
    backbone_config = config.get("model", {})
    backbone_family = str(backbone_config.get("family", "")).lower()
    backbone_size = str(backbone_config.get("size", "").lower())
    imgsz = normalize_imgsz(config)

    model = DINODetector(
        backbone_id=backbone_family + "_" + backbone_size,
        img_size=imgsz,
        device=device,
        num_classes=num_classes,
    ).to(device)

    matcher = HungarianMatcher(
        cost_class=training_config["cost_class"],
        cost_bbox=training_config["cost_bbox"],
        cost_giou=training_config["cost_giou"],
        cost_bbox_type=training_config["cost_bbox_type"],
    )
    loss_weight_dict = {
        "loss_ce": training_config["cls"],
        "loss_bbox": training_config["box"],
        "loss_giou": training_config["giou"],
    }
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
    ).to(device)

    optimizer_cls = getattr(torch.optim, training_config.get("optimizer", "AdamW"))
    optimizer = optimizer_cls(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config["lr0"],
        weight_decay=training_config["weight_decay"],
    )

    scheduler = build_scheduler(training_config, optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_metric = 0.0
    best_metric_dict = None
    metrics_history = []

    model.train()
    criterion.train()
    if training_config.get("compile", False):
        model = torch.compile(model)

    for epoch in range(training_config["epochs"]):
        epoch_loss = 0.0
        log_dict = {"avg": 0.0}
        progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{training_config['epochs']}")
        for images, targets, _ in progress:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict[key] * loss_weight_dict[key] for key in loss_dict if key in loss_weight_dict)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(total_loss.item())
            progress.set_postfix(loss=float(total_loss.item()))
            update_log_dict(log_dict, loss_dict, epoch_loss)

        epoch_metrics = log_epoch(log_dict, max(len(dataloader), 1))

        # Add map50 and mAP50-95 evaluation at the end of each epoch (train and eval datasets)
        compute_metrics(
            model=model,
            train_dataloader=dataloader,
            eval_dataloader=eval_dataloader,
            metrics_dict=epoch_metrics,
            device=device,
            num_classes=num_classes,
            conf_thresh=training_config.get("conf_thresh", 0.05),
        )
        epoch_metrics["epoch"] = epoch + 1
        print_metrics(epoch_metrics)
        metrics_history.append(epoch_metrics)

        if best_metric < epoch_metrics["map_50_95"] or best_metric_dict is None:
            best_metric = epoch_metrics["map_50_95"]
            best_metric_dict = epoch_metrics.copy()
            # save best model
            save_model_checkpoint(
                path=os.path.join(weights_dir, "best.pt"),
                model=model,
            )
            print(f"New best model found at epoch {epoch + 1} with mAP50-95: {best_metric:.4f}")

        if scheduler is not None:
            scheduler.step()

    # save last model
    save_model_checkpoint(path=os.path.join(weights_dir, "last.pt"), model=model)
    # Save training state (optimizer, scaler and scheduler states) for potential resuming
    save_training_state_checkpoint(
        path=os.path.join(run_dir, "last_training_state.pt"),
        epoch=training_config["epochs"],
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
    )

    # Save and plot metrics
    np.save(os.path.join(run_dir, "metrics.npy"), metrics_history, allow_pickle=True)
    plot_metrics(run_dir)

    # Save best metrics
    np.save(os.path.join(run_dir, "best_metric.npy"), best_metric_dict, allow_pickle=True)

    save_resolved_config(
        path=os.path.join(run_dir, "resolved_config.yaml"),
        config=config,
        device=device,
        use_amp=use_amp,
        run_dir=run_dir,
    )

    # === Save some sampled predictions with the best model ===
    best_checkpoint = torch.load(os.path.join(weights_dir, "best.pt"), map_location=device)
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    target_model.load_state_dict(best_checkpoint["model_state_dict"])

    # Eval dataset
    save_sample_predictions(
        model=model,
        subset=test_set,
        output_dir=Path(run_dir) / "eval_predictions",
        conf=0.3,
        seed=42,
        device=device,
        use_amp=use_amp,
    )
    # Train dataset
    save_sample_predictions(
        model=model,
        subset=train_set,
        output_dir=Path(run_dir) / "train_predictions",
        conf=0.3,
        seed=42,
        device=device,
        use_amp=use_amp,
    )

    return model
