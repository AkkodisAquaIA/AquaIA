import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler

from dataloading.datasets import NpyDetectionDataset, detection_collate_fn
from Detection.DINO.dino_detector import DINODetector
from Detection.loss import SetCriterion
from Detection.metric import evaluate_map, log_epoch, print_metrics, update_log_dict
from Detection.utils.matcher import HungarianMatcher
from training.checkpoint import get_model_state_dict, save_model_checkpoint, save_training_state_checkpoint, update_best_checkpoint
from training.config_utils import save_resolved_config
    

def normalize_imgsz(config):
    model_family = str(config.get("model", {}).get("family", "")).lower()
    patch_size = 14 if model_family == "dinov2" else 16
    imgsz = int(config["training"]["imgsz"])
    rounded_imgsz = max(patch_size, round(imgsz / patch_size) * patch_size)
    if rounded_imgsz != imgsz:
        print(
            f"Warning: imgsz={imgsz} is not divisible by patch size {patch_size}. "
            f"Using imgsz={rounded_imgsz} instead."
        )
        config["training"]["imgsz"] = rounded_imgsz
    return int(config["training"]["imgsz"])


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
    dataset = NpyDetectionDataset(
        dataset_root=str(Path(config["data"]["dataset_yaml"])),
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch"],
        shuffle=True,
        num_workers=training_config["workers"],
        collate_fn=detection_collate_fn,
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch"],
        shuffle=False,
        num_workers=training_config["workers"],
        collate_fn=detection_collate_fn,
    )

    run_dir = os.path.join(output_config["project"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=output_config.get("exist_ok", True))
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    last_model_path = os.path.join(weights_dir, "last.pt")
    best_model_path = os.path.join(weights_dir, "best.pt")
    training_state_path = os.path.join(run_dir, "last_training_state.pt")
    resolved_config_path = os.path.join(run_dir, "resolved_config.yaml")
    backbone_config = config.get("model", {})
    backbone_family = str(backbone_config.get("family", "")).lower()
    backbone_size = str(backbone_config.get("size", "").lower())
    imgsz = normalize_imgsz(config)

    model = DINODetector(
        backbone_id = backbone_family + "_" + backbone_size,
        img_size=imgsz,
        device=device,
        num_classes=dataset.num_classes,
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
        num_classes=dataset.num_classes,
        matcher=matcher,
    ).to(device)

    optimizer_cls = getattr(torch.optim, training_config.get("optimizer", "AdamW"))
    optimizer = optimizer_cls(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config["lr0"],
        weight_decay=training_config["weight_decay"],
    )

    scheduler = None
    if training_config.get("cos_lr", False):
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

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_loss = float("inf")
    best_epoch = 0
    best_model_state_dict = None
    metrics_history = []
    metrics_path = os.path.join(run_dir, "metrics.npy")

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
        epoch_metrics.update(
            evaluate_map(
                model=model,
                dataloader=eval_dataloader,
                device=device,
                num_classes=dataset.num_classes,
                conf_thresh=training_config["conf"],
            )
        )
        epoch_metrics["epoch"] = epoch + 1
        print_metrics(epoch_metrics)
        metrics_history.append(epoch_metrics)

        best_loss, best_epoch, best_model_state_dict = update_best_checkpoint(
            best_loss=best_loss,
            best_epoch=best_epoch,
            best_model_state_dict=best_model_state_dict,
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            model=model,
        )
        if scheduler is not None:
            scheduler.step()

    save_model_checkpoint(path=last_model_path, model_state_dict=get_model_state_dict(model))
    save_model_checkpoint(
        path=best_model_path,
        model_state_dict=best_model_state_dict if best_model_state_dict is not None else get_model_state_dict(model),
    )
    save_training_state_checkpoint(
        path=training_state_path,
        epoch=training_config["epochs"],
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
    )

    np.save(metrics_path, metrics_history, allow_pickle=True)

    save_resolved_config(
        path=resolved_config_path,
        config=config,
        device=device,
        use_amp=use_amp,
        run_dir=run_dir,
    )

    return model
