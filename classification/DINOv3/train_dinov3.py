#!/usr/bin/env python3
# train_dinov3.py

from __future__ import annotations
import json, time, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda
from transformers import AutoImageProcessor

from common_dinov3 import (
    ensure_dir, write_json, set_seed, get_env_info,
    DinoV3Classifier, freeze_all, unfreeze_last_n_blocks,
    save_checkpoint
)

from losses import FocalLoss

def compute_class_weights_from_imagefolder(train_ds, num_classes: int) -> torch.Tensor:
    """
    Calcule des poids de classes inverses des fréquences à partir de train_ds.targets.
    Retourne un tenseur de forme (C,).
    """
    counts = torch.bincount(torch.tensor(train_ds.targets, dtype=torch.long), minlength=num_classes).float()

    if torch.any(counts == 0):
        missing = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"Certaines classes n'ont aucun sample dans train: {missing}")

    # inverse fréquence
    weights = counts.sum() / counts

    # normalisation pour garder une échelle raisonnable
    weights = weights / weights.mean()

    return weights

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    run_dir: str = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/test_dinov3/focal_auto"
    make_subrun_with_timestamp: bool = True
    data_dir: str = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026_splited"
    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"

    epochs: int = 25
    batch_size: int = 32

    lr_head: float = 1e-3
    lr_backbone: float = 3e-5
    weight_decay: float = 0.05
    dropout: float = 0.0
    unfreeze_last_n_blocks: int = 0

    use_amp: bool = True
    num_workers: int = 4
    seed: int = 42

    early_patience: int = 7
    early_min_delta: float = 1e-4

    scheduler: str = "cosine"
    cosine_tmax_epochs: Optional[int] = None
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 1e-7

    save_last: bool = True

    # --- loss config
    loss_name: str = "focal"   # "ce" | "focal"
    focal_gamma: float = 2.0
    focal_alpha_mode: str = "auto"   # "none" | "scalar" | "auto"
    focal_alpha_scalar: Optional[float] = None
    focal_ignore_index: int = -100


CFG = Config()

# =========================
# UTILS
# =========================
def infer_block_index(name: str):
    patterns = [
        r"\.encoder\.layers\.(\d+)\.",
        r"\.encoder\.layer\.(\d+)\.",
        r"\.layers\.(\d+)\.",
        r"\.layer\.(\d+)\.",
        r"\.blocks\.(\d+)\.",
    ]
    for p in patterns:
        m = re.search(p, name)
        if m:
            return int(m.group(1))
    return None


def get_lrs(optimizer: optim.Optimizer) -> Dict[str, float]:
    return {f"group_{i}": float(g.get("lr", 0.0)) for i, g in enumerate(optimizer.param_groups)}


def build_criterion(cfg: Config, device: torch.device, train_ds=None, num_classes: Optional[int] = None):
    if cfg.loss_name.lower() == "ce":
        return nn.CrossEntropyLoss(), None

    elif cfg.loss_name.lower() == "focal":
        alpha = None

        if cfg.focal_alpha_mode == "none":
            alpha = None

        elif cfg.focal_alpha_mode == "scalar":
            alpha = cfg.focal_alpha_scalar
            if alpha is None:
                raise ValueError("focal_alpha_scalar doit être défini si focal_alpha_mode='scalar'")

        elif cfg.focal_alpha_mode == "auto":
            if train_ds is None or num_classes is None:
                raise ValueError("train_ds et num_classes sont requis si focal_alpha_mode='auto'")
            alpha = compute_class_weights_from_imagefolder(train_ds, num_classes).to(device)
            print(f"[INFO] Focal alpha auto = {alpha.detach().cpu().tolist()}")

        else:
            raise ValueError(f"focal_alpha_mode inconnu: {cfg.focal_alpha_mode}")

        criterion = FocalLoss(
            alpha=alpha,
            gamma=cfg.focal_gamma,
            reduction="mean",
            ignore_index=cfg.focal_ignore_index,
        ).to(device)

        if isinstance(alpha, torch.Tensor):
            alpha_resolved = alpha.detach().cpu().tolist()
        else:
            alpha_resolved = alpha

        return criterion, alpha_resolved

    else:
        raise ValueError(f"loss_name inconnu: {cfg.loss_name}")
    

# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for pixel_values, targets in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(pixel_values)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * targets.size(0)
        total += targets.size(0)

    return total_loss / max(1, total)


@torch.no_grad()
def eval_loss_acc(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for pixel_values, targets in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(pixel_values)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += targets.size(0)
        total_loss += float(loss.item()) * targets.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def main():
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    base = Path(CFG.run_dir)
    run_dir = base / time.strftime("%Y%m%d-%H%M%S") if CFG.make_subrun_with_timestamp else base

    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(tb_dir)
    ensure_dir(ckpt_dir)
    print(f"[INFO] run_dir={run_dir}")

    env = get_env_info()
    write_json(run_dir / "config.json", {"config": asdict(CFG), "env": env})

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/env", json.dumps(env, indent=2, ensure_ascii=False))
    writer.add_text("run/config", json.dumps(asdict(CFG), indent=2, ensure_ascii=False))

    root = Path(CFG.data_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(f"On attend {train_dir} et {val_dir}")

    processor = AutoImageProcessor.from_pretrained(CFG.model_id)

    def to_pixel_values(pil_img):
        return processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)

    transform = Compose([Lambda(to_pixel_values)])

    train_ds = ImageFolder(str(train_dir), transform=transform)
    val_ds = ImageFolder(str(val_dir), transform=transform)

    num_classes = len(train_ds.classes)
    if len(val_ds.classes) != num_classes or train_ds.classes != val_ds.classes:
        raise RuntimeError("Train/Val n'ont pas exactement les mêmes classes (dossiers).")

    write_json(run_dir / "class_to_idx.json", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DinoV3Classifier(CFG.model_id, num_classes=num_classes, dropout=CFG.dropout)
    freeze_all(model.backbone)
    unfreeze_last_n_blocks(model.backbone, CFG.unfreeze_last_n_blocks)

    for p in model.head.parameters():
        p.requires_grad = True

    model.to(device)

    criterion, focal_alpha_resolved = build_criterion(
    CFG, device, train_ds=train_ds, num_classes=num_classes
    )
    print(f"[INFO] Loss={CFG.loss_name}")

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": CFG.lr_backbone})
    param_groups.append({"params": head_params, "lr": CFG.lr_head})

    optimizer = optim.AdamW(param_groups, weight_decay=CFG.weight_decay)
    scaler = GradScaler(enabled=(CFG.use_amp and device.type == "cuda"))

    scheduler = None
    if CFG.scheduler == "cosine":
        tmax = CFG.cosine_tmax_epochs or CFG.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        print(f"[INFO] Scheduler=CosineAnnealingLR(T_max={tmax})")
    elif CFG.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=CFG.plateau_factor,
            patience=CFG.plateau_patience,
            min_lr=CFG.plateau_min_lr,
            verbose=True,
        )
        print("[INFO] Scheduler=ReduceLROnPlateau(mode=min)")
    else:
        print("[INFO] Scheduler=none")

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, CFG.use_amp)
        val_loss, val_acc = eval_loss_acc(model, val_loader, criterion, device)
        lrs = get_lrs(optimizer)

        print(f"[E{epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lrs={lrs}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        for k, v in lrs.items():
            writer.add_scalar(f"lr/{k}", v, epoch)

        if CFG.save_last:
            save_checkpoint(
                ckpt_dir / "last.pt",
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_to_idx": train_ds.class_to_idx,
                    "classes": train_ds.classes,
                    "config": asdict(CFG),
                    "env": env,
                    "model_id": CFG.model_id,
                    "focal_alpha_resolved": focal_alpha_resolved,
                },
            )

        if val_loss < best_val_loss - CFG.early_min_delta:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            best_epoch = epoch
            epochs_no_improve = 0

            save_checkpoint(
                ckpt_dir / "best.pt",
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_to_idx": train_ds.class_to_idx,
                    "classes": train_ds.classes,
                    "config": asdict(CFG),
                    "env": env,
                    "model_id": CFG.model_id,
                },
            )
            print("[INFO] best.pt mis à jour")
        else:
            epochs_no_improve += 1
            print(f"[INFO] ⏳ pas d'amélioration ({epochs_no_improve}/{CFG.early_patience})")

        if scheduler is not None:
            if CFG.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if epochs_no_improve >= CFG.early_patience:
            print("[EARLY STOPPING]")
            break

    results = {
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
    }
    write_json(run_dir / "results_trainval.json", results)
    writer.add_text("results/trainval", json.dumps(results, indent=2), 0)
    writer.close()

    print("[DONE] Train/Val terminé.")
    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] TensorBoard: tensorboard --logdir {run_dir.parent.resolve()}")


if __name__ == "__main__":
    main()