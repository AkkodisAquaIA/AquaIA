#!/usr/bin/env python3
"""Entrainement DINOv3 avec DANN sur deux domaines.

Arborescence attendue :
    DATA_DIR/a
      source/
        train/<taxon>/*
        val/<taxon>/*
        test/<taxon>/*
      target/
        train/<taxon>/*
        val/<taxon>/*
        test/<taxon>/*

Les domaines source et cible peuvent contenir des ensembles de taxons differents.
Le script construit l'union globale des classes et remappe tous les labels vers
ce referentiel commun.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModel

from common_dinov3 import (
    ensure_dir,
    freeze_all,
    get_env_info,
    infer_block_index,
    save_checkpoint,
    set_seed,
    unfreeze_last_n_blocks,
    write_json,
)
from losses import FocalLoss


torch.cuda.empty_cache()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    # Chemins
    data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "DATA_DIR",
            r"C:\Users\Thibaud.VANDAMME\data_10classes_cropped_balanced",
        )
    )
    run_dir: str = field(
        default_factory=lambda: os.environ.get(
            "RUN_DIR",
            r"C:\Users\Thibaud.VANDAMME\AquaIA-development\results_dann",
        )
    )
    make_subrun_with_timestamp: bool = True

    # Modele
    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    local_files_only: bool = True

    # Entrainement
    epochs: int = int(os.environ.get("EPOCHS", "50"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "32"))
    num_workers: int = int(os.environ.get("NUM_WORKERS", "4"))
    seed: int = 42
    use_amp: bool = True

    dropout: float = 0.0
    domain_hidden_dim: int = 128
    unfreeze_last_n_blocks: int = int(os.environ.get("UNFREEZE", "1"))

    lr_head: float = 1e-3
    lr_domain: float = 1e-4
    lr_backbone: float = 3e-5
    weight_decay: float = 0.05

    # L_total = ws*L_source + wt*L_target + beta*L_domain
    source_taxon_weight: float = 0.5
    target_taxon_weight: float = 0.5
    domain_loss_weight: float = 5.0

    # True car le domaine cible contient des taxons absents de la source.
    # Mettre False uniquement si les etiquettes target/train ne doivent pas
    # participer a la loss taxonomique.
    use_target_labels: bool = False

    # GRL : lambda(p) = 2 / (1 + exp(-gamma*p)) - 1
    grl_gamma: float = 10.0

    # Focal Loss taxonomique
    focal_gamma: float = 0.5
    focal_alpha_mode: str = "auto"  # auto | none
    focal_ignore_index: int = -100
    focal_max_weight: float = 5.0

    # Scheduler et early stopping
    scheduler: str = "cosine"
    early_patience: int = int(os.environ.get("EARLY_PATIENCE", "49"))
    early_min_delta: float = 1e-5
    save_last: bool = True

    def __post_init__(self) -> None:
        if not Path(self.data_dir).is_dir():
            raise FileNotFoundError(f"data_dir introuvable : {self.data_dir}")
        os.makedirs(self.run_dir, exist_ok=True)
        if self.source_taxon_weight < 0 or self.target_taxon_weight < 0:
            raise ValueError("Les poids taxonomiques doivent etre positifs.")
        if self.domain_loss_weight < 0:
            raise ValueError("domain_loss_weight doit etre positif.")


# ============================================================
# DATASETS ET TRANSFORMATIONS
# ============================================================

class ImageProcessorTransform:
    """Transform picklable compatible avec DataLoader sous Windows."""

    def __init__(
        self,
        model_id: str,
        token: Optional[str],
        local_files_only: bool,
    ) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            token=token,
            local_files_only=local_files_only,
        )

    def __call__(self, image):
        return self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)


class RemappedImageFolder(ImageFolder):
    """ImageFolder remappant ses labels locaux vers une taxonomie globale."""

    def __init__(
        self,
        root: str,
        global_class_to_idx: Dict[str, int],
        transform=None,
    ) -> None:
        super().__init__(root=root, transform=transform)

        local_classes = list(self.classes)
        local_class_to_idx = dict(self.class_to_idx)

        local_to_global = {
            local_idx: global_class_to_idx[class_name]
            for class_name, local_idx in local_class_to_idx.items()
        }

        self.samples = [
            (image_path, local_to_global[local_target])
            for image_path, local_target in self.samples
        ]
        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]

        self.local_classes = local_classes
        self.local_class_to_idx = local_class_to_idx
        self.classes = [
            class_name
            for class_name, _ in sorted(
                global_class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]
        self.class_to_idx = dict(global_class_to_idx)


def validate_layout(root: Path) -> Dict[str, Path]:
    paths = {
        f"{domain}_{split}": root / domain / split
        for domain in ("source", "target")
        for split in ("train", "val", "test")
    }

    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Dossiers manquants dans le jeu de donnees :\n  "
            + "\n  ".join(missing)
        )
    return paths


def discover_global_taxonomy(
    paths: Dict[str, Path],
) -> Tuple[Dict[str, set], list, Dict[str, int]]:
    classes_by_split = {}

    for dataset_name, dataset_path in paths.items():
        temporary_dataset = ImageFolder(str(dataset_path))
        classes_by_split[dataset_name] = set(temporary_dataset.classes)

    global_classes = sorted(set().union(*classes_by_split.values()))
    global_class_to_idx = {
        class_name: class_index
        for class_index, class_name in enumerate(global_classes)
    }

    train_classes = (
        classes_by_split["source_train"]
        | classes_by_split["target_train"]
    )
    evaluation_only_classes = set(global_classes) - train_classes

    if evaluation_only_classes:
        raise RuntimeError(
            "Classes presentes uniquement en validation/test et absentes "
            "de source/train et target/train :\n  "
            + "\n  ".join(sorted(evaluation_only_classes))
        )

    return classes_by_split, global_classes, global_class_to_idx


def make_loader(
    dataset: RemappedImageFolder,
    cfg: Config,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )


# ============================================================
# MODELE DANN
# ============================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


class GradientReversalLayer(nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        return GradientReversalFunction.apply(features, coefficient)


class DinoV3DANN(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        domain_hidden_dim: int,
        dropout: float,
        token: Optional[str],
        local_files_only: bool,
    ) -> None:
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_id,
            token=token,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.backbone.config.hidden_size)

        self.taxon_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_size, domain_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden_dim, 2),
        )

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)

        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output

        return outputs.last_hidden_state[:, 0]

    def classify_taxon(self, features: torch.Tensor) -> torch.Tensor:
        return self.taxon_head(features)

    def classify_domain(
        self,
        features: torch.Tensor,
        grl_lambda: float,
    ) -> torch.Tensor:
        reversed_features = self.grl(features, grl_lambda)
        return self.domain_head(reversed_features)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixel_values)
        return self.classify_taxon(features)


# ============================================================
# LOSSES ET OUTILS
# ============================================================

def compute_class_weights(
    train_datasets: Sequence[RemappedImageFolder],
    num_classes: int,
    max_weight: float,
) -> torch.Tensor:
    all_targets = []
    for dataset in train_datasets:
        all_targets.extend(dataset.targets)

    counts = torch.bincount(
        torch.as_tensor(all_targets, dtype=torch.long),
        minlength=num_classes,
    ).float()

    missing_indices = torch.where(counts == 0)[0].tolist()
    if missing_indices:
        index_to_class = {
            class_index: class_name
            for class_name, class_index
            in train_datasets[0].class_to_idx.items()
        }
        missing_names = [index_to_class[index] for index in missing_indices]
        raise RuntimeError(
            "Classes sans exemple dans les datasets participant a la Focal Loss :\n  "
            + "\n  ".join(missing_names)
        )

    frequencies = counts / counts.sum()
    weights = 1.0 / torch.sqrt(frequencies)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=max_weight)
    weights = weights / weights.mean()
    return weights


def build_taxon_criterion(
    cfg: Config,
    train_datasets: Sequence[RemappedImageFolder],
    num_classes: int,
    device: torch.device,
):
    alpha = None

    if cfg.focal_alpha_mode == "auto":
        alpha = compute_class_weights(
            train_datasets,
            num_classes,
            cfg.focal_max_weight,
        ).to(device)
    elif cfg.focal_alpha_mode != "none":
        raise ValueError("focal_alpha_mode doit valoir 'auto' ou 'none'.")

    criterion = FocalLoss(
        alpha=alpha,
        gamma=cfg.focal_gamma,
        reduction="mean",
        ignore_index=cfg.focal_ignore_index,
    ).to(device)

    resolved_alpha = (
        alpha.detach().cpu().tolist()
        if isinstance(alpha, torch.Tensor)
        else None
    )
    return criterion, resolved_alpha


def dann_lambda(progress: float, gamma: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


def get_lrs(optimizer: optim.Optimizer) -> Dict[str, float]:
    return {
        f"group_{group_index}": float(group["lr"])
        for group_index, group in enumerate(optimizer.param_groups)
    }


# ============================================================
# ENTRAINEMENT ET EVALUATION
# ============================================================

def train_one_epoch(
    model: DinoV3DANN,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    taxon_criterion: nn.Module,
    domain_criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch_index: int,
) -> Dict[str, float]:
    model.train()

    steps = max(len(source_loader), len(target_loader))
    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)

    sums = {
        "total": 0.0,
        "source_taxon": 0.0,
        "target_taxon": 0.0,
        "domain": 0.0,
        "domain_acc": 0.0,
    }

    for step in range(steps):
        try:
            source_pixels, source_targets = next(source_iterator)
        except StopIteration:
            source_iterator = iter(source_loader)
            source_pixels, source_targets = next(source_iterator)

        try:
            target_pixels, target_targets = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_pixels, target_targets = next(target_iterator)

        source_pixels = source_pixels.to(device, non_blocking=True)
        source_targets = source_targets.to(device, non_blocking=True)
        target_pixels = target_pixels.to(device, non_blocking=True)
        target_targets = target_targets.to(device, non_blocking=True)

        global_step = epoch_index * steps + step
        total_steps = max(1, cfg.epochs * steps - 1)
        progress = global_step / total_steps
        grl_lambda = dann_lambda(progress, cfg.grl_gamma)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(cfg.use_amp and device.type == "cuda")):
            source_features = model.extract_features(source_pixels)
            target_features = model.extract_features(target_pixels)

            source_taxon_logits = model.classify_taxon(source_features)
            source_taxon_loss = taxon_criterion(
                source_taxon_logits,
                source_targets,
            )

            if cfg.use_target_labels:
                target_taxon_logits = model.classify_taxon(target_features)
                target_taxon_loss = taxon_criterion(
                    target_taxon_logits,
                    target_targets,
                )
            else:
                target_taxon_loss = source_taxon_loss.new_zeros(())

            source_domain_logits = model.classify_domain(
                source_features,
                grl_lambda,
            )
            target_domain_logits = model.classify_domain(
                target_features,
                grl_lambda,
            )

            source_domain_targets = torch.zeros(
                source_pixels.size(0),
                dtype=torch.long,
                device=device,
            )
            target_domain_targets = torch.ones(
                target_pixels.size(0),
                dtype=torch.long,
                device=device,
            )

            domain_logits = torch.cat(
                (source_domain_logits, target_domain_logits),
                dim=0,
            )
            domain_targets = torch.cat(
                (source_domain_targets, target_domain_targets),
                dim=0,
            )
            domain_loss = domain_criterion(domain_logits, domain_targets)

            total_loss = (
                cfg.source_taxon_weight * source_taxon_loss
                + cfg.target_taxon_weight * target_taxon_loss
                + cfg.domain_loss_weight * domain_loss
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        domain_accuracy = (
            domain_logits.argmax(dim=1) == domain_targets
        ).float().mean()

        sums["total"] += float(total_loss.item())
        sums["source_taxon"] += float(source_taxon_loss.item())
        sums["target_taxon"] += float(target_taxon_loss.item())
        sums["domain"] += float(domain_loss.item())
        sums["domain_acc"] += float(domain_accuracy.item())

    metrics = {
        metric_name: metric_sum / max(1, steps)
        for metric_name, metric_sum in sums.items()
    }
    metrics["grl_lambda"] = dann_lambda(
        (epoch_index + 1) / cfg.epochs,
        cfg.grl_gamma,
    )
    return metrics


@torch.no_grad()
def evaluate_taxon(
    model: DinoV3DANN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for pixel_values, targets in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(pixel_values)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size

    return (
        loss_sum / max(1, total),
        correct / max(1, total),
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    base_run_dir = Path(cfg.run_dir)
    run_dir = (
        base_run_dir / time.strftime("%Y%m%d-%H%M%S")
        if cfg.make_subrun_with_timestamp
        else base_run_dir
    )
    tensorboard_dir = run_dir / "tb"
    checkpoint_dir = run_dir / "checkpoints"
    ensure_dir(tensorboard_dir)
    ensure_dir(checkpoint_dir)

    environment = get_env_info()
    write_json(
        run_dir / "config.json",
        {"config": asdict(cfg), "env": environment},
    )

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text(
        "run/config",
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
    )

    paths = validate_layout(Path(cfg.data_dir))

    (
        classes_by_split,
        global_classes,
        global_class_to_idx,
    ) = discover_global_taxonomy(paths)

    num_classes = len(global_classes)
    common_train_classes = (
        classes_by_split["source_train"]
        & classes_by_split["target_train"]
    )
    source_only_classes = (
        classes_by_split["source_train"]
        - classes_by_split["target_train"]
    )
    target_only_classes = (
        classes_by_split["target_train"]
        - classes_by_split["source_train"]
    )

    print(f"[INFO] classes globales={num_classes}")
    print(
        f"[INFO] classes source_train="
        f"{len(classes_by_split['source_train'])}"
    )
    print(
        f"[INFO] classes target_train="
        f"{len(classes_by_split['target_train'])}"
    )
    print(f"[INFO] classes communes={len(common_train_classes)}")
    print(f"[INFO] classes source uniquement={len(source_only_classes)}")
    print(f"[INFO] classes cible uniquement={len(target_only_classes)}")

    write_json(run_dir / "class_to_idx.json", global_class_to_idx)
    write_json(
        run_dir / "classes_by_split.json",
        {
            dataset_name: sorted(class_names)
            for dataset_name, class_names in classes_by_split.items()
        },
    )

    transform = ImageProcessorTransform(
        cfg.model_id,
        cfg.token,
        cfg.local_files_only,
    )

    datasets = {
        dataset_name: RemappedImageFolder(
            root=str(dataset_path),
            global_class_to_idx=global_class_to_idx,
            transform=transform,
        )
        for dataset_name, dataset_path in paths.items()
    }

    loaders = {
        dataset_name: make_loader(
            dataset,
            cfg,
            device,
            shuffle=dataset_name.endswith("_train"),
        )
        for dataset_name, dataset in datasets.items()
    }

    model = DinoV3DANN(
        model_id=cfg.model_id,
        num_classes=num_classes,
        domain_hidden_dim=cfg.domain_hidden_dim,
        dropout=cfg.dropout,
        token=cfg.token,
        local_files_only=cfg.local_files_only,
    )

    freeze_all(model.backbone)
    unfreeze_last_n_blocks(
        model.backbone,
        cfg.unfreeze_last_n_blocks,
    )

    for module in (model.taxon_head, model.domain_head):
        for parameter in module.parameters():
            parameter.requires_grad = True

    model.to(device)

    for parameter_name, _ in model.backbone.named_parameters():
        block_index = infer_block_index(parameter_name)
        if block_index is not None:
            print(
                f"[INFO] exemple bloc backbone : "
                f"{parameter_name} -> {block_index}"
            )
            break

    print(
        "[INFO] parametres backbone entrainables=",
        sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ),
    )

    focal_weight_datasets = [datasets["source_train"]]
    if cfg.use_target_labels:
        focal_weight_datasets.append(datasets["target_train"])

    taxon_criterion, focal_alpha = build_taxon_criterion(
        cfg,
        focal_weight_datasets,
        num_classes,
        device,
    )
    domain_criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,
).to(device)

    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    taxon_parameters = [
        parameter
        for parameter in model.taxon_head.parameters()
        if parameter.requires_grad
    ]
    domain_parameters = [
        parameter
        for parameter in model.domain_head.parameters()
        if parameter.requires_grad
    ]

    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": cfg.lr_backbone}
        )
    parameter_groups.append(
        {"params": taxon_parameters, "lr": cfg.lr_head}
    )
    parameter_groups.append(
        {"params": domain_parameters, "lr": cfg.lr_domain}
    )

    optimizer = optim.AdamW(
        parameter_groups,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
        )
    elif cfg.scheduler != "none":
        raise ValueError("scheduler doit valoir 'cosine' ou 'none'.")

    best_target_val_loss = float("inf")
    best_target_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            source_loader=loaders["source_train"],
            target_loader=loaders["target_train"],
            optimizer=optimizer,
            scaler=scaler,
            taxon_criterion=taxon_criterion,
            domain_criterion=domain_criterion,
            device=device,
            cfg=cfg,
            epoch_index=epoch - 1,
        )

        source_val_loss, source_val_acc = evaluate_taxon(
            model,
            loaders["source_val"],
            taxon_criterion,
            device,
        )
        target_val_loss, target_val_acc = evaluate_taxon(
            model,
            loaders["target_val"],
            taxon_criterion,
            device,
        )
        learning_rates = get_lrs(optimizer)

        print(
            f"[E{epoch:03d}] "
            f"total={train_metrics['total']:.4f} "
            f"src_tax={train_metrics['source_taxon']:.4f} "
            f"tgt_tax={train_metrics['target_taxon']:.4f} "
            f"domain={train_metrics['domain']:.4f} "
            f"domain_acc={train_metrics['domain_acc']:.4f} "
            f"lambda={train_metrics['grl_lambda']:.4f} "
            f"src_val_loss={source_val_loss:.4f} "
            f"src_val_acc={source_val_acc:.4f} "
            f"tgt_val_loss={target_val_loss:.4f} "
            f"tgt_val_acc={target_val_acc:.4f}"
        )

        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(
                f"train/{metric_name}",
                metric_value,
                epoch,
            )
        writer.add_scalar("val/source_loss", source_val_loss, epoch)
        writer.add_scalar("val/source_acc", source_val_acc, epoch)
        writer.add_scalar("val/target_loss", target_val_loss, epoch)
        writer.add_scalar("val/target_acc", target_val_acc, epoch)

        for group_name, learning_rate in learning_rates.items():
            writer.add_scalar(
                f"lr/{group_name}",
                learning_rate,
                epoch,
            )

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "source_val_loss": source_val_loss,
            "source_val_acc": source_val_acc,
            "target_val_loss": target_val_loss,
            "target_val_acc": target_val_acc,
            "class_to_idx": global_class_to_idx,
            "classes": global_classes,
            "classes_by_split": {
                name: sorted(values)
                for name, values in classes_by_split.items()
            },
            "config": asdict(cfg),
            "env": environment,
            "model_id": cfg.model_id,
            "focal_alpha_resolved": focal_alpha,
        }

        if cfg.save_last:
            save_checkpoint(
                checkpoint_dir / "last.pt",
                checkpoint,
            )

        if target_val_loss < best_target_val_loss - cfg.early_min_delta:
            best_target_val_loss = target_val_loss
            best_target_val_acc = target_val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_dir / "best.pt",
                checkpoint,
            )
            print("[INFO] best.pt mis a jour sur la loss cible")
        else:
            epochs_without_improvement += 1
            print(
                f"[INFO] aucune amelioration "
                f"({epochs_without_improvement}/{cfg.early_patience})"
            )

        if scheduler is not None:
            scheduler.step()

        if epochs_without_improvement >= cfg.early_patience:
            print("[EARLY STOPPING]")
            break

    best_checkpoint_path = checkpoint_dir / "best.pt"
    if not best_checkpoint_path.exists():
        raise RuntimeError("Aucun checkpoint best.pt n'a ete cree.")

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["state_dict"])

    source_test_loss, source_test_acc = evaluate_taxon(
        model,
        loaders["source_test"],
        taxon_criterion,
        device,
    )
    target_test_loss, target_test_acc = evaluate_taxon(
        model,
        loaders["target_test"],
        taxon_criterion,
        device,
    )

    results = {
        "best_epoch": best_epoch,
        "best_target_val_loss": float(best_target_val_loss),
        "best_target_val_acc": float(best_target_val_acc),
        "source_test_loss": float(source_test_loss),
        "source_test_acc": float(source_test_acc),
        "target_test_loss": float(target_test_loss),
        "target_test_acc": float(target_test_acc),
        "num_global_classes": num_classes,
        "num_common_train_classes": len(common_train_classes),
        "num_source_only_classes": len(source_only_classes),
        "num_target_only_classes": len(target_only_classes),
    }

    write_json(run_dir / "results_dann.json", results)
    writer.add_text(
        "results",
        json.dumps(results, indent=2, ensure_ascii=False),
        0,
    )
    writer.close()

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[DONE] run_dir={run_dir}")
    print(
        f"[DONE] TensorBoard : tensorboard --logdir "
        f"{run_dir.parent.resolve()}"
    )


if __name__ == "__main__":
    main()
