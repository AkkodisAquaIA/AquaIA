#!/usr/bin/env python3
# train_dinov3.py

from __future__ import annotations
import json, time, platform, random, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, Resize, CenterCrop
from transformers import AutoImageProcessor, AutoModel

from common_dinov3 import (
    ensure_dir, write_json, set_seed, get_env_info,
    DinoV3Classifier, freeze_all, unfreeze_last_n_blocks,
    save_checkpoint
)

# =========================
# CONFIG 
# =========================
@dataclass
class Config:
    run_dir: str = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/dinov3"   # ← AJOUT ICI
    make_subrun_with_timestamp: bool = True  # True -> .../YYYYMMDD-HHMMSS/
    data_dir: str = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN Benthic2/IDA/dataset_clean_splited"
    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"

    # =========================
# CONFIG 
# =========================
# Section configuration

@dataclass
class Config:
    # Déclare une dataclass de paramètres d’entraînement

    run_dir: str = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/dinov3"   # ← AJOUT ICI
    # Dossier racine où écrire les résultats (logs, checkpoints, config, etc.)

    make_subrun_with_timestamp: bool = True  # True -> .../YYYYMMDD-HHMMSS/
    # Si True: crée un sous-dossier daté pour chaque run

    data_dir: str = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN Benthic2/IDA/dataset_clean_splited"
    # Dossier dataset contenant train/ et val/

    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    # Identifiant Hugging Face du modèle DINOv3 à charger

    # max epochs (early stopping peut arrêter avant)
    epochs: int = 25
    # Nombre maximum d’époques d’entraînement

    batch_size: int = 32
    # Taille de batch

    lr_head: float = 1e-3
    # Learning rate pour la tête de classification

    lr_backbone: float = 3e-5
    # Learning rate pour le backbone (plus petit car fine-tuning sensible)

    weight_decay: float = 0.05
    # Régularisation L2 (AdamW)

    dropout: float = 0.0
    # Dropout appliqué avant la head (0.0 => aucun)

    unfreeze_last_n_blocks: int = 0   # 0=tête seule ; 1-2 fine-tuning léger
    # Nombre de blocs ViT à “dé-geler” (0 => backbone gelé, seule head entraînée)

    use_amp: bool = True
    # Active l’entraînement en mixed precision si GPU

    num_workers: int = 4
    # Nombre de workers DataLoader (chargement parallèle)

    seed: int = 42
    # Seed pour reproductibilité

    early_patience: int = 7
    # Early stopping: stop si pas d’amélioration pendant N époques

    early_min_delta: float = 1e-4
    # Amélioration minimale requise sur val_loss pour considérer un progrès

    # Scheduler: "none" | "cosine" | "plateau"
    scheduler: str = "cosine"
    # Type de scheduler de LR

    cosine_tmax_epochs: Optional[int] = None
    # Paramètre T_max du scheduler cosine (si None, on prend epochs)

    plateau_factor: float = 0.5
    # Si plateau: facteur de réduction du LR (LR <- LR * factor)

    plateau_patience: int = 2
    # Si plateau: patience du ReduceLROnPlateau

    plateau_min_lr: float = 1e-7
    # Si plateau: LR minimum

    save_last: bool = True
    # Sauvegarder le checkpoint last.pt à chaque époque


CFG = Config()
# Instancie la config avec les valeurs par défaut

# =========================
# UTILS
# =========================

def infer_block_index(name: str):
    # Tente d’extraire l’index de bloc (layer) depuis le nom d’un paramètre du modèle (string)
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
    # Renvoie un dict des learning rates actuels pour chaque param group de l’optimizer 
    return {f"group_{i}": float(g.get("lr", 0.0)) for i, g in enumerate(optimizer.param_groups)}


# =========================
# TRAIN / EVAL
# =========================
# Fonctions d'entraînement/évaluation

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool) -> float:
    # Entraîne le modèle pendant une époque et renvoie la loss moyenne

    model.train()
    # Met le modèle en mode entraînement (dropout actif, etc.)

    ce = nn.CrossEntropyLoss()
    # Déclare la fonction de loss classification multi-classes

    total_loss = 0.0
    # Accumulateur de loss pondérée par le nombre d’échantillons

    total = 0
    # Compteur d’échantillons

    for pixel_values, targets in loader:
        # Boucle sur les batches du DataLoader (pixel_values, label)

        pixel_values = pixel_values.to(device, non_blocking=True)
        # Envoie les images sur le device (GPU/CPU) ; non_blocking accélère avec pin_memory

        targets = targets.to(device, non_blocking=True)
        # Envoie les labels sur le device

        optimizer.zero_grad(set_to_none=True)
        # Remet à zéro les gradients (set_to_none=True est plus efficace)

        with autocast(enabled=use_amp):
            # Active le mixed precision si use_amp=True

            logits = model(pixel_values)
            # Forward: obtient les logits (scores par classe)

            loss = ce(logits, targets)
            # Calcule la cross-entropy

        scaler.scale(loss).backward()
        # Backprop avec scaling (évite underflow en fp16)

        scaler.step(optimizer)
        # Applique la mise à jour des poids via l’optimizer

        scaler.update()
        # Met à jour dynamiquement le facteur de scaling

        total_loss += float(loss.item()) * targets.size(0)
        # Ajoute la loss du batch pondérée par batch_size

        total += targets.size(0)
        # Ajoute le nombre d’échantillons du batch

    return total_loss / max(1, total)
    # Retourne la loss moyenne (évite division par zéro)



@torch.no_grad()
# Décorateur: désactive le suivi de gradients dans toute la fonction
def eval_loss_acc(model, loader, device) -> Tuple[float, float]:
    # Évalue le modèle (loss + accuracy) sur un DataLoader
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for pixel_values, targets in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(pixel_values)
        loss = ce(logits, targets)
        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += targets.size(0)
        total_loss += float(loss.item()) * targets.size(0)
    return total_loss / max(1, total), correct / max(1, total)
    # Renvoie (loss moyenne, accuracy)


def main():
    set_seed(CFG.seed) # Fixe toutes les seeds (python/torch/cuda) pour reproductibilité
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # --- run_dir
    base = Path(CFG.run_dir)
    run_dir = base / time.strftime("%Y%m%d-%H%M%S") if CFG.make_subrun_with_timestamp else base
    # Si option active: crée sous-dossier timestampé sinon utilise base

    tb_dir = run_dir / "tb" # Dossier TensorBoard
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(tb_dir)
    ensure_dir(ckpt_dir)
    print(f"[INFO] run_dir={run_dir}")

    # --- save config/env
    env = get_env_info()
    write_json(run_dir / "config.json", {"config": asdict(CFG), "env": env})

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/env", json.dumps(env, indent=2, ensure_ascii=False))
    writer.add_text("run/config", json.dumps(asdict(CFG), indent=2, ensure_ascii=False))

    # --- data
    root = Path(CFG.data_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(f"On attend {train_dir} et {val_dir}")

    processor = AutoImageProcessor.from_pretrained(CFG.model_id)
    # Charge le processor associé au modèle (resize/normalize/rescale)

    def to_pixel_values(pil_img):
        # Fonction de conversion PIL -> tensor format attendu par le modèle
        return processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)
        # Applique preprocessing HF puis récupère pixel_values et enlève la dimension batch


    transform = Compose([Lambda(to_pixel_values)]) # Transform torchvision: applique to_pixel_values à chaque image

    # print(processor)
    # print("size:", getattr(processor, "size", None))
    # print("crop_size:", getattr(processor, "crop_size", None))
    # print("do_resize:", getattr(processor, "do_resize", None))
    # print("do_center_crop:", getattr(processor, "do_center_crop", None))

    train_ds = ImageFolder(str(train_dir), transform=transform)
    # Crée le dataset train à partir de dossiers + applique transform
    val_ds = ImageFolder(str(val_dir), transform=transform)

    # num_classes auto
    num_classes = len(train_ds.classes)
    if len(val_ds.classes) != num_classes or train_ds.classes != val_ds.classes:
        raise RuntimeError("Train/Val n'ont pas exactement les mêmes classes (dossiers).")

    write_json(run_dir / "class_to_idx.json", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda")
    )
    # num_workers: chargement parallèle ; pin_memory: accélère transfert vers GPU
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda")
    )

    # --- model
    model = DinoV3Classifier(CFG.model_id, num_classes=num_classes, dropout=CFG.dropout)
    freeze_all(model.backbone) # Gèle tous les paramètres du backbone (requires_grad=False)
    unfreeze_last_n_blocks(model.backbone, CFG.unfreeze_last_n_blocks) # Dé-gèle les N derniers blocs si demandé (fine-tuning partiel)
    for p in model.head.parameters():  # Parcourt les paramètres de la tête
        p.requires_grad = True # S’assure que la head est entraînable
    model.to(device)

    # --- optimizer
    head_params = [p for p in model.head.parameters() if p.requires_grad] # Liste des paramètres de la head à optimiser
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad] # Liste des paramètres du backbone à optimiser
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": CFG.lr_backbone})
    param_groups.append({"params": head_params, "lr": CFG.lr_head})
    optimizer = optim.AdamW(param_groups, weight_decay=CFG.weight_decay) # Optimizer AdamW avec weight decay
    scaler = GradScaler(enabled=(CFG.use_amp and device.type == "cuda")) # GradScaler activé seulement si AMP et GPU

    # --- scheduler
    scheduler = None
    if CFG.scheduler == "cosine":
        tmax = CFG.cosine_tmax_epochs or CFG.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        print(f"[INFO] Scheduler=CosineAnnealingLR(T_max={tmax})")
    elif CFG.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=CFG.plateau_factor,
            patience=CFG.plateau_patience,
            min_lr=CFG.plateau_min_lr,
            verbose=True,
        )
        print("[INFO] Scheduler=ReduceLROnPlateau(mode=min)")
    else:
        print("[INFO] Scheduler=none")

    # --- early stopping
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, CFG.use_amp)
        val_loss, val_acc = eval_loss_acc(model, val_loader, device)
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

    results = {"best_val_loss": float(best_val_loss), "best_val_acc": float(best_val_acc), "best_epoch": int(best_epoch)}
    write_json(run_dir / "results_trainval.json", results)
    writer.add_text("results/trainval", json.dumps(results, indent=2), 0)
    writer.close()

    print("[DONE] Train/Val terminé.")
    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] TensorBoard: tensorboard --logdir {run_dir.parent.resolve()}")


if __name__ == "__main__":
    main()
