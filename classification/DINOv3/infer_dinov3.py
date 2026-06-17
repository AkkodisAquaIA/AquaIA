#!/usr/bin/env python3
# infer_folder.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from transformers import AutoImageProcessor

from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

from common_dinov3 import DinoV3Classifier

# =========================
# CONFIG
# =========================

import os

RUN_DIR = Path(os.environ.get("RUN_DIR", "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/test_dinov3/unfreeze1/focal_05/20260602-115513"))

INPUT_DIR = Path(os.environ.get("DATA_DIR", "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026_splited/test"))

CHECKPOINT_NAME = "best.pt"

LABELED_BY_SUBFOLDER = True

BATCH_SIZE = 64
NUM_WORKERS = 4
USE_AMP = True

OUT_DIR_NAME = "inference_outputs"
WRITE_TENSORBOARD = True

EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


# =========================
# LOSS
# =========================
class FocalLoss(nn.Module):
    """
    Focal Loss pour classification multi-classes.

    Args:
        alpha (None, float, list, tuple, Tensor):
            - None: pas de pondération de classes
            - float/int: facteur global appliqué à toutes les classes
            - list/tuple/Tensor de shape (C,): poids par classe
        gamma (float): paramètre de focalisation (>= 0)
        reduction (str): 'mean' | 'sum' | 'none'
        ignore_index (int): label ignoré, comme dans CrossEntropyLoss
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", ignore_index=-100):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.detach().clone().float())
        else:
            raise TypeError("alpha must be None, float, int, list, tuple, or Tensor")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits doit être de shape (B, C), reçu {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets doit être de shape (B,), reçu {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError("Batch size de logits et targets incompatible")

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        valid_mask = targets != self.ignore_index

        if valid_mask.sum() == 0:
            return logits.new_zeros(())

        ce_loss_valid = ce_loss[valid_mask]
        targets_valid = targets[valid_mask]

        pt = torch.exp(-ce_loss_valid)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            if self.alpha.ndim == 0:
                alpha_t = self.alpha.to(logits.device)
            elif self.alpha.ndim == 1:
                if self.alpha.numel() != logits.size(1):
                    raise ValueError(f"alpha a {self.alpha.numel()} classes, mais logits a {logits.size(1)} classes")
                alpha_t = self.alpha.to(logits.device)[targets_valid]
            else:
                raise ValueError("alpha doit être scalaire ou 1D")

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce_loss_valid

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            out = logits.new_zeros(targets.shape, dtype=loss.dtype)
            out[valid_mask] = loss
            return out
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def compute_class_weights_from_class_to_idx_and_samples(
    class_to_idx: Dict[str, int],
    samples: List[Tuple[Path, int]],
    num_classes: int,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, y in samples:
        counts[y] += 1.0

    if torch.any(counts == 0):
        missing = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"Impossible de calculer alpha auto: certaines classes sont absentes du dataset évalué: {missing}")

    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


def build_eval_criterion(
    ckpt: Dict,
    device: torch.device,
) -> nn.Module:
    ckpt_config = ckpt["config"]
    loss_name = str(ckpt_config.get("loss_name", "ce")).lower()

    if loss_name == "ce":
        print("[INFO] Eval criterion = CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    if loss_name != "focal":
        raise ValueError(f"loss_name inconnu dans le checkpoint: {loss_name}")

    gamma = float(ckpt_config.get("focal_gamma", 2.0))
    ignore_index = int(ckpt_config.get("focal_ignore_index", -100))

    alpha = ckpt.get("focal_alpha_resolved", None)

    criterion = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction="mean",
        ignore_index=ignore_index,
    ).to(device)

    print(f"[INFO] Eval criterion = FocalLoss(gamma={gamma}, alpha={'None' if alpha is None else alpha}, ignore_index={ignore_index})")
    return criterion


# =========================
# DATASETS
# =========================
class LabeledFolderDataset(Dataset):
    """
    INPUT_DIR/
      classA/*.jpg
      classB/*.jpg
    On calcule des métriques. Les classes vides OK (ignorées).
    """

    def __init__(self, root: Path, class_to_idx: Dict[str, int], processor):
        self.samples: List[Tuple[Path, int]] = []
        self.processor = processor
        self.class_to_idx = class_to_idx

        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in EXTS:
                    self.samples.append((p, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {root} (selon les classes du modèle).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return x, y, str(path)


class UnlabeledRecursiveDataset(Dataset):
    """
    Parcourt INPUT_DIR récursivement, sans labels.
    """

    def __init__(self, root: Path, processor):
        self.paths: List[Path] = []
        self.processor = processor

        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                self.paths.append(p)

        if len(self.paths) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return x, str(path)


# =========================
# PREDICTION
# =========================
@torch.no_grad()
def predict_labeled(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true, y_pred, y_prob, paths = [], [], [], []

    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

        preds = logits.argmax(dim=1)
        conf = probs.max(dim=1).values

        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(conf.detach().cpu().tolist())
        paths.extend(list(p))

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)

    loss_avg = total_loss / max(1, total)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    return loss_avg, acc, np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob), paths


@torch.no_grad()
def predict_unlabeled(model, loader, device, use_amp: bool):
    model.eval()
    all_pred, all_prob, paths = [], [], []

    for x, p in loader:
        x = x.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values

        all_pred.extend(pred.detach().cpu().tolist())
        all_prob.extend(conf.detach().cpu().tolist())
        paths.extend(list(p))

    return np.asarray(all_pred), np.asarray(all_prob), paths


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> plt.Figure:
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1.0), where=row_sums != 0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    fig.tight_layout()
    return fig


def write_csv(path: Path, rows: List[List[str]], header: List[str]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ckpt_path = RUN_DIR / "checkpoints" / CHECKPOINT_NAME

    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint introuvable: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
    classes: List[str] = ckpt["classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model_id: str = ckpt.get("model_id") or ckpt["config"]["model_id"]
    dropout: float = float(ckpt["config"].get("dropout", 0.0))
    num_classes = len(classes)

    processor = AutoImageProcessor.from_pretrained(model_id)

    out_root = RUN_DIR / OUT_DIR_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    writer = None
    if WRITE_TENSORBOARD:
        tb_dir = out_root / "tb"
        writer = SummaryWriter(log_dir=str(tb_dir))

    model = DinoV3Classifier(model_id, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    if LABELED_BY_SUBFOLDER:
        ds = LabeledFolderDataset(INPUT_DIR, class_to_idx, processor)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        criterion = build_eval_criterion(
            ckpt=ckpt,
            device=device,
        )

        loss, acc, y_true, y_pred, y_conf, paths = predict_labeled(model, loader, criterion, device, USE_AMP)

        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        labels = np.arange(len(classes))
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=classes,
            digits=4,
            zero_division=0,
        )

        metrics = {
            "mode": "labeled_by_subfolder",
            "input_dir": str(INPUT_DIR),
            "n_images": int(len(ds)),
            "loss": float(loss),
            "acc": float(acc),
            "macro_f1": float(macro_f1),
            "balanced_acc": float(bal_acc),
            "model_id": model_id,
            "checkpoint": str(ckpt_path),
            "run_dir": str(RUN_DIR),
            "eval_loss_name": str(ckpt["config"].get("loss_name", "ce")),
            "eval_focal_gamma": ckpt["config"].get("focal_gamma", None),
        }
        (out_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_root / "classification_report.txt").write_text(report, encoding="utf-8")
        np.save(out_root / "confusion_matrix.npy", cm)

        fig = plot_confusion_matrix(cm, classes, "Confusion Matrix (normalized by true)")
        fig_path = out_root / "confusion_matrix.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        rows = []
        for p, yt, yp, conf in zip(paths, y_true.tolist(), y_pred.tolist(), y_conf.tolist()):
            rows.append(
                [
                    p,
                    idx_to_class[int(yt)],
                    idx_to_class[int(yp)],
                    f"{float(conf):.6f}",
                ]
            )
        write_csv(
            out_root / "predictions.csv",
            rows,
            header=["path", "true_class", "pred_class", "confidence"],
        )

        if writer is not None:
            writer.add_scalar("loss", loss, 0)
            writer.add_scalar("acc", acc, 0)
            writer.add_scalar("macro_f1", macro_f1, 0)
            writer.add_scalar("balanced_acc", bal_acc, 0)
            writer.add_text("classification_report", report, 0)
            writer.add_text("model/model_id", model_id, 0)
            writer.add_text("model/checkpoint", str(ckpt_path), 0)
            writer.add_text("eval/loss_name", str(ckpt["config"].get("loss_name", "ce")), 0)

            fig2 = plot_confusion_matrix(cm, classes, "Confusion Matrix (normalized by true)")
            writer.add_figure("confusion_matrix", fig2, 0)
            plt.close(fig2)

        print("[DONE] Labeled inference terminé.")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    else:
        ds = UnlabeledRecursiveDataset(INPUT_DIR, processor)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        pred_idx, conf, paths = predict_unlabeled(model, loader, device, USE_AMP)

        rows = []
        for p, pi, c in zip(paths, pred_idx.tolist(), conf.tolist()):
            rows.append([p, idx_to_class[int(pi)], f"{float(c):.6f}"])
        write_csv(out_root / "predictions.csv", rows, header=["path", "pred_class", "confidence"])

        metrics = {
            "mode": "unlabeled_recursive",
            "input_dir": str(INPUT_DIR),
            "n_images": int(len(ds)),
            "model_id": model_id,
            "checkpoint": str(ckpt_path),
            "run_dir": str(RUN_DIR),
        }
        (out_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if writer is not None:
            writer.add_text("info", json.dumps(metrics, indent=2, ensure_ascii=False), 0)
            writer.add_text("model/model_id", model_id, 0)
            writer.add_text("model/checkpoint", str(ckpt_path), 0)

        print("[DONE] Unlabeled inference terminé.")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if writer is not None:
        writer.close()
        print(f"[DONE] TensorBoard: tensorboard --logdir {out_root / 'tb'}")


if __name__ == "__main__":
    main()
