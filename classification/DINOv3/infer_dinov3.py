#!/usr/bin/env python3
# infer_folder.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from transformers import AutoImageProcessor

from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

from common_dinov3 import (
    DinoV3Classifier
)

# =========================
# CONFIG 
# =========================
RUN_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/dinov3/20260217-214540/")  # contient checkpoints/
CHECKPOINT_NAME = "best.pt"                             # "best.pt" ou "last.pt"

# dossier d'entrée arbitraire (peut s'appeler comme tu veux)
INPUT_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN Benthic2/IDA/dataset_clean_splited/test")

# Si True: on suppose INPUT_DIR/class_name/*.jpg => on calcule métriques
# Si False: on parcourt récursivement INPUT_DIR et on prédit sans labels
LABELED_BY_SUBFOLDER = True

BATCH_SIZE = 64
NUM_WORKERS = 4
USE_AMP = True

# Export
OUT_DIR_NAME = "inference_outputs"  # créé dans RUN_DIR/
WRITE_TENSORBOARD = True

EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

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

        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
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
def predict_labeled(model, loader, device, use_amp: bool):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    y_true, y_pred, paths = [], [], []

    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
            loss = ce(logits, y)

        preds = logits.argmax(dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        paths.extend(list(p))

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)

    loss_avg = total_loss / max(1, total)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    return loss_avg, acc, np.asarray(y_true), np.asarray(y_pred), paths


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
    im = ax.imshow(cm_norm, interpolation="nearest")  # pas de cmap imposée
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

    #ckpt = load_checkpoint(ckpt_path, device)
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

    # TensorBoard (optionnel)
    writer = None
    if WRITE_TENSORBOARD:
        tb_dir = out_root / "tb"
        writer = SummaryWriter(log_dir=str(tb_dir))

    # Build model
    model = DinoV3Classifier(model_id, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    if LABELED_BY_SUBFOLDER:
        ds = LabeledFolderDataset(INPUT_DIR, class_to_idx, processor)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

        loss, acc, y_true, y_pred, paths = predict_labeled(model, loader, device, USE_AMP)

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
            zero_division=0
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
        }
        (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_root / "classification_report.txt").write_text(report, encoding="utf-8")
        np.save(out_root / "confusion_matrix.npy", cm)

        fig = plot_confusion_matrix(cm, classes, "Confusion Matrix (normalized by true)")
        fig_path = out_root / "confusion_matrix.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        # CSV per-image
        rows = []
        for p, yt, yp in zip(paths, y_true.tolist(), y_pred.tolist()):
            rows.append([p, idx_to_class[int(yt)], idx_to_class[int(yp)]])
        write_csv(out_root / "predictions.csv", rows, header=["path", "true_class", "pred_class"])

        if writer is not None:
            writer.add_scalar("loss", loss, 0)
            writer.add_scalar("acc", acc, 0)
            writer.add_scalar("macro_f1", macro_f1, 0)
            writer.add_scalar("balanced_acc", bal_acc, 0)
            writer.add_text("classification_report", report, 0)
            writer.add_text("model/model_id", model_id, 0)
            writer.add_text("model/checkpoint", str(ckpt_path), 0)


            fig2 = plot_confusion_matrix(cm, classes, "Confusion Matrix (normalized by true)")
            writer.add_figure("confusion_matrix", fig2, 0)
            plt.close(fig2)

        print("[DONE] Labeled inference terminé.")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    else:
        ds = UnlabeledRecursiveDataset(INPUT_DIR, processor)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

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
        (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

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
