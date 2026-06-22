# common_dinov3.py
from __future__ import annotations

import json
import platform
import random
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
import os

from transformers import AutoModel


# ---------- IO / misc ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    # Fixe les seeds des générateurs aléatoires (Python, NumPy, PyTorch CPU/GPU) pour assurer la reproductibilité des résultats
    # mêmes poids initiaux, mêmes batches, mêmes augmentations, mêmes métriques
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_env_info() -> Dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ---------- Freeze / unfreeze ----------
def infer_block_index(name: str):
    patterns = [
        r"^layer\.(\d+)\.",
        r"^layers\.(\d+)\.",
        r"^block\.(\d+)\.",
        r"^blocks\.(\d+)\.",
        r"\.layer\.(\d+)\.",
        r"\.layers\.(\d+)\.",
        r"\.block\.(\d+)\.",
        r"\.blocks\.(\d+)\.",
    ]

    for p in patterns:
        m = re.search(p, name)
        if m:
            return int(m.group(1))

    return None


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_n_blocks(backbone: nn.Module, n: int) -> int:
    """
    Dégèle les params appartenant aux n derniers blocks (si détectables).
    Retourne le nombre de paramètres matchés.
    """
    if n <= 0:
        return 0

    block_ids: List[int] = []
    for name, _ in backbone.named_parameters():
        idx = infer_block_index(name)
        if idx is not None:
            block_ids.append(idx)

    if not block_ids:
        return 0

    max_idx = max(block_ids)
    min_idx = max(0, max_idx - (n - 1))

    cnt = 0
    for name, p in backbone.named_parameters():
        idx = infer_block_index(name)
        if idx is not None and idx >= min_idx:
            p.requires_grad = True
            cnt += 1
    return cnt


# ---------- Model ----------
class DinoV3Classifier(nn.Module):
    """
    Backbone DINOv3 (Transformers AutoModel) + tête linéaire.
    Le token Hugging Face est lu depuis la variable d'environnement HF_TOKEN.
    """

    def __init__(self, model_id: str, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.model_id = model_id

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError("La variable d'environnement HF_TOKEN n'est pas définie. Elle est nécessaire pour accéder au modèle Hugging Face.")

        self.backbone = AutoModel.from_pretrained(model_id, token=hf_token)

        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0, :]
        return self.head(self.dropout(feat))


# ---------- Checkpoint helpers ----------
def save_checkpoint(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)
