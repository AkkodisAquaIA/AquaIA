import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# =========================
# CONFIG
# =========================
DATA_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/FIN-Benthic2")
DEST_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/Combined_dataset")

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
BATCH_SIZE = 64

K = 30                 # kNN
MIN_CLASS_SIZE = 30
MIN_THRESHOLD = 0.80   # garde-fous (évite seuil trop bas)
MAX_THRESHOLD = 0.995  # garde-fous (évite seuil trop haut)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def knn_percentile_for_class(n: int) -> float:
    # plus n est grand -> percentile plus bas -> seuil plus bas -> plus agressif -> on garde moins
    if n < 200:
        return 0.99
    elif n < 500:
        return 0.95
    elif n < 1000:
        return 0.90
    elif n < 2000:
        return 0.82
    else:
        return 0.85


def list_class_folders(root: Path):
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def list_images(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


@torch.inference_mode()
def embed_batch(model, processor, paths, device):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    feats = outputs.last_hidden_state          # (B, T, D)
    emb = feats.mean(dim=1)                    # (B, D)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy().astype(np.float32)


def compute_embeddings(model, processor, filepaths, device, batch_size=BATCH_SIZE):
    X = []
    for i in range(0, len(filepaths), batch_size):
        X.append(embed_batch(model, processor, filepaths[i:i + batch_size], device))
    return np.vstack(X) if X else np.zeros((0, 1), dtype=np.float32)


def compute_adaptive_threshold(X, k=K):
    """
    Calcule un seuil de similarité (cosine) dépendant de n.
    Méthode: kNN -> meilleure similarité par point -> percentile dépendant de n.
    """
    n = X.shape[0]
    if n < 2:
        return MAX_THRESHOLD

    X = normalize(X.astype(np.float32), norm="l2")

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    sims = 1.0 - dists  # (n, k+1) avec self en colonne 0 (sim ~ 1)

    # meilleure similarité avec un autre point (col 1..)
    best_sims = sims[:, 1:].max(axis=1)

    perc = knn_percentile_for_class(n)
    print('perc', perc)
    thr = float(np.percentile(best_sims, perc))
    print('thr', thr)

    # garde-fous
    thr = max(MIN_THRESHOLD, min(MAX_THRESHOLD, thr))
    return thr


def greedy_dedup_knn(X, sim_threshold, k=K):

    print('sim_threshold', sim_threshold)

    n = X.shape[0]
    if n == 0:
        return [], []
    if n == 1:
        return [0], []

    X = normalize(X.astype(np.float32), norm="l2")

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X)
    sims = 1.0 - dists

    adj = [set() for _ in range(n)]
    for i in range(n):
        for col in range(1, sims.shape[1]):
            j = int(nbrs[i, col])
            if j != i and sims[i, col] >= sim_threshold:
                adj[i].add(j)
                adj[j].add(i)

    remaining = set(range(n))
    kept = []
    redundant = set()

    while remaining:
        best_i = max(remaining, key=lambda i: sum((j in remaining) for j in adj[i]))
        kept.append(best_i)

        neigh = [j for j in adj[best_i] if j in remaining]
        redundant.update(neigh)

        remaining.remove(best_i)
        for j in neigh:
            remaining.discard(j)

    kept_set = set(kept)
    redundant = [i for i in redundant if i not in kept_set]
    return kept, redundant


def copy_kept(paths, kept_idx, dest_class_dir: Path):
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    for i in kept_idx:
        src = paths[i]
        dst = dest_class_dir / src.name
        if dst.exists():
            stem, suf = src.stem, src.suffix
            c = 1
            while True:
                cand = dest_class_dir / f"{stem}__{c}{suf}"
                if not cand.exists():
                    dst = cand
                    break
                c += 1
        shutil.copy2(src, dst)


def process_one_class(folder: Path, model, processor, device):
    class_name = folder.name
    images = list_images(folder)

    path = DEST_DIR / class_name
    print('class_name', class_name)

    if not path.is_dir():
        print("Le dossier n'existe pas")

        if len(images) == 0:
            print(f"[{class_name}] vide -> skip")
            return

        print(f"[{class_name}] n={len(images)} embeddings...")
        X = compute_embeddings(model, processor, images, device)

        if len(images) < MIN_CLASS_SIZE:
            kept_idx = list(range(len(images)))
            red_idx = []
            thr = None
        else:
            thr = compute_adaptive_threshold(X, k=K)
            kept_idx, red_idx = greedy_dedup_knn(X, sim_threshold=thr, k=K)

        if thr is None:
            print(f"[{class_name}] kept={len(kept_idx)} redundant={len(red_idx)}")
        else:
            print(f"[{class_name}] thr={thr:.4f} | kept={len(kept_idx)} redundant={len(red_idx)} ({100*len(red_idx)/len(images):.1f}%)")

        copy_kept(images, kept_idx, DEST_DIR / class_name)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible. Installe un PyTorch CUDA et vérifie nvidia-smi.")

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=True)
    model = AutoModel.from_pretrained(MODEL_ID, token=True).to(device).eval()

    class_folders = list_class_folders(DATA_DIR)
    print("Nb classes détectées:", len(class_folders))

    for folder in class_folders:
        print("=" * 80)
        process_one_class(folder, model, processor, device)

    print("DONE. Export:", str(DEST_DIR))


if __name__ == "__main__":
    main()