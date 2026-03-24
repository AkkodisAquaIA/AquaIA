#!/usr/bin/env python3
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# =========================
# CONFIG A MODIFIER
# =========================
DATA_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/FIN-Benthic2_cleaned2"

BLURRY_OUT_DIRNAME = "blurry_outliers"
MAD_OUT_DIRNAME = "mad_outliers"
SIMILAR_OUT_DIRNAME = "similar_samples"

BLUR_THRESHOLD = 5         # 5 pour FIN-Benthic et 50 pour FIN-Benthic2
K = 10                      # kNN pour MAD
ALPHA = 3.5                 # seuil MAD (plus grand = moins d'outliers)

# Redondance / similarité
SIM_THRESHOLD = 0.985       # plus haut = plus strict (détecte uniquement quasi-doublons)
SIM_K = 30                  # voisins explorés pour trouver des doublons
MIN_CLASS_SIZE = 30

BATCH_SIZE = 64

model_name = "dinov3"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m" if model_name == "dinov3" else "facebook/dinov2-base"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# -------------------------
# Étape 1 : flou
# -------------------------
def blur_score(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def safe_move(src: Path, dst: Path):
    """Déplace src -> dst en évitant collisions."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    base = dst.with_suffix("")
    ext = dst.suffix
    candidate = dst
    i = 1
    while candidate.exists():
        candidate = Path(f"{base}_{i}{ext}")
        i += 1

    shutil.move(str(src), str(candidate))
    return candidate


def list_class_folders(root: Path, excluded_names: set):
    return [p for p in root.iterdir() if p.is_dir() and p.name not in excluded_names]


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def move_blurry_images(data_dir: str, blur_threshold: float, out_dirname: str = "blurry_outliers"):
    root = Path(data_dir)
    out_root = root / out_dirname
    print('out_root', out_root)
    out_root.mkdir(exist_ok=True)

    excluded = {out_dirname, MAD_OUT_DIRNAME, SIMILAR_OUT_DIRNAME}
    class_folders = list_class_folders(root, excluded)

    for class_folder in class_folders:
        imgs = list_images(class_folder)
        moved = 0

        for img_path in imgs:
            s = blur_score(img_path)
            if s is None:
                continue
            if s < blur_threshold:
                dst = out_root / class_folder.name / img_path.name
                safe_move(img_path, dst)
                moved += 1

        print(f"[BLUR] {class_folder.name}: moved {moved}/{len(imgs)} (thr={blur_threshold})")

    print(f"\n[BLUR] Done. Out dir: {out_root}")


# -------------------------
# Embeddings
# -------------------------
@torch.inference_mode()
def compute_embeddings_batch(model, processor, filepaths, device):
    imgs = [Image.open(p).convert("RGB") for p in filepaths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    feats = outputs.last_hidden_state          # (B, T, D)
    emb = feats.mean(dim=1)                    # (B, D)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy().astype(np.float32)


def compute_embeddings_all(model, processor, filepaths, device, batch_size=64):
    vecs = []
    for i in range(0, len(filepaths), batch_size):
        batch = filepaths[i:i + batch_size]
        vecs.append(compute_embeddings_batch(model, processor, batch, device))
    return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)


# -------------------------
# Étape 2 : outliers embeddings (MAD sur score kNN)
# -------------------------
def knn_scores(X: np.ndarray, k: int):
    n = X.shape[0]
    X = normalize(X.astype(np.float32), norm="l2")

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    return dists[:, 1:].mean(axis=1)


def mad_threshold(scores: np.ndarray, alpha: float = 3.5):
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-12
    return float(med + alpha * mad)


def move_mad_outliers(data_dir: str, out_dirname: str = "mad_outliers",
                      k: int = 10, alpha: float = 3.5,
                      model=None, processor=None, device=None):
    root = Path(data_dir)
    out_root = root / out_dirname
    out_root.mkdir(exist_ok=True)

    excluded = {BLURRY_OUT_DIRNAME, out_dirname, SIMILAR_OUT_DIRNAME}
    class_folders = list_class_folders(root, excluded)

    for class_folder in class_folders:
        filepaths = list_images(class_folder)
        n = len(filepaths)
        print(f"[MAD] {class_folder.name}: n={n}")

        if n < max(k + 2, 10):
            print(f"[MAD] {class_folder.name}: skip (trop petit pour kNN fiable)")
            continue

        X = compute_embeddings_all(model, processor, filepaths, device, batch_size=BATCH_SIZE)
        scores = knn_scores(X, k=k)
        thr = mad_threshold(scores, alpha=alpha)

        out_mask = scores > thr
        outliers = [filepaths[i] for i in range(n) if out_mask[i]]

        moved = 0
        for src in outliers:
            dst = out_root / class_folder.name / src.name
            safe_move(src, dst)
            moved += 1

        print(f"[MAD] {class_folder.name}: moved {moved}/{n} (alpha={alpha}, thr={thr:.4f})")

    print(f"\n[MAD] Done. Out dir: {out_root}")


# -------------------------
# Étape 3 : redondance / similarité (dédup)
# -------------------------
def greedy_dedup_from_knn(X, filepaths, sim_threshold=0.985, k=30):
    """
    Construit un graphe d'images "quasi identiques" via KNN (cosine),
    puis garde un sous-ensemble avec un greedy (coverage).
    Retourne (kept_paths, redundant_paths)
    """
    n = X.shape[0]
    if n == 0:
        return [], []
    if n == 1:
        return [filepaths[0]], []

    X = normalize(X.astype(np.float32), norm="l2")

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X)
    sims = 1.0 - dists

    adj = [set() for _ in range(n)]
    for i in range(n):
        for col in range(1, sims.shape[1]):  # skip self
            j = int(nbrs[i, col])
            if j == i:
                continue
            if sims[i, col] >= sim_threshold:
                adj[i].add(j)
                adj[j].add(i)

    remaining = set(range(n))
    kept_idx = []
    redundant_idx = set()

    while remaining:
        best_i = None
        best_deg = -1
        for i in remaining:
            deg = sum((j in remaining) for j in adj[i])
            if deg > best_deg:
                best_deg = deg
                best_i = i

        i = best_i
        kept_idx.append(i)
        neigh = [j for j in adj[i] if j in remaining]
        redundant_idx.update(neigh)

        remaining.remove(i)
        for j in neigh:
            remaining.discard(j)

    kept_paths = [filepaths[i] for i in kept_idx]
    kept_set = set(kept_paths)
    redundant_paths = [filepaths[i] for i in redundant_idx if filepaths[i] not in kept_set]
    return kept_paths, redundant_paths


def sim_params_for_class(n: int) -> tuple[float, int]:
    """
    Retourne (SIM_THRESHOLD, SIM_K) adaptés à la taille de la classe.

    Version agressive :
    - threshold diminue fortement quand n augmente
    - k augmente fortement pour explorer plus de voisins
    - suppression beaucoup plus forte pour les classes géantes
    """

    if n < 30:
        return 0.993, 10
    elif n < 80:
        return 0.990, 15
    elif n < 200:
        return 0.985, 25
    elif n < 500:
        return 0.978, 35
    elif n < 2000:
        return 0.955, 350
    else:
        # classes géantes → nettoyage très agressif
        return 0.94, 400
    

def move_similar_samples(data_dir: str, out_dirname: str = "similar_samples",
                         model=None, processor=None, device=None):
    root = Path(data_dir)
    out_root = root / out_dirname
    out_root.mkdir(exist_ok=True)

    excluded = {BLURRY_OUT_DIRNAME, MAD_OUT_DIRNAME, out_dirname}
    class_folders = [p for p in root.iterdir() if p.is_dir() and p.name not in excluded]

    for class_folder in class_folders:
        filepaths = [p for p in class_folder.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        n = len(filepaths)
        print(f"[SIM] {class_folder.name}: n={n}")

        if n < 2000:
            print(f"[SIM] {class_folder.name}: skip (n<2)")
            continue

        X = compute_embeddings_all(model, processor, filepaths, device, batch_size=BATCH_SIZE)

        sim_thr, sim_k = sim_params_for_class(n)
        kept, redundant = greedy_dedup_from_knn(X, filepaths, sim_threshold=sim_thr, k=sim_k)

        moved = 0
        for src in redundant:
            dst = out_root / class_folder.name / src.name
            safe_move(src, dst)
            moved += 1

        print(f"[SIM] {class_folder.name}: moved {moved}/{n} | thr={sim_thr} k={sim_k}")

    print(f"\n[SIM] Done. Out dir: {out_root}")


def main():
    # 1) Flou
    move_blurry_images(DATA_DIR, BLUR_THRESHOLD, out_dirname=BLURRY_OUT_DIRNAME)

    # 2) Modèle (pour MAD + SIM)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible. Vérifie driver NVIDIA + PyTorch CUDA.")

    device = torch.device("cuda:0")
    print("Using GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    #  3) Outliers MAD
    move_mad_outliers(
        DATA_DIR,
        out_dirname=MAD_OUT_DIRNAME,
        k=K,
        alpha=ALPHA,
        model=model,
        processor=processor,
        device=device
    )

    # 4) Similar / redondants
    move_similar_samples(
        DATA_DIR,
        out_dirname=SIMILAR_OUT_DIRNAME,
        model=model,
        processor=processor,
        device=device
    )

    print("\n✅ Pipeline terminé.")
    root = Path(DATA_DIR)
    print("Blurry outliers:", str(root / BLURRY_OUT_DIRNAME))
    print("MAD outliers:", str(root / MAD_OUT_DIRNAME))
    print("Similar samples:", str(root / SIMILAR_OUT_DIRNAME))


if __name__ == "__main__":
    main()