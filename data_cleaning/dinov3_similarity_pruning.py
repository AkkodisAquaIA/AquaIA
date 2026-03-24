"""
Réduction de redondance intra-classe avec FiftyOne + embeddings DINOv3

Version: traitement DOSSIER PAR DOSSIER
- Boucle sur les dossiers de DATA_DIR (dossiers classes)
- Pour chaque classe:
    - import dataset (uniquement ce dossier)
    - compute embeddings
    - compute similarity (brain_key unique)
    - dédup greedy + relabelisation
    - export images kept dans DEST_DIR/<classe>/

Prérequis:
- pip install fiftyone transformers torch scikit-learn pillow
"""

import os
import re
import shutil
import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# =========================
# CONFIG A MODIFIER
# =========================
DATA_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/FIN-Benthic2"
DEST_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/Combined_dataset"

model_name = "dinov3"
if model_name == "dinov2":
    MODEL_ID = "facebook/dinov2-base"
elif model_name == "dinov3":
    MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
else:
    raise ValueError("Choisir entre 'dinov2' ou 'dinov3'")

EMB_FIELD = model_name + "_embedding"   # ListField (legacy)
VEC_FIELD = "dinov3_vec"               # VectorField (recommandé)

BATCH_SIZE = 64

LABEL_IN = "ground_truth"
LABEL_OUT = "clean_label"
REDUNDANT_TAG = "redundant"
KEPT_TAG = "kept"

K = 30
MIN_CLASS_SIZE = 2

# Si tu veux garder les datasets temporaires (pour debug) mets True
KEEP_TEMP_DATASETS = False

def knn_percentile_for_class(n):
    if n < 50:
        return 99
    elif n < 200:
        return 97.5
    elif n < 400:
        return 95
    elif n < 700:
        return 92.5
    elif n < 1500:
        return 90
    else:
        return 85

def safe_key(s: str) -> str:
    """brain_key safe: pas d'espaces / caractères bizarres"""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s))[:80]


def list_class_folders(root_dir: str):
    """Retourne la liste des dossiers classes (1 niveau sous root)."""
    out = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            out.append((name, p))
    return out


@torch.inference_mode()
def compute_embeddings(model, processor, filepaths, device):
    imgs = [Image.open(p).convert("RGB") for p in filepaths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    feats = outputs.last_hidden_state          # (B, T, D)
    emb = feats.mean(dim=1)                    # (B, D)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy().astype(np.float32)


def ensure_vector_field(dataset):
    if VEC_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(
            VEC_FIELD,
            fo.VectorField,
            description="DINO embeddings (VectorField)"
        )
        dataset.save()

    # Optionnel: garder aussi EMB_FIELD (ListField)
    if EMB_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(EMB_FIELD, fo.ListField)
        dataset.save()


def compute_embeddings_for_dataset(dataset, model, processor, device):
    """
    Calcule les embeddings pour le dataset (une seule classe),
    uniquement pour les samples qui n'ont pas VEC_FIELD.
    """
    ensure_vector_field(dataset)

    base_view = dataset.exists("filepath")
    v_missing = base_view.match(~F(VEC_FIELD).exists())

    n_total = len(base_view)
    n_missing = len(v_missing)

    if n_total == 0:
        return

    if n_missing == 0:
        print(f"  embeddings déjà présents (n={n_total})")
        return

    print(f"  calcul embeddings: missing={n_missing}/{n_total}")
    filepaths = v_missing.values("filepath")

    vecs = []
    for i in range(0, len(filepaths), BATCH_SIZE):
        batch_paths = filepaths[i:i + BATCH_SIZE]
        emb = compute_embeddings(model, processor, batch_paths, device)
        vecs.extend([e.tolist() for e in emb])

        if i % (BATCH_SIZE * 20) == 0:
            print(f"    {i}/{len(filepaths)}")

    v_missing.set_values(VEC_FIELD, vecs)
    v_missing.set_values(EMB_FIELD, vecs)
    dataset.save()
    print(f"  OK embeddings écrits dans {VEC_FIELD}")


def greedy_dedup_from_knn(X, ids, sim_threshold=0.985, k=30):
    n = X.shape[0]

    if n == 0:
        return [], []
    if n == 1:
        return [ids[0]], []

    X = normalize(X.astype(np.float32), norm="l2")

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X)
    sims = 1.0 - dists

    adj = [set() for _ in range(n)]
    for i in range(n):
        for col in range(1, sims.shape[1]):
            j = int(nbrs[i, col])
            if j == i:
                continue
            if sims[i, col] >= sim_threshold:
                adj[i].add(j)
                adj[j].add(i)

    remaining = set(range(n))
    kept = []
    redundant = set()

    while remaining:
        best_i = None
        best_deg = -1
        for i in remaining:
            deg = sum((j in remaining) for j in adj[i])
            if deg > best_deg:
                best_deg = deg
                best_i = i

        i = best_i
        kept.append(i)

        neigh = [j for j in adj[i] if j in remaining]
        for j in neigh:
            redundant.add(j)

        remaining.remove(i)
        for j in neigh:
            remaining.discard(j)

    kept_ids = [ids[i] for i in kept]
    redundant_ids = [ids[i] for i in redundant if ids[i] not in set(kept_ids)]
    return kept_ids, redundant_ids


def deduplicate_dataset_single_class(dataset):
    """
    Dédup pour un dataset contenant une seule classe.
    - Tag redundant/kept
    - clean_label = label original ou "-1" pour redondants
    """
    base_view = dataset.exists(VEC_FIELD)
    n = len(base_view)

    # reset tags
    dataset.untag_samples(REDUNDANT_TAG)
    dataset.untag_samples(KEPT_TAG)
    dataset.save()

    if n < MIN_CLASS_SIZE:
        print(f"  skip dedup n={n}")
        # tout kept si 0/1 image
        base_view.set_values(LABEL_OUT, [fo.Classification(label="0")] * n)
        dataset.save()
        return base_view.values("id"), []

    ids = base_view.values("id")
    X = np.stack(base_view.values(VEC_FIELD)).astype(np.float32)

    sim_threshold = knn_percentile_for_class(n)
        
    kept_ids, redundant_ids = greedy_dedup_from_knn(
        X, ids, sim_threshold=sim_threshold, k=K
    )

    if redundant_ids:
        dataset.select(redundant_ids).tag_samples(REDUNDANT_TAG)
    if kept_ids:
        dataset.select(kept_ids).tag_samples(KEPT_TAG)

    # clean_label: "0" (kept) / "-1" (redundant)
    # (tu peux aussi mettre la vraie classe si tu préfères)
    clean_labels = np.array(["0"] * n, dtype=object)
    red_set = set(redundant_ids)
    mask_red = np.array([_id in red_set for _id in ids], dtype=bool)
    clean_labels[mask_red] = "-1"

    base_view.set_values(
        LABEL_OUT,
        [fo.Classification(label=str(l)) for l in clean_labels],
    )
    dataset.save()

    ratio = (len(redundant_ids) / n) * 100.0
    print(f"  dedup: n={n} | kept={len(kept_ids)} | redundant={len(redundant_ids)} ({ratio:.1f}%)")
    return kept_ids, redundant_ids


def compute_similarity_single_class(dataset, class_name: str):
    """
    Similarité FiftyOne pour ce dataset (une classe).
    """
    v = dataset.exists(VEC_FIELD)
    if len(v) < MIN_CLASS_SIZE:
        print(f"  similarity: skip n={len(v)}")
        return None

    brain_key = f"{model_name}_sim_{safe_key(class_name)}"
    if brain_key in dataset.list_brain_runs():
        dataset.delete_brain_run(brain_key)
        dataset.save()

    fob.compute_similarity(
        v,
        embeddings=VEC_FIELD,
        brain_key=brain_key,
    )
    print(f"  similarity index créé: {brain_key}")
    return brain_key


def export_kept_single_class(dataset, class_name: str, dest_root: str):
    """
    Export des images kept dans dest_root/<classe>/
    """
    class_dir = os.path.join(dest_root, str(class_name))
    os.makedirs(class_dir, exist_ok=True)

    kept_view = dataset.match(F(f"{LABEL_OUT}.label") != "-1")
    print(f"  export kept: {len(kept_view)} -> {class_dir}")

    for p in kept_view.values("filepath"):
        dst = os.path.join(class_dir, os.path.basename(p))
        shutil.copy2(p, dst)


def process_class_folder(
    class_name: str,
    class_path: str,
    model,
    processor,
    device,
    dest_root: str,
):
    """
    Pipeline complet pour UNE classe (un dossier):
    - import dataset
    - embeddings
    - similarity
    - dedup + relabel
    - export kept dans DEST_DIR/<classe>/
    """
    # Dataset temporaire unique (par classe)
    ds_name = f"dedup_{model_name}_{safe_key(class_name)}"
    if fo.dataset_exists(ds_name):
        dataset = fo.load_dataset(ds_name)
        print(f"[{class_name}] dataset chargé: {ds_name}")
    else:
        # Import simple: toutes les images du dossier, label = class_name
        dataset = fo.Dataset.from_images_dir(class_path, name=ds_name)
        dataset.add_sample_field(
            LABEL_IN,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification
        )
        dataset.set_values(LABEL_IN, [fo.Classification(label=class_name)] * len(dataset))
        dataset.save()
        print(f"[{class_name}] dataset importé: {ds_name} (n={len(dataset)})")

    if len(dataset) == 0:
        print(f"[{class_name}] dossier vide -> skip")
        if not KEEP_TEMP_DATASETS:
            dataset.delete()
        return

    # 1) Embeddings
    compute_embeddings_for_dataset(dataset, model, processor, device)

    # 2) Similarité
    compute_similarity_single_class(dataset, class_name)

    # 3) Dédup (tags + clean_label)
    deduplicate_dataset_single_class(dataset)

    # 4) Export kept
    export_kept_single_class(dataset, class_name, dest_root)

    # 5) (optionnel) cleanup dataset temporaire
    if not KEEP_TEMP_DATASETS:
        dataset.delete()
        print(f"[{class_name}] dataset supprimé (temp)")

    print(f"[{class_name}] DONE\n")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible. Installe un PyTorch CUDA et vérifie nvidia-smi.")

    device = torch.device("cuda:0")
    print("Using GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    os.makedirs(DEST_DIR, exist_ok=True)

    # Charger modèle (note: token=True si HF l'exige)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=True)
    model = AutoModel.from_pretrained(MODEL_ID, token=True).to(device).eval()

    class_folders = list_class_folders(DATA_DIR)
    print("Nb classes détectées:", len(class_folders))

    for class_name, class_path in class_folders:
        print("=" * 80)
        print(f"PROCESS CLASS: {class_name} | path={class_path}")
        process_class_folder(
            class_name=class_name,
            class_path=class_path,
            model=model,
            processor=processor,
            device=device,
            dest_root=DEST_DIR,
        )

    print("ALL CLASSES DONE")
    print("Export root:", DEST_DIR)


if __name__ == "__main__":
    main()
