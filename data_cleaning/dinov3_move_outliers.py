import os
import shutil
import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

import fiftyone as fo
from fiftyone import ViewField as F

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# =========================
# CONFIG A MODIFIER
# =========================
DATA_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/FIN-Benthic"

# Où déplacer les outliers (sera créé si absent)
OUTLIERS_BASE_DIR = os.path.join(DATA_DIR, "_OUTLIERS")  # ex: .../FIN-Benthic/_OUTLIERS

# Si True: copie au lieu de déplacer
COPY_INSTEAD_OF_MOVE = True

# Modèle
model_name = "dinov3"
if model_name == "dinov2":
    MODEL_ID = "facebook/dinov2-base"
elif model_name == "dinov3":
    MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
else:
    raise ValueError("Choisir entre 'dinov2' ou 'dinov3'")

EMB_FIELD = model_name + "_embedding"
DATASET_NAME = "clean_classif_" + model_name
BATCH_SIZE = 16

# Outliers
OUTLIER_TAG = "outlier"
LABEL_IN = "ground_truth"
LABEL_OUT = "clean_label"
K = 10


@torch.inference_mode()
def compute_embeddings(model, processor, filepaths, device):
    imgs = [Image.open(p).convert("RGB") for p in filepaths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    feats = outputs.last_hidden_state
    emb = feats.mean(dim=1)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy().astype(np.float32)


def detect_outliers_knn_per_class(
    dataset,
    emb_field,
    label_in="ground_truth",
    label_out="clean_label",
    outlier_tag="outlier",
    k=10,
    min_class_size=None,
):
    """
    Outliers par classe via kNN (cosine). Marque:
      - label_out = "-1" pour outliers sinon copie label_in
      - tag outlier_tag
    Retourne la liste des sample ids outliers.
    """
    if min_class_size is None:
        min_class_size = max(k + 2, 10)

    base_view = dataset.exists(emb_field)
    classes = base_view.distinct(f"{label_in}.label")
    print("Nb classes:", len(classes))

    def knn_percentile_for_class(n):
        if n < 50:
            return 99
        elif n < 200:
            return 97.5
        elif n < 400:
            return 95
        else:
            return 93

    all_outlier_ids = []

    for c in classes:
        v = base_view.match(F(f"{label_in}.label") == c)
        n = len(v)

        if n < min_class_size:
            print(f"[{c}] skip (n={n}) trop petit pour kNN (min={min_class_size})")
            continue

        ids = v.values("id")
        X = np.stack(v.values(emb_field)).astype(np.float32)
        X = normalize(X, norm="l2")

        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        knn_score = dists[:, 1:].mean(axis=1)

        knn_pctl = knn_percentile_for_class(n)
        thr = np.percentile(knn_score, knn_pctl)
        out_mask = knn_score > thr

        out_ids = [ids[i] for i, flag in enumerate(out_mask) if flag]
        all_outlier_ids.extend(out_ids)

        print(f"[{c}] n={n} | out_kNN={out_mask.sum()} (p{knn_pctl} thr={thr:.4f})")

    # Relabel dans nouveau champ
    base_ids = base_view.values("id")
    base_labels = base_view.values(f"{label_in}.label")
    clean_labels = np.array(base_labels, dtype=object)

    out_set = set(all_outlier_ids)
    mask_global = np.array([_id in out_set for _id in base_ids], dtype=bool)
    clean_labels[mask_global] = "-1"

    base_view.set_values(label_out, [fo.Classification(label=str(l)) for l in clean_labels])

    # Tag
    if all_outlier_ids:
        dataset.select(all_outlier_ids).tag_samples(outlier_tag)

    dataset.save()
    print(f"Total outliers (toutes classes): {len(all_outlier_ids)}")
    return all_outlier_ids


def safe_move_or_copy(src, dst, do_copy=False):
    """
    Déplace (ou copie) src -> dst.
    Si dst existe déjà, ajoute un suffixe _1, _2, ...
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(src):
        return False, f"Source introuvable: {src}"

    base, ext = os.path.splitext(dst)
    candidate = dst
    i = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{i}{ext}"
        i += 1

    if do_copy:
        shutil.copy2(src, candidate)
        return True, f"Copié vers {candidate}"
    else:
        shutil.move(src, candidate)
        return True, f"Déplacé vers {candidate}"


def move_outliers_to_folders(dataset, outlier_ids, data_dir, out_base_dir, do_copy=False):
    """
    Déplace les outliers dans out_base_dir en recréant le dossier source (ex: la classe).

    Exemple:
      DATA_DIR/.../FIN-Benthic/<classe>/img.jpg
      -> OUTLIERS_BASE_DIR/<classe>/img.jpg

    Si l'image ne vient pas de DATA_DIR, on met dans OUTLIERS_BASE_DIR/_UNKNOWN/
    """
    if not outlier_ids:
        print("Aucun outlier à déplacer.")
        return

    os.makedirs(out_base_dir, exist_ok=True)

    view = dataset.select(outlier_ids)
    filepaths = view.values("filepath")

    moved = 0
    errors = 0

    for fp in filepaths:
        fp = os.path.abspath(fp)
        data_dir_abs = os.path.abspath(data_dir)

        # Trouver le nom du dossier d'où vient l'image (dossier parent dans l'arborescence dataset)
        # On suppose: DATA_DIR/<folder_name>/<image>
        # folder_name = premier niveau sous DATA_DIR
        folder_name = "_UNKNOWN"
        rel = None
        try:
            rel = os.path.relpath(fp, data_dir_abs)
            # rel: "<folder_name>/.../img.jpg"
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                folder_name = parts[0]
        except Exception:
            folder_name = "_UNKNOWN"

        dst_dir = os.path.join(out_base_dir, folder_name)
        dst_path = os.path.join(dst_dir, os.path.basename(fp))

        ok, msg = safe_move_or_copy(fp, dst_path, do_copy=do_copy)
        if ok:
            moved += 1
        else:
            errors += 1
        print(msg)

    print(f"\nTerminé: {moved} outliers {'copiés' if do_copy else 'déplacés'} dans {out_base_dir} | erreurs={errors}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible. Installe un PyTorch CUDA et vérifie nvidia-smi.")

    device = torch.device("cuda:0")
    print("Using GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    # 1) Charger dataset "dossier par classe"
    if fo.dataset_exists(DATASET_NAME):
        dataset = fo.load_dataset(DATASET_NAME)
        print("Dataset chargé:", DATASET_NAME)
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=DATA_DIR,
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            name=DATASET_NAME,
        )
        print("Dataset importé:", DATASET_NAME)

    print(dataset)
    print("Classes:", dataset.distinct("ground_truth.label"))

    # 2) Charger modèle
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=True)
    model = AutoModel.from_pretrained(MODEL_ID, token=True).to(device).eval()

    # 3) Calcul embeddings + stockage (si pas déjà fait)
    if EMB_FIELD in dataset.get_field_schema():
        print(f"Champ {EMB_FIELD} existe déjà -> on ne recalcule pas")
    else:
        filepaths = dataset.values("filepath")
        n = len(filepaths)
        print("Nb images:", n)

        all_embs = []
        for i in range(0, n, BATCH_SIZE):
            batch_paths = filepaths[i:i + BATCH_SIZE]
            emb = compute_embeddings(model, processor, batch_paths, device)
            all_embs.extend([e.tolist() for e in emb])

            if i % (BATCH_SIZE * 20) == 0:
                print(f"Embeddings: {i}/{n}")

        dataset.set_values(EMB_FIELD, all_embs)
        dataset.save()
        print("Embeddings sauvegardés dans:", EMB_FIELD)

    # 4) Détection outliers
    outlier_ids = detect_outliers_knn_per_class(
        dataset,
        EMB_FIELD,
        label_in=LABEL_IN,
        label_out=LABEL_OUT,
        outlier_tag=OUTLIER_TAG,
        k=K,
    )

    # 5) Déplacement (ou copie) des outliers sans ouvrir FiftyOne
    move_outliers_to_folders(
        dataset,
        outlier_ids,
        data_dir=DATA_DIR,
        out_base_dir=OUTLIERS_BASE_DIR,
        do_copy=COPY_INSTEAD_OF_MOVE,
    )

    print("\n✅ Fini. (Aucune interface FiftyOne n'a été ouverte.)")


if __name__ == "__main__":
    main()