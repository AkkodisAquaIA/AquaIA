import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import hdbscan


# =========================
# CONFIG A MODIFIER
# =========================
DATA_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN-Benthic/Cropped images"

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
LABEL_OUT = "clean_label"  # nouveau champ recommandé (ne pas écraser ground_truth au début)

K = 10  # kNN local
KNN_PCTL = 95  # top 5% isolés (par classe)
MIN_CLUSTER_SIZE = 20
PCA_DIMS = 64  # PCA avant HDBSCAN (stabilité)


@torch.inference_mode()
def compute_embeddings(model, processor, filepaths, device):
    imgs = [Image.open(p).convert("RGB") for p in filepaths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    feats = outputs.last_hidden_state  # (B, T, D)
    emb = feats.mean(dim=1)  # (B, D)
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
    Détecte les outliers par classe UNIQUEMENT via kNN, sur la base des embeddings.

    - Pour chaque classe:
        * calcule un score = distance cosine moyenne aux k plus proches voisins
        * marque outlier si score > percentile(knn_pctl)
    - Écrit un nouveau champ label_out:
        * label_out = "-1" pour outliers
        * sinon copie label_in
    - Tag les outliers avec outlier_tag

    Args:
        dataset: fiftyone.Dataset
        emb_field: str, champ embeddings (ex: "dinov3_embedding")
        label_in: str, champ label source (défaut "ground_truth")
        label_out: str, champ label destination (défaut "clean_label")
        outlier_tag: str, tag à ajouter aux outliers (défaut "outlier")
        k: int, nombre de voisins (défaut 10)
        min_class_size: int|None, taille min par classe pour appliquer kNN.
                        Si None: max(k+2, 10)

    Returns:
        List[str]: liste des sample ids marqués outliers
    """

    # Taille minimale par classe
    if min_class_size is None:
        min_class_size = max(k + 2, 10)

    # On travaille uniquement sur samples avec embeddings
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
        X = normalize(X, norm="l2")  # essentiel pour cosine

        # kNN score
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)

        # Ignore self-distance (colonne 0)
        knn_score = dists[:, 1:].mean(axis=1)

        knn_pctl = knn_percentile_for_class(n)
        # Threshold
        thr = np.percentile(knn_score, knn_pctl)
        out_mask = knn_score > thr

        out_ids = [ids[i] for i, flag in enumerate(out_mask) if flag]
        all_outlier_ids.extend(out_ids)

        print(f"[{c}] n={n} | out_kNN={out_mask.sum()} (p{knn_pctl} thr={thr:.4f})")

    # --- Relabelisation dans un nouveau champ ---
    base_ids = base_view.values("id")
    base_labels = base_view.values(f"{label_in}.label")
    clean_labels = np.array(base_labels, dtype=object)

    out_set = set(all_outlier_ids)
    mask_global = np.array([_id in out_set for _id in base_ids], dtype=bool)
    clean_labels[mask_global] = "-1"

    base_view.set_values(label_out, [fo.Classification(label=str(lbl)) for lbl in clean_labels])

    # Tag outliers
    if all_outlier_ids:
        dataset.select(all_outlier_ids).tag_samples(outlier_tag)

    dataset.save()
    print(f"Total outliers (toutes classes): {len(all_outlier_ids)}")
    return all_outlier_ids


def detect_outliers_knn_hdbscan_per_class(dataset, emb_field):
    """
    Détecte les outliers par classe et:
      - écrit clean_label = "-1" pour outliers, sinon copie label original
      - tag 'outlier'
    """
    # On travaille uniquement sur samples avec embedding
    base_view = dataset.exists(emb_field)

    classes = base_view.distinct(f"{LABEL_IN}.label")
    print("Nb classes:", len(classes))

    all_outlier_ids = []

    for c in classes:
        v = base_view.match(F(f"{LABEL_IN}.label") == c)
        n = len(v)
        if n < max(MIN_CLUSTER_SIZE, K + 2):
            # Trop peu d'images: skip (ou traite différemment)
            print(f"[{c}] skip (n={n}) trop petit pour kNN/HDBSCAN")
            continue

        ids = v.values("id")
        X = np.stack(v.values(emb_field)).astype(np.float32)
        X = normalize(X, norm="l2")  # cosine-friendly

        # --- kNN score ---
        nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        knn_score = dists[:, 1:].mean(axis=1)
        knn_thr = np.percentile(knn_score, KNN_PCTL)
        out_knn = knn_score > knn_thr

        # --- HDBSCAN (avec PCA pour stabilité) ---
        pca_dims = min(PCA_DIMS, X.shape[1], n - 1)
        Xp = PCA(n_components=pca_dims, random_state=0).fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(Xp)
        out_hdb = labels == -1

        # Fusion
        outlier_mask = out_knn | out_hdb
        out_ids = [ids[i] for i, flag in enumerate(outlier_mask) if flag]
        all_outlier_ids.extend(out_ids)

        print(f"[{c}] n={n} | out_kNN={out_knn.sum()} | out_HDB={out_hdb.sum()} | out_final={len(out_ids)}")

    # --- Relabelisation dans un nouveau champ ---
    # Copie des labels originaux vers LABEL_OUT, puis overwrite outliers en "-1"
    base_ids = base_view.values("id")
    base_labels = base_view.values(f"{LABEL_IN}.label")
    clean_labels = np.array(base_labels, dtype=object)

    out_set = set(all_outlier_ids)
    mask_global = np.array([_id in out_set for _id in base_ids], dtype=bool)
    clean_labels[mask_global] = "-1"

    base_view.set_values(LABEL_OUT, [fo.Classification(label=str(lbl)) for lbl in clean_labels])

    # Tag outliers
    if all_outlier_ids:
        dataset.select(all_outlier_ids).tag_samples(OUTLIER_TAG)

    dataset.save()
    print(f"Total outliers (toutes classes): {len(all_outlier_ids)}")
    return all_outlier_ids


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
    #    (si tu veux recalculer à chaque run, enlève la condition)
    if EMB_FIELD in dataset.get_field_schema():
        print(f"Champ {EMB_FIELD} existe déjà -> on ne recalcule pas")
    else:
        filepaths = dataset.values("filepath")
        n = len(filepaths)
        print("Nb images:", n)

        all_embs = []
        for i in range(0, n, BATCH_SIZE):
            batch_paths = filepaths[i : i + BATCH_SIZE]
            emb = compute_embeddings(model, processor, batch_paths, device)
            all_embs.extend([e.tolist() for e in emb])

            if i % (BATCH_SIZE * 20) == 0:
                print(f"Embeddings: {i}/{n}")

        dataset.set_values(EMB_FIELD, all_embs)
        dataset.save()
        print("Embeddings sauvegardés dans:", EMB_FIELD)

    # 4) Détection outliers + relabelisation (-1) par classe
    # detect_outliers_knn_hdbscan_per_class(dataset, EMB_FIELD)
    detect_outliers_knn_per_class(dataset, EMB_FIELD)

    # Vue "propres" (facultatif) : exclure outliers
    clean_view = dataset.match(F(f"{LABEL_OUT}.label") != "-1")
    print("Nb clean samples:", len(clean_view))

    # 5) UMAP (sur clean_view de préférence)
    fob.compute_visualization(
        clean_view,
        embeddings=EMB_FIELD,
        method="umap",
        brain_key=model_name + "_umap",
    )
    print("UMAP créée:", model_name + "_umap")

    viz = dataset.load_brain_results(model_name + "_umap")
    points = viz.points  # correspond à la view utilisée

    # Écrit x/y sur la view (pas tout le dataset)
    clean_view.set_values(model_name + "_umap_x", points[:, 0].tolist())
    clean_view.set_values(model_name + "_umap_y", points[:, 1].tolist())
    dataset.save()
    print("Champs UMAP écrits sur clean_view:", model_name + "_umap_x / " + model_name + "_umap_y")

    # Similarity (sur clean_view)
    if model_name + "_sim" in dataset.list_brain_runs():
        dataset.delete_brain_run(model_name + "_sim")
        dataset.save()

    fob.compute_similarity(
        clean_view,
        embeddings=EMB_FIELD,
        brain_key=model_name + "_sim",
    )
    print("Similarity index créé:", model_name + "_sim")

    # Uniqueness (sur clean_view)
    fob.compute_uniqueness(clean_view, embeddings=EMB_FIELD)
    print("Uniqueness calculée")

    # 6) App
    session = fo.launch_app(dataset)
    print("\nDans FiftyOne : filtre sur tag =", OUTLIER_TAG, "ou sur", LABEL_OUT + ".label == '-1'")
    session.wait()


if __name__ == "__main__":
    main()
