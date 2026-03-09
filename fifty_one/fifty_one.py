import fiftyone as fo
from fiftyone import ViewField as F
# from fiftyone import utils

import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import timm
from concurrent.futures import ThreadPoolExecutor
import faiss
from tqdm import tqdm
from yaspin import yaspin
import random

from bboxes.bboxes import detect_bbox_problemes_detail_tolere, afficher_bbox_erreurs_compact

from tools import utility as util
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors

#==========================================================================================
# ================= FONCTIONS =================

# --- Chargement image RGB avec gestion erreurs ---
def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        display.print(f"Impossible de charger l'image : {path}", colors['error'])
        return None

# --- Normalisation L2 ---
def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

# --- Labels orphelins YOLO ---
def orphelins_YOLO():
    images = set(Path(DATASET_DIR, "images").rglob("*.*"))
    labels = set(Path(DATASET_DIR, "labels").rglob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    orphan_labels = sorted(label_stems - image_stems)
    util.display_and_save_errors(
        orphan_labels,
        "labels_orphelins.txt",
        "Labels orphelins (txt sans image)"
    )
    print()

# --- CONTROLE IMAGES SANS LABEL AVANT CREATION DATASET ---
def controle_images_sans_label():
    images_dir = Path(DATASET_DIR) / "images"
    labels_dir = Path(DATASET_DIR) / "labels"

    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in ct.IMAGE_EXT]
    labels = list(labels_dir.rglob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    images_without_label = sorted(image_stems - label_stems)

    if images_without_label:
        util.display_and_save_errors(
            images_without_label,
            "images_sans_labels.txt",
            "Images sans labels"
        )
        prompt = f"Arrêt du programme. Corrige les erreurs avant de continuer.{ct.BELL}"
        display.print(prompt, colors['error'])
        exit(1)

    display.print("Toutes les images ont un fichier label.\n", colors['ok'])

# --- Encodage des images avec skip intelligent et batch save ---
def encoding(dataset, batch_size=ct.BATCH_SIZE):
    # Sélection des images à encoder
    missing_view = dataset.match(F(VEC_FIELD) == None)
    total_missing = len(missing_view)

    if total_missing == 0:
        display.print("Tous les embeddings existent déjà, skip encodage.", colors['warning'])
        return

    display.print(f"Encodage des {total_missing} images manquantes...", colors['info'])
    
    missing_ids = missing_view.values("id")
    missing_paths = missing_view.values("filepath")

    executor = ThreadPoolExecutor(max_workers=ct.NUM_WORKERS)

    # Barre de progression
    with tqdm(
        total=total_missing,
        desc="Images encodées",
        unit="img",
        position=0,
        leave=True,
        ncols=ct.TQDM_NCOLS,
        dynamic_ncols=False
    ) as pbar, tqdm(
        total=0,
        desc="",
        position=1,
        bar_format="{desc}",
        leave=False,
        ncols=ct.TQDM_NCOLS,
        dynamic_ncols=False
    ) as mbar:

        total_batches = (total_missing + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, total_missing, batch_size), 1):
            end = min(start + batch_size, total_missing)
            batch_paths = missing_paths[start:end]
            batch_ids = missing_ids[start:end]

            mbar.set_description(f"Batch {batch_idx}/{total_batches}")
            mbar.refresh()

            # --- Chargement images et filtrage celles corrompues ---
            loaded_images = list(executor.map(load_rgb, batch_paths))
            loaded_images = [img for img in loaded_images if img is not None]

            if len(loaded_images) == 0:
                pbar.update(len(batch_paths))  # avance barre même si toutes corrompues
                continue

            # --- Prétraitement ---
            images = [preprocess(img) for img in loaded_images]
            x = torch.stack(images).to(DEVICE)

            # --- Passage dans le modèle ---
            with torch.no_grad():
                if DEVICE == "cuda":
                    with torch.autocast(device_type="cuda"):
                        feats = model(x)
                else:
                    feats = model(x)

            if feats.ndim == 3:
                feats = feats[:, 0, :]

            feats = feats.float().cpu().numpy()
            feats = l2_normalize(feats)

            # --- Sauvegarde batch par batch ---
            batch_embeddings = {sid: vec.tolist() for sid, vec in zip(batch_ids, feats)}
            dataset.set_values(VEC_FIELD, batch_embeddings, key_field="id")
            dataset.save()

            pbar.update(len(batch_paths))

    executor.shutdown()
    display.print("Embeddings enregistrés.\n", colors['info'])



#==========================================================================================
# ================= CONFIG =================
DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco128"
dataset_name = "coco128_local"
VEC_FIELD = "emb_dinov3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Fitty_One\Model\DINOv3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DUP_THRESHOLD = 0.98
NEIGHBORS = 20

display = util.DisplayColor()

# ================= REPRO =================
torch.manual_seed(ct.SEED)
torch.cuda.manual_seed_all(ct.SEED)
np.random.seed(ct.SEED)
random.seed(ct.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(ct.logo)


display.print(f"Report mode {'ON' if ct.REPORT_MODE else 'OFF'}.", colors['warning'])

prompt = (
        f"CUDA{'' if torch.cuda.is_available() else ' not'} available"
        f" - Running on {'GPU' if torch.cuda.is_available() else 'CPU'}.\n"
    )
display.print(prompt, colors['warning'])


# ================= DATASET =================
yaml_path = Path(DATASET_DIR) / "dataset.yaml"
if not yaml_path.exists():
    display.print(f"dataset.yaml introuvable dans {DATASET_DIR}", colors['error'])
    exit(1)
    
if dataset_name in fo.list_datasets():
    display.print(f"Suppression du dataset existant '{dataset_name}'", colors['info'])
    fo.delete_dataset(dataset_name)

controle_images_sans_label()


display.print(f"Création du dataset FiftyOne à partir du dossier '{DATASET_DIR}'...", colors['info'])
with yaspin(text="Chargement en cours...", color="cyan") as spinner:
    try:
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            dataset_dir=str(DATASET_DIR),
            name=dataset_name,
            progress=False  
        )
        spinner.text = " "
        spinner.ok("Ok ") 
    except Exception as e:
        spinner.fail("Out ")
        raise e

total_images = len(dataset)
display.print(f"Dataset chargé avec succès : {total_images} images", colors['info'])

# ================= MODEL =================
display.print("Chargement du modèle DINOv3...\n", colors['info'])
model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# --- Labels orphelins YOLO ---
orphelins_YOLO()

# --- Détections bbox invalides ---
display.print("Détection des problèmes de bbox...", colors['info'])
bbox_erreurs = detect_bbox_problemes_detail_tolere(dataset, bbox_tol=1e-6)

if any(len(paths) > 0 for paths in bbox_erreurs.values()):
    afficher_bbox_erreurs_compact(bbox_erreurs, noms_par_ligne=ct.n_per_line)


# ================= ENCODING =================
encoding(dataset)

# ================= FAISS OPTIMISE =================
display.print("Détection doublons FAISS...", colors['info'])
embeddings = np.array(dataset.values(VEC_FIELD), dtype="float32")
num_embeddings, dim = embeddings.shape

# Paramètres FAISS IVF
nlist = int(np.sqrt(num_embeddings))
nprobe = max(5, nlist // 20)

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

# GPU optionnel
try:
    if faiss.get_num_gpus() > 0:
        display.print("Utilisation FAISS GPU", colors['info'])
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
except Exception as e:
    display.print(f"FAISS GPU indisponible → CPU : {e}", colors['warning'])

# Train et ajout embeddings
display.print("Training index FAISS...", colors['info'])
index.train(embeddings)
index.add(embeddings)
index.nprobe = nprobe

display.print(f"Recherche FAISS (nprobe={nprobe})...", colors['info'])
D, I = index.search(embeddings, NEIGHBORS)

# ================= CLUSTERING =================
parent = np.arange(num_embeddings)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    parent[find(x)] = find(y)

for i in range(num_embeddings):
    for j in range(1, NEIGHBORS):
        if I[i, j] <= i:
            continue
        if D[i, j] >= DUP_THRESHOLD:
            union(i, I[i, j])

clusters = {}
sample_ids = dataset.values("id")
for i in range(num_embeddings):
    root = find(i)
    clusters.setdefault(root, []).append(sample_ids[i])

dup_ids = []
for group in clusters.values():
    if len(group) > 1:
        dup_ids.extend(group)

dataset.select(list(dup_ids)).tag_samples("dups")

# --- Affichage et sauvegarde ---
dup_paths = [p for p in dataset.select(list(dup_ids)).values("filepath")]
util.display_and_save_errors(
    dup_paths,
    "images_doublons.txt",
    f"Images doublons (seuil {DUP_THRESHOLD})"
)

# --- Vue combinée : doublons + bbox hors limites ---
combined_view = dataset.match(
    (F("tags").contains("dups")) # |
   # (F("filepath").is_in(invalid_bbox_paths))
)

if len(combined_view) > 0:
    display.print(f"Lancement de l'interface FiftyOne pour les doublons et bbox hors limites ({len(combined_view)} images)...", colors['info'])
    util.launch_fiftyone_interface(combined_view)
else:
    display.print("Aucun doublon ni bbox hors limites à afficher.", colors['info'])

print()
prompt = f"Script terminé.{ct.BELL}"
display.print(prompt, colors['goodbye'])
