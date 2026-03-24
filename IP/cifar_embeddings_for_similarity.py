import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import torch
from typing import List
from PIL import Image
import numpy as np
from torchvision import transforms
import timm


BRAIN_KEY = "dinov3_uniqueness"
# Nom de la run d’analyse FiftyOne Brain (identifiant interne)

VEC_FIELD = "emb_dinov3"
# Nom du champ dataset où seront stockés les embeddings

BATCH_SIZE = 256
# Nombre d’images traitées par batch (optimisation GPU)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Utilise GPU si disponible, sinon CPU

MODEL_PATH = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
# Chemin vers les poids du modèle DINOv3


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
# Ouvre une image et force la conversion en RGB

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)
# Normalise chaque vecteur embedding (norme L2 = 1)
# np.clip évite la division par zéro

# ===== PREPROCESS =====
preprocess = transforms.Compose([
    transforms.Resize(256),
    # Redimensionne l’image (plus grand côté = 256)

    transforms.CenterCrop(224),
    # Crop centré en 224×224 (taille attendue par ViT)

    transforms.ToTensor(),
    # Convertit PIL → Tensor PyTorch (0–1)

    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
    # Normalisation ImageNet (indispensable pour modèles pré-entraînés)
])


# ===== DATASET =====
dataset = foz.load_zoo_dataset("cifar10", split="test")
# Charge CIFAR-10 (split test) depuis FiftyOne Zoo

view = dataset
# Crée une vue (ici identique au dataset)

# ===== MODEL =====
model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0) 
# Recrée l'architecture du modele "Vision Transformer small"
# pretrained=False → on spécifie qu'on veut charger nos propres poids
# num_classes=0 → sortie = features, pas de classification

state_dict = torch.load(MODEL_PATH, map_location="cpu")
# Charge les poids depuis le fichier .pth

model.load_state_dict(state_dict, strict=False)
# Injecte les poids dans le modèle
# strict=False → tolère des différences mineures

model = model.to(DEVICE).eval()
# Envoie le modèle sur GPU/CPU
# eval() → désactive dropout/batchnorm training
print("[INFO] device:", DEVICE)


# ===== IDS + PATHS =====
sample_ids: List[str] = view.values("id")
# Liste des IDs des samples FiftyOne

filepaths: List[str] = view.values("filepath")
# Liste des chemins d’images


# ===== EMBEDDINGS =====
with torch.no_grad():
# Désactive le calcul des gradients (plus rapide, moins de mémoire)

    for start in range(0, len(sample_ids), BATCH_SIZE):
        # Boucle batch par batch

        end = min(start + BATCH_SIZE, len(sample_ids)) # Dernier batch potentiellement plus petit
        batch_ids = sample_ids[start:end]
        batch_paths = filepaths[start:end]
        # Sélection des IDs et paths du batch


        images = [preprocess(load_rgb(p)) for p in batch_paths] # Charge + preprocess chaque image
        x = torch.stack(images, dim=0).to(DEVICE) # Empile en batch tensor [B, C, H, W]

        feats = model(x)   # num_classes=0 → extraction des features

        if feats.ndim == 3:
            feats = feats[:, 0, :] # Si sortie = tokens ViT → on prend le token CLS

        feats = feats.float().cpu().numpy() # Tensor → float → CPU → NumPy
        feats = l2_normalize(feats) # Normalisation L2 (important pour similarité / uniqueness)

        vecs_list = [v.tolist() for v in feats] # Convertit NumPy → listes Python (stockage FiftyOne)
        batch_view = view.select(batch_ids) # Sélectionne les samples correspondant au batch

        for sample, vec in zip(batch_view, vecs_list):
            sample[VEC_FIELD] = vec
            sample.save()
        # Sauvegarde embedding dans chaque sample


        if start == 0 or (start // BATCH_SIZE) % 20 == 0:
            print(f"[INFO] encodé: {end}/{len(sample_ids)}")
        # Log périodique de progression

dataset.save()
print(f"[DONE] embeddings écrits dans '{VEC_FIELD}'")


# ===== UNIQUENESS =====
if BRAIN_KEY in dataset.list_brain_runs():
    # Vérifie si une analyse précédente existe

    dataset.delete_brain_run(BRAIN_KEY)
    # Supprime ancienne run pour éviter conflit

    dataset.save()

fob.compute_uniqueness(
    dataset,
    embeddings=VEC_FIELD,
)
# Calcule un score d’unicité basé sur les embeddings

print(f"[DONE] uniqueness calculé: {BRAIN_KEY}")

 # 5) Lancer l'app Fiftyone
session = fo.launch_app(dataset)
session.wait() # Bloque le script tant que l’app est ouverte

