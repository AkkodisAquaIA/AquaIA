import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

import fiftyone as fo
import fiftyone.brain as fob

from sklearn.neighbors import NearestNeighbors
import hdbscan


# =========================
# CONFIG A MODIFIER
# =========================
DATA_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN-Benthic/Cropped images"  # racine: class_a/, class_b/, ...

model_name = "dinov3" # "dinov3"

if model_name == "dinov2":
    MODEL_ID = "facebook/dinov2-base"
elif model_name == "dinov3":
    MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
else: 
    raise ValueError("Choisir entre le modèle 'dinov2' ou 'dinov3'")

EMB_FIELD = model_name + "_embedding"
DATASET_NAME = "clean_classif_" + model_name

BATCH_SIZE = 16


@torch.inference_mode()
def compute_embeddings(model, processor, filepaths, device):
    imgs = [Image.open(p).convert("RGB") for p in filepaths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    feats = outputs.last_hidden_state  # (B, T, D)
    emb = feats.mean(dim=1)            # (B, D)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.detach().cpu().numpy().astype(np.float32)


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device:", device)
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
    # processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    # model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        token=True,   #force l’usage du token local HF
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        token=True,
    )

    model = model.to(device)
    model.eval()

    # 3) Calcul embeddings (robuste) + stockage
    filepaths = dataset.values("filepath")
    n = len(filepaths)
    print("Nb images:", n)

    all_embs = []
    for i in range(0, n, BATCH_SIZE):
        batch_paths = filepaths[i:i + BATCH_SIZE]
        emb = compute_embeddings(model, processor, batch_paths, device)

        # Convertir en listes Python (plus sûr pour FiftyOne)
        all_embs.extend([e.tolist() for e in emb])

        if i % (BATCH_SIZE * 20) == 0:
            print(f"Embeddings: {i}/{n}")

    dataset.set_values(EMB_FIELD, all_embs)
    dataset.save()
    print("Embeddings sauvegardés dans:", EMB_FIELD)

    # Sécurité : uniquement samples avec embeddings
    view = dataset.match(fo.ViewField(EMB_FIELD) != None)
    print("Nb samples avec embeddings:", len(view))

    # 4a) UMAP
    fob.compute_visualization(
        view,
        embeddings=EMB_FIELD,
        method="umap",
        brain_key=model_name + "_umap",
    )
    print("UMAP créée: brain_key=" + model_name + "_umap")

    # --- Forcer l'écriture des coordonnées UMAP comme champs ---
    # Récupère les résultats du brain run
    viz = dataset.load_brain_results(model_name + "_umap")  # brain_key
    points = viz.points  # shape (N, 2)

    dataset.set_values(model_name + "_umap" +"_x", points[:, 0].tolist())
    dataset.set_values(model_name + "_umap" + "_y", points[:, 1].tolist())
    dataset.save()

    print("Champs UMAP écrits:" + model_name + "_umap_x / " + model_name + "_umap_y")

    if model_name + "_sim" in dataset.list_brain_runs():
        dataset.delete_brain_run( model_name + "_sim")
        dataset.save()
        print("Ancien brain run supprimé: " +  model_name + "_sim")

    # 4b) Similarity
    fob.compute_similarity(
        view,
        embeddings=EMB_FIELD,
        brain_key= model_name + "_sim",
    )
    print("Similarity index créé: brain_key=" +  model_name + "_sim")

    # 4c) Uniqueness
    fob.compute_uniqueness(
        view,
        embeddings=EMB_FIELD,
    )
    print("Uniqueness calculée")
    # Debug utile
    print("Brain runs:", dataset.list_brain_runs())

    print("Brain runs:", dataset.list_brain_runs())
    print("Has" + model_name + "_umap.x ?", model_name + "_umap.x" in dataset.get_field_schema())
    print("Has" + model_name + "_umap.y ?", model_name + "_umap.y" in dataset.get_field_schema())
    print("All fields:", list(dataset.get_field_schema().keys()))

    # 5) Lancer l'app
    session = fo.launch_app(dataset)
    print("\nDans l'app FiftyOne :")
    print("- Dans la colonne de gauche -> Others : tu dois voir " + model_name + "_umap.x et " + model_name + "_umap.y")
    print("- Utilise la vue scatter/embedding si elle apparaît, sinon explore via les champs UMAP")
    session.wait()


if __name__ == "__main__":
    main()
