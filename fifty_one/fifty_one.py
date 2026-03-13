import os
from pathlib import Path
import time
import torch
from PIL import Image
import numpy as np
# from torchvision import transforms
import timm
from concurrent.futures import ThreadPoolExecutor
# import faiss
from tqdm import tqdm
from yaspin import yaspin
import random
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time

import fiftyone as fo
from fiftyone import ViewField as F




from bboxes import bboxes as bb
from statistics_yolo import dataset_statistics_yolo as ds



from tools import utility as util
from tools import constants as ct
import tools.display_color as dc
from tools.constants import DISPLAY_COLORS as colors

#==========================================================================================
# ================= FONCTIONS =================

    
 # image_path = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Image1.png"
def splash_screen_circle(image_path, duration=2000, fade_steps=20):
    """
    Splash screen circulaire en couleur avec texte Aqua-IA.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)
    transparent_color = "magenta"
    splash.configure(bg=transparent_color)

    # --- Charger l'image et créer un cercle ---
    image_path = "Image1.png"    
    img = Image.open(image_path).convert("RGBA")
    size = min(img.width, img.height)
    img = img.resize((size, size))

    # Masque circulaire
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)


    img_circle = Image.new("RGBA", (size, size), (0,0,0,0))
    img_circle.paste(img, (0,0), mask=mask)

    # Ajouter le texte "Aqua-IA" centré
    draw_text = ImageDraw.Draw(img_circle)
    font_size = size // 8
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    text = "Aqua-IA"
    bbox = draw_text.textbbox((0,0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw_text.text(((size-w)/2, (size-h)/2), text, font=font, fill=(0,204,153,255))

    photo = ImageTk.PhotoImage(img_circle)

    label = tk.Label(splash, image=photo, bg=transparent_color, bd=0)
    label.pack()
    splash.wm_attributes("-transparentcolor", transparent_color)

    # Centrer la fenêtre
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - size) // 2
    y = (screen_height - size) // 2
    splash.geometry(f"{size}x{size}+{x}+{y}")



    splash.update()
    fade_in(splash)
    splash.after(duration, lambda: fade_out(splash, fade_steps))
    splash.mainloop()

def fade_in(window, steps=40, delay=50):
    alpha = 0.0
    window.attributes("-alpha", alpha)

    increment = 1.0 / steps

    def _fade():
        nonlocal alpha
        alpha += increment
        if alpha >= 1.0:
            window.attributes("-alpha", 1.0)
        else:
            window.attributes("-alpha", alpha)
            window.after(delay, _fade)

    _fade()

def fade_out(window, steps):
    alpha = window.attributes("-alpha") or 1.0
    decrement = alpha / steps
    def fade():
        nonlocal alpha
        alpha -= decrement
        if alpha <= 0:
            window.destroy()
        else:
            window.attributes("-alpha", alpha)
            window.after(50, fade)
    fade()



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

def maain():
    # ================= CONFIG =================

    VEC_FIELD = "emb_dinov3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DUP_THRESHOLD = 0.98
    NEIGHBORS = 20

    display = dc.DisplayColor()

    # ================= REPRO =================
    torch.manual_seed(ct.SEED)
    torch.cuda.manual_seed_all(ct.SEED)
    np.random.seed(ct.SEED)
    random.seed(ct.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    # # Display du logo et infos système
    print()
    splash_screen_circle("Image.png", duration=2500)

    display.print("Analyse des micro-invertébrés aquatiques", colors['aqua'])
    display.print("Version : 0.5.00 Beta", colors['aqua_dark'])
    print()

    display.print(f"Debug mode {'ON' if ct.DEBUG_MODE else 'OFF'}.", colors['warning'])

    display.print(f"Report mode {'ON' if ct.REPORT_MODE else 'OFF'}.", colors['warning'])

    prompt = (
            f"CUDA{'' if torch.cuda.is_available() else ' not'} available"
            f" - Running on {'GPU' if torch.cuda.is_available() else 'CPU'}.\n"
        )
    display.print(prompt, colors['warning'])



    DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco128"
    MODEL_PATH = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Fitty_One\Model\DINOv3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


    # DATASET_DIR = util.get_path_color("Entrée le chemin du dataset")
    # MODEL_PATH = util.get_path_color("Entrée le chemin du modèle DINOv3")

    print()
    display.print("Démarrage du traitement ...", colors['info'])

    dataset_name = "coco128_local"


    # Chargement des noms de classes pour les stats
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    class_names = ds.load_class_names(dataset_yaml)

    # ================= DATASET =================
    yaml_path = Path(DATASET_DIR) / "dataset.yaml"
    if not yaml_path.exists():
        display.print(f"dataset.yaml introuvable dans {DATASET_DIR}", colors['error'])
        exit(1)
        
    if dataset_name in fo.list_datasets():
        display.print(f"Suppression du dataset existant '{dataset_name}'", colors['info'])
        fo.delete_dataset(dataset_name)


    # validation des labels avant création du dataset FiftyOne
    erreur, all_bboxes, rapport, ctrl_ok = bb.validate_yolo_dataset_detailed(DATASET_DIR)

    print("===== RESUME VALIDATION =====")
    total_errors = sum(len(v) for v in erreur.values())
    print(f"Total types warning/erreurs : {len(erreur)}")
    print(f"Total warning/erreurs : {total_errors}")

    for k,v in erreur.items():
        print(f"{k:25} : {len(v)}")
    print()

    if not ctrl_ok:
        display.print("Erreurs détectées dans les labels. Arrêt du programme.", colors['error'])
        util.afficher_bbox_erreurs_compact(erreur)
        exit(1) 
    else:    
        display.print("Aucune erreur de label détectée. Création du dataset FiftyOne...\n", colors['ok'])


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
            util.format_and_display_error(f"création  ", rep="default_reports")
            exit(1)
    
    total_images = len(dataset)
    display.print(f"Dataset chargé avec succès : {total_images} images", colors['info'])

    # ================= STATISTICS =================
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    class_names = ds.load_class_names(dataset_yaml)

    results = ds.dataset_statistics_yolo(DATASET_DIR)
    ds.afficher_dataset_statistics(results, class_names, classes_par_ligne=3, afficher_hist=False)


    print()
    prompt = f"Script terminé.{ct.BELL}"
    display.print(prompt, colors['goodbye'])


#==========================================================================================
if __name__ == "__main__":
    maain()


# # ================= ENCODING =================
# encoding(dataset)


# # ================= FAISS OPTIMISE =================
# display.print("Détection doublons FAISS...", colors['info'])
# embeddings = np.array(dataset.values(VEC_FIELD), dtype="float32")
# num_embeddings, dim = embeddings.shape

# # Paramètres FAISS IVF
# nlist = int(np.sqrt(num_embeddings))
# nprobe = max(5, nlist // 20)

# quantizer = faiss.IndexFlatIP(dim)
# index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

# # GPU optionnel
# try:
#     if faiss.get_num_gpus() > 0:
#         display.print("Utilisation FAISS GPU", colors['info'])
#         res = faiss.StandardGpuResources()
#         index = faiss.index_cpu_to_gpu(res, 0, index)
# except Exception as e:
#     display.print(f"FAISS GPU indisponible → CPU : {e}", colors['warning'])

# # Train et ajout embeddings
# display.print("Training index FAISS...", colors['info'])
# index.train(embeddings)
# index.add(embeddings)
# index.nprobe = nprobe

# display.print(f"Recherche FAISS (nprobe={nprobe})...", colors['info'])
# D, I = index.search(embeddings, NEIGHBORS)

# # ================= CLUSTERING =================
# parent = np.arange(num_embeddings)

# def find(x):
#     while parent[x] != x:
#         parent[x] = parent[parent[x]]
#         x = parent[x]
#     return x

# def union(x, y):
#     parent[find(x)] = find(y)

# for i in range(num_embeddings):
#     for j in range(1, NEIGHBORS):
#         if I[i, j] <= i:
#             continue
#         if D[i, j] >= DUP_THRESHOLD:
#             union(i, I[i, j])

# clusters = {}
# sample_ids = dataset.values("id")
# for i in range(num_embeddings):
#     root = find(i)
#     clusters.setdefault(root, []).append(sample_ids[i])

# dup_ids = []
# for group in clusters.values():
#     if len(group) > 1:
#         dup_ids.extend(group)

# dataset.select(list(dup_ids)).tag_samples("dups")

# # --- Affichage et sauvegarde ---
# dup_paths = [p for p in dataset.select(list(dup_ids)).values("filepath")]
# util.display_and_save_errors(
#     dup_paths,
#     "images_doublons.txt",
#     f"Images doublons (seuil {DUP_THRESHOLD})"
# )

# # --- Vue combinée : doublons + bbox hors limites ---
# combined_view = dataset.match(
#     (F("tags").contains("dups")) # |
#    # (F("filepath").is_in(invalid_bbox_paths))
# )

# if len(combined_view) > 0:
#     display.print(f"Lancement de l'interface FiftyOne pour les doublons et bbox hors limites ({len(combined_view)} images)...", colors['info'])
#     util.launch_fiftyone_interface(combined_view)
# else:
#     display.print("Aucun doublon ni bbox hors limites à afficher.", colors['info'])


