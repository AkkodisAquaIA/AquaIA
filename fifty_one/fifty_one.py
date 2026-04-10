import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import faiss
from tqdm import tqdm
import timm
import random
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

import fiftyone as fo
from fiftyone import ViewField as F

from bboxes import bboxes as bb
from statistics_yolo import dataset_statistics_yolo as ds

from tools import utility as util
from tools import constants as ct
import tools.display_color as dc
from tools.constants import DISPLAY_COLORS as colors
from graphe import graphe as gr


#==========================================================================================
# ================= FONCTIONS =================
 
    
 # image_path = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Image1.png"

def splash_screen_circle(image_path, duration=3000):
    """
    Splash screen circulaire avec un cercle extérieur clair
    et texte centré foncé.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)

    splash.attributes("-alpha", 0.0)

    transparent_color = "magenta"
    splash.configure(bg=transparent_color)

    # --- Charger l'image ---
    image_path = "Image1.png" 
    img = Image.open(image_path).convert("RGBA")
    size = min(img.width, img.height)
    img = img.resize((size, size))

    # --- Créer image finale ---
    img_circle = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img_circle)

    # Couleurs
    circle_color = (64, 224, 208, 255)  # Turquoise clair
    text_color = (0, 102, 102, 255)     # Bleu-vert foncé

    # Dessiner le cercle extérieur
    border_width = size // 20
    draw.ellipse((0, 0, size, size), fill=circle_color)

    # Masque circulaire pour l'image
    inner_size = size - 2*border_width
    img_resized = img.resize((inner_size, inner_size))
    mask_inner = Image.new("L", (inner_size, inner_size), 0)
    draw_mask_inner = ImageDraw.Draw(mask_inner)
    draw_mask_inner.ellipse((0,0,inner_size, inner_size), fill=255)

    # Coller l'image centrée
    img_circle.paste(img_resized, (border_width, border_width), mask_inner)

    # Ajouter le texte centré
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
    draw_text.text(((size-w)/2, (size-h)/2), text, font=font, fill=text_color)

    # Conversion en image Tkinter
    photo = ImageTk.PhotoImage(img_circle)

    # Affichage
    label = tk.Label(splash, image=photo, bg=transparent_color, bd=0)
    label.pack()
    splash.wm_attributes("-transparentcolor", transparent_color)

    # Centrer la fenêtre
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - size) // 2
    y = (screen_height - size) // 2
    splash.geometry(f"{size}x{size}+{x}+{y}")

    splash.after(10, lambda: fade_in(splash, steps=60, delay=30))
    
    # Afficher le splash screen pour la durée
    splash.after(duration, lambda: fade_out(splash, steps=40))
    splash.mainloop()

def fade_in(window, steps=60, delay=30):
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


def statistique(DATASET_DIR, path_user):
     # ================= STATISTICS =================
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    class_names = ds.load_class_names(dataset_yaml)

    results = ds.dataset_statistics_yolo(DATASET_DIR)
    seuils = util.calibrer_seuils_overflow(results,
                                            warning_percentile=ct.PERCENTILE_WARNING, 
                                            error_percentile=ct.PERCENTILE_ERROR)
    
    BBOX_OVERFLOW_WARNING = seuils['BBOX_OVERFLOW_WARNING']
    BBOX_OVERFLOW_ERROR   = seuils['BBOX_OVERFLOW_ERROR']

    outside_ratios = [a['outside_ratio_pct'] for a in results.get('anomalies',
                                            []) if 'outside_ratio_pct' in a]

    
    gr.bbox_overflow(outside_ratios, BBOX_OVERFLOW_WARNING, BBOX_OVERFLOW_ERROR ) 

    ds.afficher_dataset_statistics(results, path_user, class_names, classes_par_ligne=4, afficher_hist=True)


def create_dataset(DATASET_DIR):
    display = dc.DisplayColor()

    dataset_name = "coco128_local"

    display.print(f"Création du dataset FiftyOne à partir du dossier :\n   '{DATASET_DIR}'...", colors['info'])

    if dataset_name in fo.list_datasets():
        display.print(f"Suppression du dataset existant '{dataset_name}'", colors['info'])
        fo.delete_dataset(dataset_name)    
    
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        dataset_dir=str(DATASET_DIR),
        name=dataset_name
    )

    total_images = len(dataset)
    display.print(f"Dataset chargé avec succès : {total_images} images", colors['info'])

    return dataset

def load_model(MODEL_PATH, total_images, DEVICE):

    display = dc.DisplayColor()
    # ================= MODEL =================
    display.print("Chargement du modèle DINOv3...", colors['info'])
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()
    display.print(f"Vérification de la complétude du dataset...", colors['info'])
    display.print(f"Total images dans le dataset : {total_images}\n", colors['info'])

    return model    


# --- Chargement image RGB avec gestion erreurs ---
def load_rgb(path):
    
    display = dc.DisplayColor()

    try:
        return Image.open(path).convert("RGB")
    except Exception:
        display.print(f"Impossible de charger l'image : {path}", colors['error'])
        return None

# --- Normalisation L2 ---
def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

# --- Encodage des images avec skip intelligent et batch save ---
def encoding(dataset, VEC_FIELD, total_images, DEVICE, model):

    # --- Prétraitement standard pour DINOv3 ---
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),
    ])

    display = dc.DisplayColor()
    existing = dataset.match(F(VEC_FIELD) != None)

    if len(existing) == total_images:
        display.print(f"Embeddings déjà présents, skip encodage.", colors['warning'])
    else:

        filepaths = dataset.values("filepath")
        sample_ids = dataset.values("id")
        display.print(f"Encodage des {total_images} images...", colors['info'])
        
        executor = ThreadPoolExecutor(max_workers=ct.NUM_WORKERS)
        all_embeddings = {}

        batch_size = ct.BATCH_SIZE
        total_batches = (total_images + batch_size - 1) // batch_size

        # --- Barre principale sur images, secondaire sur batch ---
        with tqdm(
            total=total_images,
            desc="Images encodées",
            unit="img",
            position=0,
            leave=True,
            ncols=100,
            dynamic_ncols=False
        ) as pbar, tqdm(
            total=0,
            desc="",
            position=1,
            bar_format="{desc}",
            leave=False,
            ncols=100,
            dynamic_ncols=False
        ) as mbar:

            for batch_idx, start in enumerate(range(0, total_images, batch_size), 1):
                end = min(start + batch_size, total_images)
                batch_paths = filepaths[start:end]
                batch_ids = sample_ids[start:end]

                # Mettre à jour la barre secondaire avec le batch courant
                mbar.set_description(f"Batch {batch_idx}/{total_batches}")
                mbar.refresh()

                # Chargement + prétraitement
                images = list(executor.map(lambda p: preprocess(load_rgb(p)), batch_paths))
                x = torch.stack(images).to(DEVICE)

                # Passage dans le modèle
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

                for sid, vec in zip(batch_ids, feats):
                    all_embeddings[sid] = vec.tolist()

                # Mise à jour de la barre principale par le nombre d'images traitées
                pbar.update(len(batch_paths))

        # Enregistrement
        dataset.set_values(VEC_FIELD, all_embeddings, key_field="id")
        dataset.save()
        executor.shutdown()
        display.print("Embeddings enregistrés.\n", colors['info'])



#------------------------------------------------------------------------------------------------
def main():
    # ================= CONFIG =================

    dataset_name = "coco128_local"
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

    
    # Display du logo et infos système
    print()
    splash_screen_circle("Image.png", duration=3000)
    display.print(ct.INFO_PROD, colors['aqua'])
    
    # Affichge du mode de débugage
    display.print(f"Debug mode {'ON' if ct.DEBUG_MODE else 'OFF'}.", colors['warning'])
 
    # Affichage si mode de rapport actif
    path_user = ct.PATH_USER
    if not os.path.exists(path_user):
        path_user = Path.cwd()
        display.print(f"Utilisation du répertoire de travail", colors["warning"])
    status = f"ON : Sauvegarde dans :\n    {path_user}" if ct.REPORT_MODE else "OFF"
    display.print(f"Report mode {status}.\n", colors['warning'])
 

    if ct.TEST_MODE :
        # Pour les simulation 
        DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco128"
        MODEL_PATH = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Fitty_One\Model\DINOv3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    else :
        DATASET_DIR = util.get_path_color("Entrée le chemin du dataset")
        MODEL_PATH = util.get_path_color("Entrée le chemin du modèle DINOv3")

    display.print("Démarrage du traitement ...", colors['info'])

    # Chargement des noms de classes pour les stats
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    try:
        class_names = ds.load_class_names(dataset_yaml)
    except Exception as e:
        display.print(f"dataset.yaml introuvable dans {DATASET_DIR}", colors['error'])
        exit(1)

    # validation des labels avant création du dataset FiftyOne
    erreur, all_bboxes, rapport, ctrl_ok = bb.validate_yolo_dataset_detailed(DATASET_DIR, path_user)
 
    if not ctrl_ok:
        display.print(f"Erreurs détectées dans les images/labels. Arrêt du programme {ct.BELL}", colors['error'])
        total_errors = sum(len(v) for v in erreur.values())
        
        label1 = "Total Types           :"
        label2 = "Total warning/erreurs :"
                
        value1 = len(erreur)
        value2 = total_errors

        label_width = max(len(label1), len(label2))
        value_width = max(len(str(value1)), len(str(value2)))

        display.print(f"{label1:<{label_width}} {value1:>{value_width}}", colors['error'])
        display.print(f"{label2:<{label_width}} {value2:>{value_width}}", colors['error'])    
        print()
        util.afficher_bbox_erreurs_compact(erreur)
     
    else:    
        display.print("Aucune erreur détectée. Analyse du Dataset...\n", colors['ok'])
        # create_dataset(DATASET_DIR)

        statistique(DATASET_DIR, path_user)

        # Création du dataset FiftyOne
        dataset = create_dataset(DATASET_DIR)

        total_images = len(dataset)
        # sample_ids = dataset.values("id")
        # filepaths = dataset.values("filepath")

        model = load_model(MODEL_PATH, total_images, DEVICE)

        # ================= ENCODING =================
        encoding(dataset, VEC_FIELD, total_images, DEVICE, model )
 


    print()
    prompt = f"Script terminé.{ct.BELL}"
    display.print(prompt, colors['goodbye'])

#==========================================================================================
if __name__ == "__main__":
    main()

