import os
# os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")

# TOUT le reste des imports AVANT fiftyone
import platform
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import faiss
from tqdm import tqdm
import timm
import random
import tkinter as tk

# ONLY HERE
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from PIL import Image, ImageTk, ImageDraw, ImageFont

from bboxes import bboxes as bb
from statistics_yolo import dataset_statistics_yolo as ds

from tools import utility as util
from config import process as pr
from config import constants as ct
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from graphe import graphe as gr


#==========================================================================================
# ================= FONCTIONS =================
 
def est_windows():
    return platform.system().lower() == "windows"
    
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

def def_status(etat, path_user):
       
    display = dc.DisplayColor()
 
    status: str = (
        f"ON : Saving to:\n    {path_user}"
        if etat
        else "OFF"
    )
    return status

def statistique(DATASET_DIR, path_user):
     # ================= STATISTICS =================
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    class_names = ds.load_class_names(dataset_yaml)

    results = ds.dataset_statistics_yolo(DATASET_DIR)
    seuils = util.calibrer_seuils_overflow(
        results,
        warning_percentile= pr.PERCENTILE_WARNING,
        error_percentile= pr.PERCENTILE_ERROR,
        min_warning= pr.MIN_BBOX_OVERFLOW_WARNING,
        min_error= pr.MIN_BBOX_OVERFLOW_ERROR
    )
    
    BBOX_OVERFLOW_WARNING = seuils['BBOX_OVERFLOW_WARNING']
    BBOX_OVERFLOW_ERROR   = seuils['BBOX_OVERFLOW_ERROR']

    outside_ratios = [a['outside_ratio_pct'] for a in results.get('anomalies',
                                            []) if 'outside_ratio_pct' in a]

    if outside_ratios :
        gr.bbox_overflow(outside_ratios, BBOX_OVERFLOW_WARNING, BBOX_OVERFLOW_ERROR, path_user) 

    resultat = ds.afficher_dataset_statistics(results, path_user, class_names, classes_par_ligne=4, afficher_hist=True)

    return resultat


def group_anomalies(anomalies):
    grouped = defaultdict(list)
    for a in anomalies:
        grouped[a["image"]].append(a)
    return grouped

def create_anomaly_dataset(anomalies, DATASET_DIR):

    display = dc.DisplayColor()

    dataset_name = "anomalies_dataset"

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)

    grouped = group_anomalies(anomalies)

    samples = []

    for img_name, image_anomalies in grouped.items():

        image_path = Path(DATASET_DIR) / "images/train2017"  /img_name
        label_path = Path(DATASET_DIR) / "labels/train2017" / img_name.replace(".jpg", ".txt")

        if not image_path.exists() or not label_path.exists():
            print(f"Fichier manquant pour {img_name}")
            continue

        detections = []

        # Lire fichier YOLO
        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])

            # Associer bbox aux anomalies via width/height
            for anomaly in image_anomalies:
                if abs(anomaly["width"] - width) < 1e-6 and \
                   abs(anomaly["height"] - height) < 1e-6:

                    # YOLO → FiftyOne format
                    x = x_center - width / 2
                    y = y_center - height / 2

                    detection = fol.Detection(
                        label=anomaly["type"],
                        bounding_box=[x, y, width, height],
                        confidence=1.0,
                    )

                    detections.append(detection)

        sample = fo.Sample(filepath=str(image_path))
        sample["anomalies"] = fol.Detections(detections=detections)

        samples.append(sample)

    dataset.add_samples(samples)

    display.print(f"Dataset anomalies créé : {len(dataset)} images", colors['ok']) # type: ignore
    
    return dataset





def create_dataset(DATASET_DIR):
    display = dc.DisplayColor()
    dataset_name = "coco_small_local"

    display.print("Création du dataset FiftyOne :", colors['info'])
    print(f"    '{DATASET_DIR}'")

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    fo.close_app()     

    # ✅ mini barre
    progress = util.MiniProgressBar("Chargement dataset", width=20)
    progress.start()

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset, # type: ignore
        dataset_dir=str(DATASET_DIR),
        name=dataset_name
    )

    progress.stop()

    total_images = len(dataset) # type: ignore

    display.print(
        f" Dataset chargé : {total_images} images",
        colors['info']
    )

    return dataset



def load_model(MODEL_NAME, total_images, DEVICE):

    display = dc.DisplayColor()
    # ================= MODEL =================
    display.print("Chargement du modèle DINOv3...", colors['info'])

    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    state_dict = torch.load(MODEL_NAME, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()

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
        
        executor = ThreadPoolExecutor(max_workers= ct.NUM_WORKERS)
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
                x = torch.stack(images).to(DEVICE) # type: ignore

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
        display.print("Embeddings terminés et enregistrés.\n", colors['ok'])



#------------------------------------------------------------------------------------------------
def main():
    # ================= CONFIG =================

    dataset_name = "coco128_local"
    VEC_FIELD = "emb_dinov3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    fo.config.show_progress_bars = False 

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

    # Efface l'écran avant de commencer
    util.clear_screen()
    
    # Display du logo et infos système
    print()
    if est_windows():
        splash_screen_circle("Image.png", duration=3000)
    display.print(ct.INFO_PROD, colors['aqua'])
    

    if torch.cuda.is_available: 
        prompt = ("CUDA available - Running on 'GPU'")
        display.print(prompt, colors['ok'])
    else:
        prompt = ("CUDA not available - Running on 'CPU'")
        display.print(prompt, colors['warning'])
    print()


    # Affichge du mode de débugage
    display.print(f"Debug mode {'ON' if ct.DEBUG_MODE else 'OFF'}.", colors['warning'])
 
    print()
    # Contrôle répertoire de sauvegarde
    try:
        path_user: Path = Path(pr.PATH_USER)
        if not path_user.exists():
            path_user = Path.cwd()
            display.print("Path not defied : Using current working directory", colors["error"])
    except Exception as e : 
        path_user = Path.cwd()
        display.print("Path not defied : Using current working directory", colors["error"])
 
    # Report mode handling
    status = def_status(pr.REPORT_MODE, path_user)
    display.print(f"Report mode {status}.\n", colors['warning'])
 
    # Graphe mode handling
    status = def_status(pr.SAVE_PLOT, path_user)
    display.print(f"Save Plot mode {status}.\n", colors['warning'])


    if ct.TEST_MODE :
        # Pour les simulation
        if est_windows(): 
            # DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco128"
            DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco_small"
            MODEL_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Fitty_One\Model\DINOv3"
            MODEL_NAME = r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        else :
            DATASET_DIR = r"/media/DataLinux/Travail/_AKKA/___Akka_Reacher/2026/Aqua-/AQUA/datasets/coco128"
            MODEL_NAME = r"/media/DataLinux/Travail/_AKKA/___Akka_Reacher/2026/Aqua-/Fitty_One/Model/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    else :
        DATASET_DIR = util.get_path_color("Entrée le chemin du dataset")
        MODEL_DIR = util.get_path_color("Entrée le chemin du modèle")
        MODEL_NAME = util.get_file_name_color("Entrée le nom du modèle DINOv3")

    print()
    display.print("Démarrage du traitement ...", colors['info'])

    # Chargement des noms de classes pour les stats
    DATASET_DIR  = Path(DATASET_DIR )
    dataset_yaml = DATASET_DIR / "dataset.yaml"
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

        def_image = statistique(DATASET_DIR, path_user)

        if not def_image:
            display.print("Dataset Ok ", colors['ok'])
            
            dataset = create_dataset(DATASET_DIR)

            total_images = len(dataset) # type: ignore
            MODEL_DIR = Path(MODEL_DIR) # type: ignore
            model_ = MODEL_DIR / MODEL_NAME
            # model_ =  os.path.join(MODEL_DIR, MODEL_NAME)
            model = load_model(model_, total_images, DEVICE)

            # ================= ENCODING =================
            encoding(dataset, VEC_FIELD, total_images, DEVICE, model )

        else:
            display.print("Dataset Not Ok ", colors['warning'])
            display.print(" Création d'un dataset d'anomalies ", colors['warning'])
            dataset = create_anomaly_dataset(def_image, DATASET_DIR)
        
            # launch interface FiftyOne
            util.launch_fiftyone_interface(dataset)

        

    print()

    prompt = f"Script terminé.{ct.BELL}"
    display.print(prompt, colors['goodbye'])

#==========================================================================================
if __name__ == "__main__":
    main()

