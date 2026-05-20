import os
from tools import system as syst

if syst.est_linux():
    os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")

# TOUT le reste des imports AVANT fiftyone
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tqdm import tqdm
import timm
import random


# ONLY HERE
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F
from PIL import Image

from bboxes import bboxes as bb
from statistics_yolo import dataset_statistics_yolo as ds
from config import valid_conf as vc
from config.process import load_config
from config import constants as ct
from tools import utility as util

import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from tools import graphe as gr

from tools import logo_win as lw
from tools import logo_linux as ll

#==========================================================================================
# ================= FONCTIONS =================



def def_status(etat, path_user):
       
    display = dc.DisplayColor()
 
    status: str = (
        f"ON : Saving to:\n    {path_user}"
        if etat
        else "OFF"
    )
    return status

def statistique(DATASET_DIR, cfg, class_names, path_user):
     # ================= STATISTICS =================
    dataset_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    # class_names = ds.load_class_names(dataset_yaml)



    results = ds.dataset_statistics_yolo(DATASET_DIR, cfg)


    seuils = util.calibrer_seuils_overflow(
        results,
        warning_percentile= cfg["PERCENTILE_WARNING"],
        error_percentile= cfg["PERCENTILE_ERROR"],
        min_warning= cfg["MIN_BBOX_OVERFLOW_WARNING"],
        min_error= cfg["MIN_BBOX_OVERFLOW_ERROR"]
    )
    
    BBOX_OVERFLOW_WARNING = seuils['BBOX_OVERFLOW_WARNING']
    BBOX_OVERFLOW_ERROR   = seuils['BBOX_OVERFLOW_ERROR']

    outside_ratios = [a['outside_ratio_pct'] for a in results.get('anomalies',
                                            []) if 'outside_ratio_pct' in a]

    if outside_ratios :
        gr.bbox_overflow(cfg, outside_ratios, BBOX_OVERFLOW_WARNING, BBOX_OVERFLOW_ERROR) 

    resultat = ds.afficher_dataset_statistics(results, cfg, path_user, class_names, classes_par_ligne=4, afficher_hist=True)

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

    # ✅ mini barre
    progress = util.MiniProgressBar("Chargement dataset", width=20)
    progress.start()

    dataset = fo.Dataset(dataset_name)

    progress.stop()

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
            line = line.strip()

            # ✅ ignorer ligne vide
            if not line:
                continue

            parts = line.split()

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
    fo.config.show_progress_bars = False 

    display = dc.DisplayColor()

    # Efface l'écran avant de commencer
    syst.clear_screen()

    # Display du logo et infos système
    if syst.est_windows():
        lw.splash_screen_circle("Image1.png") 
    else:
        ll.splash_screen_circle("Image1.png")

    display.print(ct.INFO_PROD, colors['aqua'])


    # Chargement & Vérification du fichier de Paramètrage
    cfg = load_config()
    print()
    vc.controle(cfg)
    print()

    # ================= REPRO =================
    if ct.SEED != 0 :
        torch.manual_seed(ct.SEED)
        torch.cuda.manual_seed_all(ct.SEED)
        np.random.seed(ct.SEED)
        random.seed(ct.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda": 
        prompt = ("CUDA available - Running on 'GPU'")
        display.print(prompt, colors['ok'])
    else:
        prompt = ("CUDA not available - Running on 'CPU'")
        display.print(prompt, colors['warning'])
    print()

    print()
    # Contrôle répertoire de sauvegarde
    try:
        path_user: Path = Path(cfg["PATH_USER"])
        if not path_user.exists():
            path_user = Path.cwd()
            display.print("Chemin non défini : Utilisation du répertoire de travail actuel", colors["error"])
    except Exception as e : 
        path_user = Path.cwd()
        display.print("Chemin non défini : Utilisation du répertoire de travail actuel", colors["error"])
 
    # Report mode handling
    status = def_status(cfg["REPORT_MODE"], path_user)
    display.print(f"Report mode {status}.\n", colors['warning'])
 
    # Graphe mode handling
    status = def_status(cfg["SAVE_PLOT"], path_user)
    display.print(f"Save Plot mode {status}.\n", colors['warning'])


    if ct.TEST_MODE :
        # Pour les simulation
        if syst.est_windows(): 
            # DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco128"
            DATASET_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Data\coco_small"
            MODEL_DIR = r"C:\Users\Pierre.FANCELLI\Documents\___Dev\Aqua-IA\Fitty_One\Model\DINOv3"
        else :
            DATASET_DIR = r"/media/DataLinux/Travail/_AKKA/___Akka_Reacher/Data/coco_small_Work"
            MODEL_DIR = r"/media/DataLinux/Travail/_AKKA/___Akka_Reacher/2026/Aqua-/Fitty_One/Model/DINOv3"
        
        MODEL_NAME = r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

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
        display.print(f"dataset.yaml introuvable dans {DATASET_DIR}\n", colors['error'])
        util.sortie_de_programme()

    # validation des labels avant création du dataset FiftyOne
    erreur, ctrl_ok = bb.validate_yolo_dataset_detailed(DATASET_DIR, path_user, cfg)
 
    
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

        def_image = statistique(DATASET_DIR, cfg, class_names, path_user) # type: ignore

        if not def_image:
            display.print("Dataset Ok ", colors['ok'])
            
            
            dataset = create_dataset(DATASET_DIR)
                
            if util.answer_yes_or_no("Voulez-vous lancer Fifty_one"):
                # launch interface FiftyOne
                util.launch_fiftyone_interface(dataset) # type: ignore

            # ================= ENCODING =================
            total_images = len(dataset) # type: ignore
            MODEL_DIR = Path(MODEL_DIR) # type: ignore
            model_ = MODEL_DIR / MODEL_NAME
            model = load_model(model_, total_images, DEVICE)

            encoding(dataset, cfg["VEC_FIELD"], total_images, DEVICE, model )
                

        else:
            display.print("Dataset Not Ok ", colors['warning'])
            display.print(" Création d'un dataset d'anomalies ", colors['warning'])
            dataset = create_anomaly_dataset(def_image, DATASET_DIR)
        
            # launch interface FiftyOne
            util.launch_fiftyone_interface(dataset)


    print()
    util.sortie_de_programme()

#==========================================================================================
if __name__ == "__main__":
    main()

