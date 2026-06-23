import os
from collections import defaultdict

import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from collections import defaultdict
import fiftyone as fo
import fiftyone.core.labels as fol


from statistics_yolo import imbalance as imb
import statistics_yolo.info_general as ige
import statistics_yolo.info_classe as icl
import statistics_yolo.info_img_classe as iic
import statistics_yolo.anomalies as ano
import statistics_yolo.lautch_fifty_one as lfo

from tools import system as syst
import tools.display_color as dc
from tools import utility as util
from tools import rapport as rp
from tools import menu as menu
from config import constants as ct
from config.constants import DISPLAY_COLORS as colors
from tools import graphe as gr

display = dc.DisplayColor()

#==========================================================================================

# --- fonction utilitaire pour stats ---
def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def save_anomalies_readable(
    anomalies: list[util.Anomaly],
    file_name: str,
    path_user: Path
    ) -> None:
    """
    Sauvegarde les anomalies dans un fichier texte lisible :
    - Résumé des anomalies par type
    - Images regroupées par type
    - Plusieurs images par ligne (configurable via ct.N_PER_LINE)
    """

    # Regroupement par type
    anomalies_by_type = defaultdict(set)
    for a in anomalies:
        typ = a.get("type")
        img_name = os.path.basename(a.get("image", "unknown"))
        if typ and img_name:
            anomalies_by_type[typ].add(img_name)

    # Tri alphabétique des images par type
    for typ in anomalies_by_type:
        anomalies_by_type[typ] = sorted(anomalies_by_type[typ]) # type: ignore

    new = util.horodatage(file_name, defaut="def")
    output_path =  path_user / new

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # --- Résumé ---
            if len(anomalies) == 0:
                f.write("[√] Aucune anomalie détectée \n\n")
            else:

                f.write(f" - {util.format_nombre(len(anomalies))} anomalie(s) détectée(s) :\n") 
                
                anomalies_by_type = defaultdict(list)

                for a in anomalies:
                    typ = a["type"] # type: ignore
                    img = os.path.basename(a["image"]) # type: ignore
                    anomalies_by_type[typ].append(img)

                for typ, imgs in anomalies_by_type.items():
                    f.write(
                        f"   - {typ}: {util.format_nombre(len(imgs))} anomalie(s) sur {util.format_nombre(len(set(imgs)))} image(s)\n"
                    )
                
                f.write("\n")
                # --- Détails par type ---
                for typ, images in anomalies_by_type.items():
                    f.write(f"--- {typ} ---\n")
                    for i in range(0, len(images), ct.N_PER_LINE):
                        line_images = images[i:i+ ct.N_PER_LINE] # type: ignore
                        f.write(" | ".join(line_images) + "\n")
                    f.write("\n")

                display.print(f" ****** '{file_name}' create *****", colors["warning"])

    except FileNotFoundError:
             display.print(f"Impossible de sauvegarder : {output_path}", colors['error'])

def group_anomalies(anomalies):
    grouped = defaultdict(list)
    for a in anomalies:
        grouped[a["image"]].append(a)
    return grouped

def create_dataset(DATASET_DIR,  yaml_path=None, anomalies=None):

    dataset_name = (
                f"{Path(DATASET_DIR).name}"
                f"{'_def' if anomalies is not None else '_ok'}"
                )

    display.print("Création du dataset FiftyOne :", colors['info'])
    print(f"    '{dataset_name}'")

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    fo.close_app()

    # Mini barre
    progress = util.MiniProgressBar("Chargement dataset", width=20)
    progress.start()


    try:
        # =========================================================
        # CAS 1 : dataset classique YOLO
        # =========================================================
        if anomalies is None:
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.YOLOv5Dataset, # type: ignore
                dataset_dir=str(DATASET_DIR),
                yaml_path=str(yaml_path),
                name=dataset_name
            )

        # =========================================================
        # CAS 2 : dataset avec anomalies
        # =========================================================
        else:

            dataset = fo.Dataset(dataset_name)

            grouped = group_anomalies(anomalies)
            samples = []

            for img_name, image_anomalies in grouped.items():

                image_path = (
                    Path(DATASET_DIR) / "images/train2017" / img_name
                )

                label_path = (
                    Path(DATASET_DIR)
                    / "labels/train2017"
                    / img_name.replace(".jpg", ".txt")
                )

                if not image_path.exists() or not label_path.exists():
                    print(f"Fichier manquant pour {img_name}")
                    continue

                detections = []

                # Lire fichier YOLO
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:

                    line = line.strip()

                    # ignorer ligne vide
                    if not line:
                        continue

                    parts = line.split()

                    x_center, y_center, width, height = map(
                        float,
                        parts[1:5]
                    )

                    # Associer bbox aux anomalies
                    for anomaly in image_anomalies:

                        if (
                            abs(anomaly["width"] - width) < 1e-6
                            and abs(anomaly["height"] - height) < 1e-6
                        ):

                            # YOLO → FiftyOne
                            x = x_center - width / 2
                            y = y_center - height / 2

                            detection = fol.Detection(
                                label=anomaly["type"],
                                bounding_box=[x, y, width, height],
                                confidence=1.0,
                            )

                            detections.append(detection)

                sample = fo.Sample(filepath=str(image_path))
                sample["anomalies"] = fol.Detections(
                    detections=detections
                )

                samples.append(sample)

            dataset.add_samples(samples)

    finally:
        progress.stop()

    display.print(f"Dataset créé avec {util.format_nombre(len(dataset))} images", colors['ok'])  # type: ignore

    return dataset


#==========================================================================================================

def dataset_statistics_yolo(DATASET_DIR, rapport, cfg):

    display.print(" Analyse statistique", colors['info'])

    images_dir, labels_dir = util.get_dataset_paths(DATASET_DIR)

    split_pattern = re.compile(r"[,\s]+")

    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    classes = []
    image_paths = []
    class_to_images = defaultdict(set)
    bbox_centers_x = []
    bbox_centers_y = []

    total_boxes = 0

    # --- Début de l'analyse du Dataser -------------------------------------------------
    # --- Début Analyse statistique -----------------------------------------------------
    # --- lecture labels ---

    debut = rp.suivi("Analyse statistique", rapport, "D")

    label_files = labels_dir.glob("*.txt")
    for label_file in tqdm(
        label_files,
        total=len(list(labels_dir.glob("*.txt"))),
        desc=" - Labels",
        unit=" fichier",
        ncols=100,
        position=0
    ):

        image_name = label_file.with_suffix(".jpg").name

        with label_file.open("r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = split_pattern.split(line)

                if len(parts) != 5:
                    continue

                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except:
                    continue

                classes.append(cls)
                class_to_images[cls].add(image_name)

                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_centers_x.append(x)
                bbox_centers_y.append(y)
                bbox_areas.append(w * h)

                image_paths.append(image_name)

                total_boxes += 1


    image_count = len([
        f for f in Path(images_dir).glob("*.*")
        if f.suffix.lower() in ct.IMAGE_EXT
        ])
    label_count = len(list(Path(labels_dir).glob("*.txt")))


    # --- statistiques générales ---
    stats = {
        "images": image_count,
        "labels": label_count,
        "bounding_boxes": total_boxes,
        "bbox_width": compute_stats(bbox_widths),
        "bbox_height": compute_stats(bbox_heights),
        "bbox_area": compute_stats(bbox_areas)
    }

    class_distribution = Counter(classes)
    anomalies = []

    # --- détection anomalies bbox ---

    for img_name, x, y, w, h in tqdm(
        zip(
            image_paths,
            bbox_centers_x,
            bbox_centers_y,
            bbox_widths,
            bbox_heights
        ),
        desc=" - Bboxes",
        unit=" bbox",
        ncols=100,
        total=len(image_paths),  # indispensable
        position=0
        ):

        area = w * h

        if w < cfg["MIN_BBOX"] or h < cfg["MIN_BBOX"]:
            anomalies.append({
                "type": "bbox_trop_petite",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


        if w > cfg["MAX_BBOX"] or h > cfg["MAX_BBOX"]:
            anomalies.append({
                "type": "bbox_trop_grande",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


        if area < cfg["MIN_BBOX_AREA"]:
            anomalies.append({
                "type": "bbox_surface_trop_petite",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


        if area > cfg["MAX_BBOX_AREA"]:
            anomalies.append({
                "type": "bbox_surface_trop_grande",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


       # --- Détection des bbox hors limites (YOLO) ---
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2

        # --- Overflow par côté ---
        overflow_left   = max(0 - x_min, 0)
        overflow_right  = max(x_max - 1, 0)
        overflow_top    = max(0 - y_min, 0)
        overflow_bottom = max(y_max - 1, 0)

        # --- Overflow max et total ---
        overflow_max = max(overflow_left, overflow_right, overflow_top, overflow_bottom)
        overflow_sum = overflow_left + overflow_right + overflow_top + overflow_bottom

        overflow_max_pct = overflow_max * 100
        overflow_sum_pct = overflow_sum * 100

        # --- Surface visible et ratio hors image ---
        visible_w = max(0, min(x_max, 1) - max(x_min, 0))
        visible_h = max(0, min(y_max, 1) - max(y_min, 0))
        visible_area = visible_w * visible_h
        bbox_area = w * h
        outside_ratio = 1 - (visible_area / bbox_area) if bbox_area > 0 else 0
        outside_ratio_pct = outside_ratio * 100

        # --- Détection anomalies ---
        # ERROR
        if outside_ratio_pct > cfg["MIN_BBOX_OVERFLOW_ERROR"]:
            anomalies.append({
                "type": "bbox_hors_limite_error",
                "width": w,
                "height": h,
                "overflow_max_pct": overflow_max_pct,
                "overflow_sum_pct": overflow_sum_pct,
                "outside_ratio_pct": outside_ratio_pct,
                "image": img_name
            })

        # WARNING
        elif outside_ratio_pct >= cfg["MIN_BBOX_OVERFLOW_WARNING"]:
            anomalies.append({
                "type": "bbox_hors_limite_warning",
                "width": w,
                "height": h,
                "overflow_max_pct": overflow_max_pct,
                "overflow_sum_pct": overflow_sum_pct,
                "outside_ratio_pct": outside_ratio_pct,
                "image": img_name
            })

    fin =rp.suivi("Analyse statistique", rapport)

    # calcul du temps d'analyse
    rp.temps_de_traitement(debut, fin ,rapport )

    return {

        "stats": stats,
        "class_distribution": class_distribution,
        "anomalies": anomalies,
        "bbox_areas": bbox_areas,

        "classes": classes,
        "image_names": image_paths,
        "class_to_images": class_to_images,
    }


def afficher_dataset_statistics(DATASET_DIR,resultats, cfg,  dataset_yaml, path_user, class_names=None):

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]
    bbox_areas = resultats.get("bbox_areas", [])
    class_to_images = resultats.get("class_to_images", {})

    total_boxes = stats["bounding_boxes"]
    # total = sum(class_distribution.values())
    total_classes = len(class_names) if class_names else max(class_distribution.keys()) + 1


    print()
    display.print(f"Analyse statistique terminé {'avec' if anomalies else 'sans'} problème",
                  colors["warning" if anomalies else "ok"]
                  )

    main_menu = menu.Menu('MAIN', style= "heavy")
    while True : 

        display.print("Menu", colors['titre'])
        main_menu.display_menu()
        choice = main_menu.selection()

        if choice == 1:
            # ---- Information générales --------------------------------------------------
            syst.clear_screen()
            display.print("Information générales", colors['titre'])
            info_general = (total_boxes, total_classes, class_distribution, class_to_images )
            ige.afficher_info_general(stats, info_general, class_names, cfg)

        elif choice == 2:
            # ---- Information sur les classes --------------------------------------------
            syst.clear_screen()
            display.print("Information sur les classes", colors['titre'])
            classes_info =(class_distribution, class_names)

            icl.info_classes(classes_info, False, cfg)
                     
        elif choice == 3:   
            # --- Images par classe -------------------------------------------------------
            syst.clear_screen()
            display.print("Images par classe", colors['titre'])
            data_info_img_cla = (class_to_images, class_names)
            iic.info_images_par_classe(data_info_img_cla, False)

        elif choice == 4:
            # --- histogramme des tailles de bbox -----------------------------------------
            syst.clear_screen()
            display.print("Distribution de la taille des BBoxes", colors['titre'])

            nb_bins = ct.BINS

            counts, edges = np.histogram(bbox_areas, bins=nb_bins)
            y_max = counts.max()

            print(f" - Nombre maximum d'occurrences : {util.format_nombre(y_max)}")
            print()

            y_max_affichage = y_max

            while True:

                display.print("Attente fermeture du graphe", colors['wait'])
                gr.histogram_taille_bbox(
                    bbox_areas,
                    "Distribution des tailles de BBox",
                    "Aire bbox",
                    "Nombre",
                    cfg,
                    y_max_affichage,
                )

                if not util.answer_yes_or_no("Voulez-vous modifier la valeur de 'y_max'"):
                    break

                y_max_affichage = int(
                    util.input_value("Entrer une valeur")
                )

        elif choice == 5:
            # --- anomalies ---------------------------------------------------------------
            syst.clear_screen()
            display.print("Anomalies", colors['titre'])
            info_anomalie = (anomalies, resultats)
            ano.recherche_anomalie(stats, info_anomalie, path_user, False, cfg)

        elif choice == 6:
            # --- Affichage Métriques d'imbalance -----------------------------------------
            syst.clear_screen()
            display.print("Métriques d'imbalance", colors['titre'])
            metrics = imb.imbalance_metrics(class_distribution, cfg)
            imb.afficher_imbalance_avance(metrics, display, colors, cfg)

        elif choice == 7:
            # --- Lancement de Fifty_One
            syst.clear_screen()
            display.print("Lancement de FiftyOne", colors['titre'])

            data_fifty_one =(anomalies, DATASET_DIR, dataset_yaml)
            lfo.lautch_fifty_one(data_fifty_one)

        elif choice == 8:
            # --- Sortie ------------------------------------------------------------------
            if util.answer_yes_or_no("Voulez-vous sortir", True):
                break

    return   # anomalies

    #====================================================================================
    #====================================================================================

def file_dataset_statistics(resultats, cfg, path_user, class_names=None):

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]
    class_to_images = resultats.get("class_to_images", {})

    total_boxes = stats["bounding_boxes"]
    # total = sum(class_distribution.values())
    total_classes = len(class_names) if class_names else max(class_distribution.keys()) + 1

    print()
    display.print(f"Analyse statistique terminé {'avec' if anomalies else 'sans'} problème",
                    colors["warning" if anomalies else "ok"]
                    )

    #  1 ---- Information générales --------------------------------------------------
    display.print("Information générales", colors['titre'])
    info_general = (total_boxes, total_classes, class_distribution, class_to_images )
    ige.afficher_info_general(stats, info_general, class_names, cfg)


    # 2 ---- Information sur les classes ---------------------------------------------
    display.print("Information sur les classes", colors['titre'])
    classes_info =(class_distribution, class_names)
    icl.info_classes(classes_info, True, cfg)    
                

    # # 3 --- Images par classe ------------------------------------------------------
    display.print("Images par classe", colors['titre'])
    data_info_img_cla = (class_to_images, class_names)
    iic.info_images_par_classe(data_info_img_cla, True)

    # # 4 --- histogramme des tailles de bbox ----------------------------------------
    # Pas de données à afficher

    # 5 --- anomalies ---------------------------------------------------------------
    display.print("Anomalies", colors['titre'])
    info_anomalie = (anomalies, resultats)
    ano.recherche_anomalie(stats, info_anomalie, path_user, True, cfg)

    # 6 --- Métriques d'imbalance -----------------------------------------
    display.print("Métriques d'imbalance", colors['titre'])
    metrics = imb.imbalance_metrics(class_distribution, cfg)
    imb.afficher_imbalance_avance(metrics, display, colors, cfg)

    # --- Rapport de défauts de conformité
    save_anomalies_readable(anomalies, "erreurs_dataset.txt", path_user)

    print("\n")

    return 
