import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from collections import defaultdict

from statistics_yolo import imbalance as imb
import statistics_yolo.info_general as ige
import statistics_yolo.info_classe as icl
import statistics_yolo.info_img_classe as iic
import statistics_yolo.anomalies as ano

import tools.display_color as dc
from tools import utility as util
from config import constants as ct
from config.constants import DISPLAY_COLORS as colors
from tools import graphe as gr


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

def dataset_statistics_yolo(DATASET_DIR, cfg):

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

    # --- lecture labels ---
    for label_file in labels_dir.glob("*.txt"):

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
        f for f in Path(images_dir).rglob("*.*")
        if f.suffix.lower() in ct.IMAGE_EXT
        ])
    label_count = len(list(Path(labels_dir).rglob("*.txt")))


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
    for img_name, x, y, w, h in zip(
        image_paths,
        bbox_centers_x,
        bbox_centers_y,
        bbox_widths,
        bbox_heights
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

    return {

        "stats": stats,
        "class_distribution": class_distribution,
        "anomalies": anomalies,
        "bbox_areas": bbox_areas,

        "classes": classes,
        "image_names": image_paths,
        "class_to_images": class_to_images,
    }


#==========================================================================================================

def afficher_dataset_statistics(resultats, cfg, path_user, class_names=None,  afficher_hist=False):

    display = dc.DisplayColor()

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]
    bbox_areas = resultats.get("bbox_areas", [])
    class_to_images = resultats.get("class_to_images", {})

    total_boxes = stats["bounding_boxes"]
    total = sum(class_distribution.values())
    total_classes = len(class_names) if class_names else max(class_distribution.keys()) + 1



    # ---- Information générales -----------------------------------------------
    info_general = (total_boxes, total_classes, class_distribution, class_to_images )
    ige.afficher_info_general(stats, info_general, class_names, cfg)

    # ---- Information sur les classes -----------------------------------------
    classes_info =(class_distribution, class_names)
    icl.info_classes(classes_info, cfg)

   
    # --- IMAGES PAR CLASSE --------------------------------------------------------
    data_info_img_cla = (class_to_images, class_names)
    iic.info_images_par_classe(data_info_img_cla)


    # --- histogramme des tailles de bbox ---
    if afficher_hist and bbox_areas:
        gr.histogram_taille_bbox(bbox_areas,
                      "Distribution des tailles de BBox",
                      "Aire bbox",
                      "Nombre",
                      cfg,
                      )


    # --- anomalies ---------------------------------------------------------------
    info_anomalie = (anomalies, resultats, afficher_hist)
    ano.recherche_anomalie(stats, info_anomalie, path_user, cfg)


    # --- AFFICHAGE DES METRIQUES D'IMBALANCE ---
    metrics = imb.imbalance_metrics(class_distribution, cfg)
    imb.afficher_imbalance_avance(metrics, display, colors, cfg)

    return anomalies