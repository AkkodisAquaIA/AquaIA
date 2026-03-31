
import os
import re
from networkx import display
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn import metrics
import yaml

import tools.display_color as dc
from tools import utility as util
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors
from graphe import graphe as gr

#==========================================================================================
def load_class_names(dataset_yaml_path):

    with open(dataset_yaml_path, "r", encoding="utf-8") as f:

        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, dict):

        names = [names[i] for i in sorted(names.keys())]

    return names

# --- fonction utilitaire pour stats ---
def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def imbalance_metrics(class_distribution):
    counts = np.array(list(class_distribution.values()))
    total = np.sum(counts)

    # probabilités
    p = counts / total

    # ratio
    ratio = np.max(counts) / np.min(counts)

    # entropie
    entropy = -np.sum(p * np.log(p + 1e-9))
    max_entropy = np.log(len(counts))
    entropy_norm = entropy / max_entropy

    return {
        "ratio": ratio,
        "entropy": entropy,
        "entropy_norm": entropy_norm
    }


def compute_global_score(ratio, entropy_norm):
    """
    Score global 0 → 100
    """
    # ratio pénalisé (log pour éviter explosion)
    ratio_score = max(0, 1 - (np.log10(ratio) / 3))  # ratio ~1000 → 0

    # entropie directe
    entropy_score = entropy_norm

    # pondération
    score = (0.6 * entropy_score + 0.4 * ratio_score) * 100

    return max(0, min(100, score))

def draw_bar(value, vmin, vmax, length=40):
    """
    Barre visuelle normalisée
    """
    ratio = (value - vmin) / (vmax - vmin)
    ratio = max(0, min(1, ratio))

    filled = int(ratio * length)
    empty = length - filled

    scale = "█" * filled + "░" * empty
    print(f"{'':25}  {scale}\n")

def evaluate_metric(
    label,
    value,
    thresholds,
    statuses,
    colors_map,
    bar_min,
    bar_max,
    display,
    higher_is_better=True,
    value_format="8.2f",
    suffix=""
):
    warning_th, ok_th = thresholds
    error_status, warning_status, ok_status = statuses

    if higher_is_better:
        if value >= ok_th:
            status = ok_status
            color = colors_map["ok"]
        elif value >= warning_th:
            status = warning_status
            color = colors_map["warning"]
        else:
            status = error_status
            color = colors_map["error"]
    else:
        # ✅ Cas inverse (ex: ratio)
        if value <= ok_th:
            status = ok_status
            color = colors_map["ok"]
        elif value <= warning_th:
            status = warning_status
            color = colors_map["warning"]
        else:
            status = error_status
            color = colors_map["error"]

    print(f"{label:25}: {value:{value_format}}{suffix}   ", end="")
    display.print(status, color)
    draw_bar(value, bar_min, bar_max)


def afficher_imbalance_avance(metrics, display, colors):

    ratio = float(metrics["ratio"])
    entropy_norm = float(metrics["entropy_norm"])

    print("\n---------------- DATASET IMBALANCE ----------------\n")

    # --- RATIO (plus petit = mieux) ---
    evaluate_metric(
    label="Ratio max/min",
    value=min(ratio, 300),
    thresholds=(ct.RATIO_WARNING, ct.RATIO_OK),
    statuses=("Très déséquilibré", "Déséquilibré", "Équilibré"),
    colors_map=colors,
    bar_min=1,
    bar_max=300,
    display=display,
    higher_is_better=False
)

    # --- ENTROPIE (plus grand = mieux) ---
    evaluate_metric(
    label="Entropie normalisée",
    value=entropy_norm,
    thresholds=(ct.ENTROPY_WARNING, ct.ENTROPY_OK),
    statuses=("Déséquilibré", "Moyennement équilibré", "Équilibré"),
    colors_map=colors,
    bar_min=0,
    bar_max=1,
    display=display,
    higher_is_better=True
)

    # --- SCORE GLOBAL ---
    score = compute_global_score(ratio, entropy_norm)

    evaluate_metric(
        label="Score global",
        value=score,
        thresholds=(ct.SCORE_WARNING, ct.SCORE_OK),
        statuses=("Faible", "Moyen", "Bon"),
        colors_map=colors,
        bar_min=0,
        bar_max=100,
        value_format="6.1f",
        suffix=" / 100",
        display=display,
        higher_is_better=True
    )

    # --- DIAGNOSTIC ---
    print("Diagnostic :")

    if ratio > 100:
        print("- Dataset très déséquilibré (ratio élevé)")

    if entropy_norm < 0.7:
        print("- Distribution globale déséquilibrée")

    if entropy_norm > 0.75 and ratio > 100:
        print("- Beaucoup de classes rares malgré une diversité correcte")

    # --- RECOMMANDATIONS ---
    print("\nRecommandations :")

    if ratio > 100:
        print("- Augmenter les classes très rares (< 1%)")

    if entropy_norm < 0.7:
        print("- Rééquilibrer globalement le dataset")

    print("- Utiliser data augmentation ciblée")
    print("- Envisager un sampling équilibré")

    print("--------------------------------------------------\n")

def dataset_statistics_yolo(DATASET_DIR):

    labels_dir = os.path.join(DATASET_DIR, "labels", "train2017")
    images_dir = os.path.join(DATASET_DIR, "images", "train2017")

    split_pattern = re.compile(r"[,\s]+")

    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    classes = []
    image_paths = []

    total_boxes = 0

    # --- lecture labels ---
    for entry in os.scandir(labels_dir):

        if not entry.name.endswith(".txt"):
            continue

        image_name = os.path.splitext(entry.name)[0] + ".jpg"

        with open(entry.path, "r", encoding="utf-8") as f:

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

                bbox_widths.append(w)
                bbox_heights.append(h)
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
    for img_name, w, h in zip(image_paths, bbox_widths, bbox_heights):
        area = w * h

        if w < ct.MIN_BBOX or h < ct.MIN_BBOX:
            anomalies.append({
                "type": "bbox_trop_petite",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


        if w > ct.MAX_BBOX or h > ct.MAX_BBOX:
            anomalies.append({
                "type": "bbox_trop_grande",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


        if area < ct.MIN_BBOX_AREA:
            anomalies.append({
                "type": "bbox_surface_trop_petite",
                "width": w,
                "height": h,
                "area": area,
                "image": img_name
            })


        if area > ct.MAX_BBOX_AREA:
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
        if outside_ratio_pct > ct.BBOX_OVERFLOW_ERROR:
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
        elif outside_ratio_pct >= ct.BBOX_OVERFLOW_WARNING:
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
        "image_names": image_paths
    }

def print_multi_columns(items, values=None, class_names=None, per_line=5):
    texts = []
    for cls in items:
        name = class_names[cls] if class_names and cls < len(class_names) else f"UNKNOWN_{cls}"
        if values:
            entry = f"{cls} {name} ({values[cls]:.2f}%)"
        else:
            entry = f"{cls} {name}"
        texts.append(entry)
    max_width = max(len(t) for t in texts) + 2
    for i in range(0, len(texts), per_line):
        print(" | ".join(f"{t:<{max_width}}" for t in texts[i:i+per_line]))

def afficher_stats_bbox(stats):

    label_width = 10
    col_width = 11

    def upper_line():
        print("┌" + "─" * label_width +
              "┬" + "─" * col_width +
              "┬" + "─" * col_width +
              "┬" + "─" * col_width +
              "┬" + "─" * col_width + "┐")

    def inter_line():
        print("├" + "─" * label_width +
              "┼" + "─" * col_width +
              "┼" + "─" * col_width +
              "┼" + "─" * col_width +
              "┼" + "─" * col_width + "┤")

    def down_line():
        print("└" + "─" * label_width +
              "┴" + "─" * col_width +
              "┴" + "─" * col_width +
              "┴" + "─" * col_width +
              "┴" + "─" * col_width + "┘")

    def ligne_header():
        print(f"|{'':<{label_width}}"
              f"|{'Mean':^{col_width}}"
              f"|{'Std':^{col_width}}"
              f"|{'Min':^{col_width}}"
              f"|{'Max':^{col_width}}|")

    def format_value(v, width):
        if v == 0:
            return f"{0:>{width}.4f}"
        elif abs(v) < 1e-2:
            return f"{v:>{width}.2e}"   # scientifique
        else:
            return f"{v:>{width}.4f}"   # normal

    def ligne_data(label, data):
        print(f"|{label:<{label_width}}"
            f"|{format_value(data['mean'], col_width)}"
            f"|{format_value(data['std'], col_width)}"
            f"|{format_value(data['min'], col_width)}"
            f"|{format_value(data['max'], col_width)}|")

    upper_line()
    ligne_header()
    inter_line()

    ligne_data("Width", stats["bbox_width"])
    ligne_data("Height", stats["bbox_height"])
    ligne_data("Area", stats["bbox_area"])

    down_line()

def verifier_classes_dataset(class_distribution, class_names):
    n_classes = len(class_names)
    classes_presentes = set(class_distribution.keys())
    classes_yaml = set(range(n_classes))
    inutilisees = classes_yaml - classes_presentes
    manquantes = classes_presentes - classes_yaml
    valides = classes_presentes & classes_yaml
    return {"inutilisees": inutilisees, "manquantes": manquantes, "valides": valides}

def afficher_tableau_croise_anomalies(resultats):
    anomalies = resultats.get("anomalies", [])
    images = sorted(set(resultats.get("image_names", [])))
    types_anomalies = [
        "bbox_trop_petite",
        "bbox_trop_grande",
        "bbox_surface_trop_petite",
        "bbox_surface_trop_grande",
        "bbox_hors_limite_warning",
        "bbox_hors_limite_error"
    ]

    # Création du dictionnaire croisé
    tableau = {img: {t: 0 for t in types_anomalies} for img in images}
    for a in anomalies:
        img = a["image"]
        t = a["type"]
        if t in types_anomalies:
            tableau[img][t] += 1

    # Affichage
    header = ["Image"] + [t.replace("bbox_", "") for t in types_anomalies]
    col_widths = [max(len(h), 12) for h in header]

    # Ligne d'entête
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Lignes par image
    for img in images:
        row = [img] + [str(tableau[img][t]) for t in types_anomalies]
        print(" | ".join(f"{c:<{w}}" for c, w in zip(row, col_widths)))


def afficher_dataset_statistics(resultats, path_user, class_names=None, classes_par_ligne=4, afficher_hist=False):

    display = dc.DisplayColor()

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]
    bbox_areas = resultats.get("bbox_areas", [])

    total_boxes = stats["bounding_boxes"]
    total = sum(class_distribution.values())
    total_classes = len(class_names) if class_names else max(class_distribution.keys()) + 1

    print("\n================ DATASET SUMMARY ================")
    print(f"{'Images':18}: {stats['images']}")
    print(f"{'Labels':18}: {stats['labels']}")
    print(f"{'Bounding boxes':18}: {total_boxes}")
    print(f"{'BBox / image':18}: {total_boxes / stats['images']:.2f}")
    print(f"{'Classes':18}: {total_classes}")

    # --- statistiques BBOX ---
    print("\n--------------- BBOX STATISTICS -----------------")
    afficher_stats_bbox(stats)

    # --- Vérification YAML ---
    missing_label  = False
    if class_names:
        verif = verifier_classes_dataset(class_distribution, class_names)
        # Classes inutilisées
        if verif["inutilisees"]:
            print("")
            display.print(f"Classes définies dans YAML mais jamais utilisées ({len(verif['inutilisees'])}) : ", colors["warning"])
            inutilisees = [f"{cls} {class_names[cls]}" for cls in sorted(verif["inutilisees"])]
            max_width = max(len(t) for t in inutilisees) + 2
            for i in range(0, len(inutilisees), 5):
                print(" | ".join(f"{entry:<{max_width}}" for entry in inutilisees[i:i+5]))
        # Classes manquantes
        if verif["manquantes"]:
            print("")
            missing_label  = True
            display.print(f"Classes présentes dans labels mais absentes du YAML ({len(verif['manquantes'])}) : ", colors["warning"])
            manquantes = [str(cls) for cls in sorted(verif["manquantes"])]
            max_width = max(len(t) for t in manquantes) + 2
            for i in range(0, len(manquantes), 5):
                print(" | ".join(f"{entry:<{max_width}}" for entry in manquantes[i:i+5]))

    # --- distribution par classe ---
    print("\n---------------- CLASS DISTRIBUTION -------------")
    items = sorted(class_distribution.items())
    ligne_texts = []
    for cls, count in items:
        pct = (count / total) * 100
        name = class_names[cls] if class_names and cls < len(class_names) else f"UNKNOWN_{cls}"
        bar = "█" * int(pct / 2)
        ligne_texts.append(f"{cls} {name} {count} ({pct:.2f}%) {bar}")
    max_width = max(len(t) for t in ligne_texts) + 2
    for i in range(0, len(ligne_texts), classes_par_ligne):
        print(" | ".join(f"{t:<{max_width}}" for t in ligne_texts[i:i+classes_par_ligne]))

    # --- classes sous-représentées ---------------------------
    if ct.SEUIL_PCT is not None:
        classes_faibles = []
        print()
        for cls, count in items:
            pct = (count / total) * 100
            if pct < ct.SEUIL_PCT:
                name = class_names[cls] if class_names and cls < len(class_names) else f"UNKNOWN_{cls}"
                classes_faibles.append((cls, name, count, pct))

        if not classes_faibles:
            display.print(f"Aucune classe sous {ct.SEUIL_PCT}% ", colors['ok'])
        else:
            message = f"------- CLASSES SOUS-REPRÉSENTÉES < {ct.SEUIL_PCT}% -------"
            display.print(message, colors['warning'])
            # tri optionnel (du pire au moins pire)
            classes_faibles.sort(key=lambda x: x[3])  # tri par %

            # --- regroupement par pourcentage ---
            grouped = defaultdict(list)

            for cls, name, count, pct in classes_faibles:
                key = round(pct, 2)  # regroupe par % arrondi
                grouped[key].append((cls, name, count))

            # tri par % croissant
            for pct in sorted(grouped.keys()):
                print(f"--- {pct:.2f}% ---")

                entries = grouped[pct]
                texts = [f"{cls} {name}" for cls, name, _ in entries]

                max_width = max(len(t) for t in texts) + 2
                for i in range(0, len(texts), ct.N_PER_LINE):
                    print(" | ".join(f"{t:<{max_width}}" for t in texts[i:i+ ct.N_PER_LINE]))
                print("")


    # --- histogramme bbox ---
    if afficher_hist and bbox_areas:
        gr.histograme(bbox_areas,
                      "Distribution des tailles de bounding boxes",
                      "Aire bbox",
                      "Nombre"
                      )

    # --- anomalies ---------------------------------------------------------------
    # regroupement anomalies par image et type
    anomaly_images = defaultdict(lambda: defaultdict(int))
    for a in anomalies:
        img = a["image"]
        typ = a["type"]
        anomaly_images[img][typ] += 1

    # ---- 
    print()
    display.print('-' * 120, colors['info'])
    if missing_label :
        display.print('Attention : des classes sont présentes dans les labels mais absentes du YAML !!', colors['error'])

    if not anomaly_images :
        display.print('Aucune anomalie trouvé !!', colors['ok'])
    else :
        # --- Types d'anomalies à afficher dans le tableau croisé
        types_anomalies = [
            "bbox_trop_petite",
            "bbox_surface_trop_petite",
            "bbox_trop_grande",
            "bbox_surface_trop_grande",
            "bbox_hors_limite_warning",
            "bbox_hors_limite_error"
        ]

        print()
        display.print("---------------- ANOMALIES ----------------------", colors['warning'])
        print("------ LEGENDES ------")
        print("1 : bbox_trop_petite        | 2 : bbox_trop_grande")
        print("3 : surface_trop_petite     | 4 : surface_trop_grande")
        print("5 : bbox_warning_hors_zone  | 6 : bbox_error_hors_zone")
        print()

        type_to_id = {
            "bbox_trop_petite": 1, "bbox_trop_grande": 2,
            "bbox_surface_trop_petite": 3, "bbox_surface_trop_grande": 4,
            "bbox_hors_limite_warning": 5, "bbox_hors_limite_error": 6,
        }

        id_to_type = {v: k for k, v in type_to_id.items()}

        # --- construction d'un dictionnaire comptant les anomalies par image et type ---
        anomaly_count_per_image = defaultdict(lambda: {t: 0 for t in types_anomalies})
        for a in anomalies:
            img = a["image"]
            t = a["type"]
            anomaly_count_per_image[img][t] += 1

        # --- initialisation des totaux ---
        totaux = {i: 0 for i in range(1, 7)}

        col_width = 5

        # header
        header = f"{'Image':25} | " + " | ".join(f"{i:^{col_width}}" for i in range(1, 7))
        print(header)
        print("─" * len(header))

        # lignes
        for img, anomalies_dict in sorted(anomaly_images.items()):
            line_parts = [f"{img:25}"]
            row_sum = 0

            for i in range(1, 7):
                t = id_to_type[i]
                count = anomalies_dict.get(t, 0)
                row_sum += count

                # accumulation des totaux
                totaux[i] += count

                if count > 0:
                    color = colors["error"] if i == 6 else colors["warning"]

                    r, g, b, _ = color
                    rgb = f"\033[38;2;{r};{g};{b}m"

                    cell = f"{count:^{col_width}}"
                    cell = f"{rgb}{cell}\033[0m"
                else:
                    cell = f"{0:^{col_width}}"

                line_parts.append(cell)


            # ajouter la colonne SUM
            line_parts.append(f"{row_sum:^{col_width}}")
            print(" | ".join(line_parts))      

        # --- ligne de séparation ---
        print("─" * len(header))

        # --- ligne TOTAL ---
        total_line = [f"{'TOTAL':25}"]

        for i in range(1, 7):
            total = totaux[i]

            if total > 0:
                color = colors["error"] if i == 6 else colors["warning"]
                r, g, b, _ = color
                rgb = f"\033[38;2;{r};{g};{b}m"

                cell = f"{total:^{col_width}}"
                cell = f"{rgb}{cell}\033[0m"
            else:
                cell = f"{0:^{col_width}}"

            total_line.append(cell)

        print(" | ".join(total_line))
    

        # --- pire images ----------------------------------------------
        # pondération des erreurs
        weights = {
            "bbox_trop_petite": 1,
            "bbox_surface_trop_petite": 1,
            "bbox_trop_grande": 2,
            "bbox_surface_trop_grande": 2,
            "bbox_hors_limite_warning": 3,
            "bbox_hors_limite_error": 5
        }

        bbox_per_image = defaultdict(int)
        for img in resultats.get("image_names", []):
            bbox_per_image[img] += 1

        score_images = defaultdict(lambda: {"count":0, "severity":0, "bbox_total":0, "score":0})
        for a in anomalies:
            img = a["image"]
            t = a["type"]
            score_images[img]["count"] += 1
            score_images[img]["severity"] += weights.get(t,0)
        for img, total_bbox in bbox_per_image.items():
            score_images[img]["bbox_total"] = total_bbox
        for img, data in score_images.items():
            if data["bbox_total"] == 0:
                continue
            error_ratio = data["count"] / data["bbox_total"]
            avg_severity = data["severity"] / data["count"] if data["count"]>0 else 0
            data["score"] = error_ratio * avg_severity

        worst_images = sorted(score_images.items(), key=lambda x:x[1]["score"], reverse=True)
        if worst_images:
            print("\n-------------------------- QUALITE DES IMAGES -----------------------")
            for img,d in worst_images[:10]:
                if d["score"] == 0:
                    continue
                ratio = (d['count'] / d['bbox_total']) *100
                print(   
                    f"score = {d['score']:.2f} : "
                    f"anomalies = {d['count']:<3} "
                    f"| Nb bbox = {d['bbox_total']:<3} | ration {ratio:6.2f}% : "
                    f"{img:<25}"
                )
            print("---------------------------------------------------------------------")
        images_problematiques = sum(1 for d in score_images.values() if d["count"] > 0)
        total_bboxes_problematiques = len(anomalies)
        pct_images = (images_problematiques / stats["images"]) * 100 if stats["images"] else 0
        dataset_score = (sum(d["score"] for d in score_images.values()) / len(score_images)) if score_images else 0
        
        text = f"Nombre d'images avec au moins une bbox problématique : {images_problematiques} ({pct_images:.2f}%)" 
        display.print(text, colors['warning'])
        
        text = f"Total de bboxes problématiques : {total_bboxes_problematiques}"
        display.print(text, colors['warning'])
        
        print(f"Score moyen du dataset : {dataset_score:.3f}\n")
        

    # --- histogramme anomalies par type ---
    if afficher_hist : 
        type_counts = Counter(a["type"] for a in anomalies)
        if type_counts:
            gr.histo_multipl(type_counts,
                            "Nombre",
                            anomalies) 

    # anomalies = resultats['anomalies'] après dataset_statistics_yolo
    if ct.REPORT_MODE :
        util.save_anomalies_readable(anomalies, "erreurs_dataset.txt", path_user)

    metrics = imbalance_metrics(class_distribution)
    afficher_imbalance_avance(metrics, display, colors)

    display.print('-' * 120, colors['info'])

