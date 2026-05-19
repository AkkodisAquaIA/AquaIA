import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import yaml
from colorama import Fore, Style, init
from collections import defaultdict

from statistics_yolo import imbalance as im

import tools.display_color as dc
from tools import utility as util
from config import constants as ct
from config import process as pr
from config.constants import DISPLAY_COLORS as colors
from tools import graphe as gr

init(autoreset=True)

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

#==========================================================================================

def afficher_stats_bbox(stats, cfg):

    def format_value(v):
        if v == 0:
            return "0.0000"
        elif abs(v) < 1e-2:
            return f"{v:.3e}"
        else:
            return f"{v:.4f}"


    def format_const(min_c, max_c):
        return f"{min_c:.3e} → {max_c:.4f}"


    columns = [
        {"title": ""                   , "width": 10, "align": "<"},
        {"title": "Mean"               , "width": 11, "align": ">"},
        {"title": "Std"                , "width": 11, "align": ">"},
        {"title": "Min"                , "width": 11, "align": ">"},
        {"title": "Max"                , "width": 11, "align": ">"},
        {"title": "Const (min → max)"  , "width": 23, "align": "^"},
    ]

    table = util.TablePrinter(columns)
    table.header()

    def add_row(label, data, cmin, cmax):

        table.row([
            (label, True),
            (format_value(data["mean"]), True),
            (format_value(data["std"]), True),
            (format_value(data["min"]), data["min"] >= cmin),
            (format_value(data["max"]), data["max"] <= cmax),
            (format_const(cmin, cmax), True),
        ])

    add_row("Width",  stats["bbox_width"],  cfg["MIN_BBOX"],      cfg["MAX_BBOX"])
    add_row("Height", stats["bbox_height"], cfg["MIN_BBOX"],      cfg["MAX_BBOX"])
    add_row("Area",   stats["bbox_area"],   cfg["MIN_BBOX_AREA"], cfg["MAX_BBOX_AREA"])

    table.footer()


#==========================================================================================================
def verifier_classes_dataset(class_distribution, class_names):
    n_classes = len(class_names)
    classes_presentes = set(class_distribution.keys())
    classes_yaml = set(range(n_classes))
    inutilisees = classes_yaml - classes_presentes
    manquantes = classes_presentes - classes_yaml
    valides = classes_presentes & classes_yaml
    return {"inutilisees": inutilisees, "manquantes": manquantes, "valides": valides}

def afficher_dataset_statistics(resultats, cfg, path_user, class_names=None, classes_par_ligne=4, afficher_hist=False):

    display = dc.DisplayColor()

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]
    bbox_areas = resultats.get("bbox_areas", [])
    class_to_images = resultats.get("class_to_images", {})

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
    afficher_stats_bbox(stats, cfg)

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
                print(" │ ".join(f"{entry:<{max_width}}" for entry in inutilisees[i:i+5]))
        # Classes manquantes
        if verif["manquantes"]:
            print("")
            missing_label  = True
            display.print(f"Classes présentes dans labels mais absentes du YAML ({len(verif['manquantes'])}) : ", colors["warning"])
            manquantes = [str(cls) for cls in sorted(verif["manquantes"])]
            max_width = max(len(t) for t in manquantes) + 2
            for i in range(0, len(manquantes), 5):
                print(" │ ".join(f"{entry:<{max_width}}" for entry in manquantes[i:i+5]))

    # --- distribution par classe ---
    items = sorted(class_distribution.items())
   
    max_name_len = max(
        len(class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}")
        for cls, _ in items
    )
    
    blocs = []
    dom, moy, rary = 0, 0, 0
    
    for cls, count in items:
        pct = (count / total) * 100
        name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"

        # Couleur automatique selon importance
        if pct < cfg["RARE"] :
            color = Fore.RED
            rary +=1
        elif pct < cfg["DOMINANT"] :
            color = Fore.YELLOW
            moy +=1
        else:
            color = Fore.GREEN
            dom +=1

        BAR_WIDTH = 20 # largeur maximale de la barre pour 100% (ajustable)
        bbare = util.draw_bar(pct, 0, 100, BAR_WIDTH)
  
        bloc = (
            f"{color}"
            f"{cls:>2} "
            f"{name:<{max_name_len}} "
            f'{count:5} '
            f"{pct:5.2f}% "
            f"{bbare}"
            f"{Style.RESET_ALL}"
        )

        blocs.append(bloc)

    print()
    display.print(f"----------- CLASS DISTRIBUTION ({rary + moy + dom}) -----------", colors['info'])
    legend_colored = (
        f'    {Fore.GREEN}■ ({dom}) ≥ {cfg["DOMINANT"]}% Dominant{Style.RESET_ALL}   '
        f'│ {Fore.YELLOW}■ ({moy}) {cfg["RARE"]}–{cfg["DOMINANT"]}% Moyen{Style.RESET_ALL}   '
        f'│ {Fore.RED}■ ({rary}) < {cfg["RARE"]}% Rare{Style.RESET_ALL}'
    )
    print(f"{legend_colored}\n")

    # Largeur terminal
    term_width = 220 # shutil.get_terminal_size().columns (317)
    bloc_width = max(len(b) for b in blocs) + 1
    classes_par_ligne = max(1, term_width // bloc_width)

    for i in range(0, len(blocs), classes_par_ligne):
        ligne = blocs[i:i + classes_par_ligne]
        print("│ ".join(f"{b:<{bloc_width}}" for b in ligne))

    # Affichage de l'histogramme de distribution des classes
    gr.histogram_classe(items, class_names, cfg, total )

    # --- classes Rares ---------------------------------------
    if cfg["RARE"] is not None:
        classes_faibles = []
        print()
        for cls, count in items:
            pct = (count / total) * 100
            if pct < cfg["RARE"]:
                name = class_names[cls] if class_names and cls < len(class_names) else f"UNKNOWN_{cls}"
                classes_faibles.append((cls, name, count, pct))

        if not classes_faibles:
            display.print(f'Aucune classe sous {cfg["RARE"]}% ', colors['ok'])
        else:
            message = f'------- CLASSES RARES ({rary}) < {cfg["RARE"]}% -------'
            display.print(message, colors['error'])
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
                    print(" │ ".join(f"{t:<{max_width}}" for t in texts[i:i+ ct.N_PER_LINE]))
                print("")



# --- IMAGES PAR CLASSE --------------------------------------------------------
    display.print("------------- IMAGES PAR CLASSE -------------", colors['info'])

    MAX_IMAGES_DISPLAY = 30     # nombre max d'images par classe
    MAX_CLASSES_SELECT = 6      # max classes que l'utilisateur peut demander

    def parse_selection(user_input, available_classes):
        """
        Parse une entrée du type:
        1 3-5 8
        Retourne une liste d'entiers uniques et valides.
        """
        result = set()
        parts = user_input.split()

        for part in parts:
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    for i in range(start, end + 1):
                        if i in available_classes:
                            result.add(i)
                except ValueError:
                    continue
            else:
                try:
                    num = int(part)
                    if num in available_classes:
                        result.add(num)
                except ValueError:
                    continue

        return sorted(result)

    # available_classes = sorted(class_to_images.keys())
    available_classes = sorted(
        class_to_images.keys(),
        key=lambda cls: (-len(class_to_images[cls]), cls)
    )

    # Compute max lengths dynamically
    max_name_length = max(
        len(class_names[cls]) if class_names and cls < len(class_names) else len(f"UNK_{cls}")
        for cls in available_classes
    )

    max_count_length = max(
        len(str(len(class_to_images[cls]))) 
        for cls in available_classes
    )

    rows = []
    for cls in available_classes:
        name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"
        count = len(class_to_images[cls])
        formatted = f"{cls:>3} - {name:<{max_name_length}} : {count:>{max_count_length}}"
        rows.append(formatted)

    COLUMNS = 5
    col_width = max(len(r) for r in rows) + 0

    for i in range(0, len(rows), COLUMNS):
        line = rows[i:i + COLUMNS]
        print("│ ".join(f"{item:<{col_width}}" for item in line))

    print()

    # Affichage des noms des images par classe
    tag = f"Affichage des noms des images par classe"
    display.print(tag, colors['info'])
    while True:
        tag = f"Entrez jusqu'à {MAX_CLASSES_SELECT} classes (ex: 1 3-5 8) ou 'Return' pour quitter : "
        display.print(tag, colors['input']) 
        user_input = input("  > ").strip()

        if user_input.lower() == '':
            break

        selected_classes = parse_selection(user_input, available_classes)            

        if not selected_classes:
            print()
            display.print("Aucune classe valide sélectionnée.\n", colors['warning'])
            continue

        if len(selected_classes) > MAX_CLASSES_SELECT:
            print()
            display.print(f"Maximum {MAX_CLASSES_SELECT} classes autorisées.\n", colors['warning'])
            continue    

        for cls in selected_classes:

            name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"
            all_images = sorted(class_to_images[cls])
            images = all_images[:MAX_IMAGES_DISPLAY]

            print(f"\n{cls:>2} {name}  ({len(all_images)} images)")

            if images:
                max_width = max(len(img) for img in images) + 1
                for i in range(0, len(images), 5):
                    ligne = images[i:i+5]
                    print(" │ ".join(f"{img:<{max_width}}" for img in ligne))
                    
            if len(all_images) > MAX_IMAGES_DISPLAY:
                display.print(f"... + {len(all_images) - MAX_IMAGES_DISPLAY} autres images", colors['warning'])
        
        print()

    # --- histogramme bbox ---
    if afficher_hist and bbox_areas:
        gr.histogram_taille_bbox(bbox_areas,
                      "Distribution des tailles de BBox",
                      "Aire bbox",
                      "Nombre",
                      cfg,
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
        display.print('Attention : des classes sont présentes dans les labels mais absentes du YAML !!', colors['warning'])

    if not anomaly_images :
        print()
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
        print("1 : bbox_trop_petite        │ 2 : bbox_trop_grande")
        print("3 : surface_trop_petite     │ 4 : surface_trop_grande")
        print("5 : bbox_warning_hors_zone  │ 6 : bbox_error_hors_zone")
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
        header = f"{'Image':25} │ " + " │ ".join(f"{i:^{col_width}}" for i in range(1, 7))
        print(header)

        line = (len(header) + 7)
        print("─" * line )

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
            print(" │ ".join(line_parts))      

        # --- ligne de séparation ---
        print("─" * line)

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

        print(" │ ".join(total_line))
    

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

        max_weight = max(weights.values())

        bbox_per_image = defaultdict(int)
        for img in resultats.get("image_names", []):
            bbox_per_image[img] += 1

        score_images = defaultdict(lambda: {"count":0, "severity":0, "bbox_total":0, "score":0.0})

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

            if data["count"] > 0:
                avg_severity = data["severity"] / data["count"]
                normalized_severity = avg_severity / max_weight
            else:
                normalized_severity = 0

            data["score"] = error_ratio * normalized_severity


        worst_images = sorted(score_images.items(), key=lambda x:x[1]["score"], reverse=True)

        
        if worst_images:
            print("\n-------------------------- QUALITE DES IMAGES -----------------------")
            
            # Filter first
            valid_images = [
                (img, d) for img, d in worst_images
                if d["score"] != 0
            ]

            n = len(valid_images)
            max_n = ct.MAX_WORST_IMAGES

            if n == 1:
                texte = "La pire image"
            else:
                if n > max_n:
                    texte = f"Les {max_n} images les plus mauvaises"
                else:
                    texte = f"Les {n} pires images"

            print(f"\n-------------------------- {texte} ----------------------")
            
            # Then limit to 10
            for img, d in valid_images[:ct.MAX_WORST_IMAGES]:
                ratio = (d['count'] / d['bbox_total']) * 100

                print(
                    f"score = {d['score']:.2f} : "
                    f"anomalies = {d['count']:<3} "
                    f"│ Nb bbox = {d['bbox_total']:<3} │ ratio {ratio:6.2f}% : "
                    f"{img:<25}"
                )

            print("---------------------------------------------------------------------")
        images_problematiques = sum(1 for d in score_images.values() if d["count"] > 0)
        total_bboxes_problematiques = len(anomalies)
        pct_images = (images_problematiques / stats["images"]) * 100 if stats["images"] else 0
        dataset_score = (sum(d["score"] for d in score_images.values()) / len(score_images)) if score_images else 0
        
        text = f"Nombre d'images avec au moins une bbox problématique : {images_problematiques} ({pct_images:.3f}%)" 
        display.print(text, colors['warning'])
        
        text = f"Total de bboxes problématiques : {total_bboxes_problematiques}"
        display.print(text, colors['warning'])
        
        print(f"Score moyen du dataset : {dataset_score:.3f}\n")
        

        # --- histogramme anomalies par type ---
        if afficher_hist : 
            type_counts = Counter(a["type"] for a in anomalies)
            if type_counts:
                gr.histogram_anomalies(type_counts,
                                "Nombre",
                                cfg,
                                anomalies,
                    ) 

        anomalies = resultats['anomalies'] 

        if cfg["REPORT_MODE"] :
            util.save_anomalies_readable(anomalies, "erreurs_dataset.txt", path_user)


    # --- AFFICHAGE DES METRIQUES D'IMBALANCE ---
    metrics = im.imbalance_metrics(class_distribution, cfg)
    im.afficher_imbalance_avance(metrics, display, colors, cfg)
  
    print()
    display.print('-' * 120, colors['info'])

    return anomalies