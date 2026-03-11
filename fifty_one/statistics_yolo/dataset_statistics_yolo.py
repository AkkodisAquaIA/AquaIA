import os
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.dummy import class_distribution
import yaml


import tools.display_color as dc
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors


#==========================================================================================
def dataset_statistics_yolo(DATASET_DIR):

    labels_dir = os.path.join(DATASET_DIR, "labels", "train2017")
    images_dir = os.path.join(DATASET_DIR, "images", "train2017")

    split_pattern = re.compile(r"[,\s]+")

    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    classes = []
    image_paths = []  # stocke le nom de l'image pour chaque bbox

    total_boxes = 0

    # --- lecture labels ---
    for entry in os.scandir(labels_dir):
        if not entry.name.endswith(".txt"):
            continue

        # nom du fichier image correspondant (sans le chemin complet)
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
                image_paths.append(image_name)  # on stocke juste le nom de l'image

                total_boxes += 1

    image_count = len(list(Path(images_dir).rglob("*.*")))
    label_count = len(list(Path(labels_dir).rglob("*.txt")))

    # --- statistiques générales ---
    stats = {
        "images": image_count,
        "labels": label_count,
        "bounding_boxes": total_boxes,
        "bbox_width_mean": float(np.mean(bbox_widths)),
        "bbox_height_mean": float(np.mean(bbox_heights)),
        "bbox_area_mean": float(np.mean(bbox_areas)),
        "bbox_width_min": float(np.min(bbox_widths)),
        "bbox_width_max": float(np.max(bbox_widths)),
        "bbox_height_min": float(np.min(bbox_heights)),
        "bbox_height_max": float(np.max(bbox_heights)),
    }

    # --- distribution des classes ---
    class_distribution = Counter(classes)

    # --- détection annotations aberrantes ---
    anomalies = []

    for img_name, w, h in zip(image_paths, bbox_widths, bbox_heights):
        area = w * h

        if area < ct.MIN_BBOX_AREA:
            anomalies.append({
                "type": "bbox_trop_petite",
                "area": area,
                "image": img_name
            })

        if area > ct.MAX_BBOX_AREA:
            anomalies.append({
                "type": "bbox_trop_grande",
                "area": area,
                "image": img_name
            })

    # --- histogramme bbox ---
    plt.figure()
    plt.hist(bbox_areas, bins=50)
    plt.title("Distribution des tailles de bounding boxes")
    plt.xlabel("aire bbox")
    plt.ylabel("nombre")
    plt.show()

    return {
        "stats": stats,
        "class_distribution": class_distribution,
        "anomalies": anomalies,
        "bbox_areas": bbox_areas
    }


def load_class_names(dataset_yaml_path):
    """Charge les noms de classes depuis dataset.yaml"""
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    return names

def verifier_classes_dataset(class_distribution, class_names):
    """Vérifie classes présentes vs classes définies dans YAML"""
    n_classes = len(class_names)
    classes_presentes = set(class_distribution.keys())
    classes_yaml = set(range(n_classes))

    inutilisees = classes_yaml - classes_presentes
    manquantes = classes_presentes - classes_yaml
    valides = classes_presentes & classes_yaml

    return {
        "inutilisees": inutilisees,
        "manquantes": manquantes,
        "valides": valides
    }


# --- Fonction utilitaire multi-colonnes dynamique ---
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

def afficher_dataset_statistics(resultats, class_names=None, classes_par_ligne=3, afficher_hist=True):
    """
    Affiche un résumé complet du dataset YOLO avec analyses avancées.
    
    resultats       : dict retourné par dataset_statistics_yolo()
    class_names     : liste des noms de classes
    classes_par_ligne: nb de classes par ligne pour affichage
    afficher_hist   : True pour afficher histogrammes
    """

    display = dc.DisplayColor()

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]
    bbox_areas = resultats.get("bbox_areas", [])

    total_boxes = stats["bounding_boxes"]
    total_classes = len(class_names) if class_names else max(class_distribution.keys()) + 1
    total = sum(class_distribution.values())

    print("\n================ DATASET SUMMARY ================\n")
    print(f"{'Images':18}: {stats['images']}")
    print(f"{'Labels':18}: {stats['labels']}")
    print(f"{'Bounding boxes':18}: {total_boxes}")
    print(f"{'BBox / image':18}: {total_boxes / stats['images']:.2f}")
    print(f"{'Classes':18}: {total_classes}")

    print("\n--------------- BBOX STATISTICS -----------------\n")
    print(f"{'Width mean':18}: {stats['bbox_width_mean']:.4f}")
    print(f"{'Height mean':18}: {stats['bbox_height_mean']:.4f}")
    print(f"{'Area mean':18}: {stats['bbox_area_mean']:.4f}")
    print(f"{'Width min':18}: {stats['bbox_width_min']:.4f}")
    print(f"{'Width max':18}: {stats['bbox_width_max']:.4f}")
    print(f"{'Height min':18}: {stats['bbox_height_min']:.4f}")
    print(f"{'Height max':18}: {stats['bbox_height_max']:.4f}")

    print("\n---------------- CLASS DISTRIBUTION -------------")

    # --- Vérification YAML ---
    if class_names:
        verif = verifier_classes_dataset(class_distribution, class_names)

        # Classes inutilisées
        if verif["inutilisees"]:
            display.print(f"Classes définies dans YAML mais jamais utilisées {len(verif['inutilisees'])} : ", colors["warning"])
            inutilisees = [f"{cls} {class_names[cls]}" for cls in sorted(verif["inutilisees"])]
            max_width = max(len(t) for t in inutilisees) + 2
            for i in range(0, len(inutilisees), 5):
                print(" | ".join(f"{entry:<{max_width}}" for entry in inutilisees[i:i+5]))

        # Classes présentes dans labels mais absentes du YAML
        if verif["manquantes"]:
            print("")
            display.print(f"Classes présentes dans labels mais absentes du YAML {len(verif['manquantes'])} : ", colors["error"])
            manquantes = [str(cls) for cls in sorted(verif["manquantes"])]
            max_width = max(len(t) for t in manquantes) + 2
            for i in range(0, len(manquantes), 5):
                print(" | ".join(f"{entry:<{max_width}}" for entry in manquantes[i:i+5]))

    # --- Distribution des classes ---
    print("\nDistribution des classes (ID, nom, count, %, barre):")
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


    # --- Classes rares ---
    classes_rares = {cls: (count/total)*100 for cls,count in class_distribution.items() if (count/total)*100 < 1}
    if classes_rares:
        print("")
        display.print(f"Classes rares (<1% des annotations) {len(classes_rares)} :", colors["warning"])
        # Trier les IDs de classes rares
        classes_rares_sorted = sorted(classes_rares.keys())
        print_multi_columns(classes_rares_sorted, values=classes_rares, class_names=class_names, per_line=5)

    # --- Dataset déséquilibré (>60%) ---
    if items:
        max_cls, max_count = max(items, key=lambda x:x[1])
        pct_max = (max_count/total)*100
        if pct_max > 20:
            print("")
            name = class_names[max_cls] if class_names and max_cls < len(class_names) else f"UNKNOWN_{max_cls}"
            display.print(f"Dataset très déséquilibré :", colors["warning"])
            print(f" classe {max_cls} '{name}' domine ({pct_max:.1f}%)")
          
    # --- Anomalies ---
    print("\n---------------- ANOMALIES ----------------------")
 
    # dictionnaires de comptage
    grandes_par_image = defaultdict(int)
    petites_par_image = defaultdict(int)

    total_grandes = 0
    total_petites = 0

    for a in anomalies:

        if a["type"] == "bbox_trop_grande":
            grandes_par_image[a["image"]] += 1
            total_grandes += 1

        elif a["type"] == "bbox_trop_petite":
            petites_par_image[a["image"]] += 1
            total_petites += 1


    # --- zones trop grandes ---
    if grandes_par_image:

        display.print(
            f"Zones trop grandes : {total_grandes} bbox sur {len(grandes_par_image)} images",
            colors["warning"]
        )

        top_grandes = sorted(grandes_par_image.items(), key=lambda x: x[1], reverse=True)

        for img, count in top_grandes[:10]:
            print(f"{img:<25} : {count} bbox trop grandes")

        print()


    # --- zones trop petites ---
    if petites_par_image:

        display.print(
            f"Zones trop petites : {total_petites} bbox sur {len(petites_par_image)} images",
            colors["warning"]
        )

        top_petites = sorted(petites_par_image.items(), key=lambda x: x[1], reverse=True)

        for img, count in top_petites[:10]:
            print(f"{img:<25} : {count} bbox trop petites")


    #---- pire image (bbox aberrantes) ---
    print("\n---------------- QUALITE DES IMAGES -------------")

    score_images = defaultdict(int)

    # pénalité anomalies bbox
    for a in anomalies:

        if a["type"] == "bbox_trop_petite":
            score_images[a["image"]] += 1

        elif a["type"] == "bbox_trop_grande":
            score_images[a["image"]] += 2


    # pénalité classes rares
    if classes_rares:

        for cls in classes_rares:

            for img_name, img_cls in zip(resultats.get("image_names", []), resultats.get("classes", [])):
                if img_cls == cls:
                    score_images[img_name] += 1

    if score_images:

        display.print("Images les plus problématiques :", colors["warning"])

        worst_images = sorted(score_images.items(), key=lambda x: x[1], reverse=True)

        for img, score in worst_images[:10]:
            print(f"{img:<25} score = {score}")

    images_problematiques = len(score_images)
    pct = (images_problematiques / stats["images"]) * 100

    print(f"\nImages problématiques : {images_problematiques} ({pct:.2f}%)")



    # --- Histogramme des aires de bbox ---
    if afficher_hist:
        plt.figure()
        plt.hist(bbox_areas if bbox_areas else [stats["bbox_area_mean"]], bins=50)
        plt.title("Distribution des tailles de bounding boxes")
        plt.xlabel("Aire bbox")
        plt.ylabel("Nombre")
        plt.show()

    print("\n=================================================\n")
