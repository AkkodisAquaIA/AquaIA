import os
import re
from pathlib import Path
from collections import Counter, defaultdict

from tools import utility as util
import tools.display_color as dc
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors


#-----------------------------------------------------------------------------------
def check_bbox_overflow(x, y, w, h):
    """
    Vérifie si une bbox dépasse légèrement l'image.
    Retourne :
        "ok"    → bbox entièrement dans tolérance warning
        "warning" → débordement < BBOX_OVERFLOW_ERROR
        "error" → débordement > BBOX_OVERFLOW_ERROR
    """
    # convertir % → fraction
    warn_tol = ct.BBOX_OVERFLOW_WARNING / 100
    err_tol  = ct.BBOX_OVERFLOW_ERROR / 100

    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2

    # débordement réel
    overflow = max(-xmin, xmax-1, -ymin, ymax-1, 0)

    if overflow <= warn_tol:
        return "ok"
    elif overflow <= err_tol:
        return "warning"
    else:
        return "error"




#-----------------------------------------------------------------------------------

display = dc.DisplayColor()


# def validate_yolo_dataset_detailed(DATASET_DIR):
#     """
#     Validation détaillée d'un dataset YOLO.
#     Retourne un dictionnaire d'erreurs par type et un rapport complet par fichier/ligne.
#     """

#     labels_dir = os.path.join(DATASET_DIR, "labels", "train2017")
#     images_dir = os.path.join(DATASET_DIR, "images", "train2017")
#     split_pattern = re.compile(r"[,\s]+")

#     erreurs = defaultdict(list)  # erreurs par type
#     rapport_detail = defaultdict(lambda: defaultdict(list))  # rapport[fichier][ligne] = [erreurs]

#     Ctrl_ok = True
#     label_stems = set()

#     for entry in os.scandir(labels_dir):
#         if not entry.name.lower().endswith(".txt"):
#             continue
#         label_stems.add(Path(entry.name).stem)
#         path = entry.path
#         has_content = False
#         seen_boxes = set()

#         with open(path, "r", encoding="utf-8") as f:
#             for i, line in enumerate(f, start=1):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 has_content = True
#                 parts = split_pattern.split(line)

#                 erreurs_ligne = []

#                 # valeurs par défaut
#                 cls, x, y, w, h = -1, -1, -1, 0, 0

#                 # --- Colonnes ---
#                 if len(parts) != 5:
#                     erreurs["colonnes_incorrectes"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("colonnes_incorrectes")
#                     Ctrl_ok = False
#                 else:
#                     # --- Classe ---
#                     try:
#                         cls = int(parts[0])
#                         if ct.nb_classes is not None and cls >= ct. nb_classes:
#                             erreurs["classe_hors_plage"].append(f"{entry.name} (ligne {i})")
#                             erreurs_ligne.append("classe_hors_plage")
#                             Ctrl_ok = False
#                     except ValueError:
#                         erreurs["classe_invalide"].append(f"{entry.name} (ligne {i})")
#                         erreurs_ligne.append("classe_invalide")
#                         Ctrl_ok = False

#                     # --- Conversion bbox ---
#                     try:
#                         x, y, w, h = map(float, parts[1:])
#                     except ValueError:
#                         erreurs["valeurs_non_numeriques"].append(f"{entry.name} (ligne {i})")
#                         erreurs_ligne.append("valeurs_non_numeriques")
#                         Ctrl_ok = False

#                 # --- Tests bbox ---
#                 if x < 0 or y < 0:
#                     erreurs["coord_negatives"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("coord_negatives")
#                     Ctrl_ok = False
#                 if w <= 0 or h <= 0:
#                     erreurs["taille_non_positive"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("taille_non_positive")
#                     Ctrl_ok = False

#                 if cls < 0:
#                     erreurs["classe_negative"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("classe_negative")
#                     Ctrl_ok = False
                    
#                 if w < ct.MIN_BBOX or h < ct.MIN_BBOX:
#                     erreurs["bbox_trop_petites"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("bbox_trop_petites")
#                     Ctrl_ok = False
#                 if w > ct.MAX_BBOX or h > ct.MAX_BBOX:
#                     erreurs["bbox_trop_grandes"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("bbox_trop_grandes")
#                     Ctrl_ok = False
#                 if x > 1 or y > 1 or w > 1 or h > 1:
#                     erreurs["hors_limites"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("hors_limites")
#                     Ctrl_ok = False

#                 if w * h < ct.MIN_BBOX_AREA:
#                     erreurs["bbox_surface_trop_petite"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("bbox_surface_trop_petite")
#                     # Ctrl_ok = False

#                 status = check_bbox_overflow(x, y, w, h)

#                 if status == "warning":
#                     erreurs["bbox_sort_image_warning"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("bbox_sort_image_warning")

#                 elif status == "error":
#                     erreurs["bbox_sort_image"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("bbox_sort_image")
#                     Ctrl_ok = False

#                 # --- Dupliquées ---
#                 bbox = (cls, x, y, w, h)
#                 if bbox in seen_boxes:
#                     erreurs["bbox_dupliquees"].append(f"{entry.name} (ligne {i})")
#                     erreurs_ligne.append("bbox_dupliquees")
#                 seen_boxes.add(bbox)

#                 if erreurs_ligne:
#                     rapport_detail[entry.name][i].extend(erreurs_ligne)

#         if not has_content:
#             erreurs["labels_vides"].append(entry.name)
#             rapport_detail[entry.name][0].append("labels_vides")
#             Ctrl_ok = False

#     # --- Analyse images ---
#     image_paths = [p for p in Path(images_dir).rglob("*") if p.suffix.lower() in ct.IMAGE_EXT]
#     image_stems = {p.stem for p in image_paths}

#     # labels orphelins
#     orphan_labels = sorted(label_stems - image_stems)
#     if orphan_labels:
#         erreurs["labels_orphelins"] = orphan_labels
#         # Ctrl_ok = False
#         for lbl in orphan_labels:
#             rapport_detail[lbl][0].append("labels_orphelins")

#     # images sans label
#     images_sans_label = sorted(image_stems - label_stems)
#     if images_sans_label:
#         erreurs["images_sans_label"] = images_sans_label
#         Ctrl_ok = False

#     # images dupliquées
#     image_names = [p.name for p in image_paths]
#     duplicates = [k for k, v in Counter(image_names).items() if v > 1]
#     if duplicates:
#         erreurs["images_dupliquees"] = duplicates

#     # --- Affichage des erreurs par type ---
#     for key, values in erreurs.items():
#         if values:
#             util.display_and_save_errors(
#                 sorted(values),
#                 f"{key}.txt",
#                 key.replace("_", " ").capitalize()
#             )

#     # # --- Rapport détaillé ligne par ligne ---
#     # # Exemple d'affichage console, vous pouvez le sauvegarder dans un fichier CSV ou JSON
#     # print("\n--- Rapport détaillé des erreurs par fichier/ligne ---")
#     # for fichier, lignes in rapport_detail.items():
#     #     print(f"\nFichier: {fichier}")
#     #     for ligne, err_list in lignes.items():
#     #         print(f"  Ligne {ligne}: {', '.join(err_list)}")

#     return erreurs, rapport_detail, Ctrl_ok

def validate_yolo_dataset_detailed(DATASET_DIR):
    """
    Lecture détaillée d'un dataset YOLO.
    Retourne :
        - erreurs_syntaxe : dict par type d'erreur (colonnes, classe, coord)
        - all_bboxes     : liste de tuples (cls, x, y, w, h, image_name)
        - rapport_detail : dict[fichier][ligne] -> liste d'erreurs
        - Ctrl_ok        : bool global (True si pas d'erreurs syntaxiques graves)
    """

    labels_dir = os.path.join(DATASET_DIR, "labels", "train2017")
    images_dir = os.path.join(DATASET_DIR, "images", "train2017")
    split_pattern = re.compile(r"[,\s]+")
    
    erreurs_syntaxe = defaultdict(list)
    rapport_detail = defaultdict(lambda: defaultdict(list))
    Ctrl_ok = True
    label_stems = set()
    all_bboxes = []

    for entry in os.scandir(labels_dir):
        if not entry.name.lower().endswith(".txt"):
            continue
        label_stems.add(Path(entry.name).stem)
        path = entry.path
        has_content = False
        seen_boxes = set()

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                has_content = True
                parts = split_pattern.split(line)
                erreurs_ligne = []

                # valeurs par défaut
                cls, x, y, w, h = -1, -1, -1, 0, 0

                # --- Colonnes ---
                if len(parts) != 5:
                    erreurs_syntaxe["colonnes_incorrectes"].append(f"{entry.name} (ligne {i})")
                    erreurs_ligne.append("colonnes_incorrectes")
                    Ctrl_ok = False
                    continue  # on ne peut pas continuer sur cette ligne
                else:
                    # --- Classe ---
                    try:
                        cls = int(parts[0])
                        if ct.nb_classes is not None and cls >= ct.nb_classes:
                            erreurs_syntaxe["classe_hors_plage"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("classe_hors_plage")
                            Ctrl_ok = False
                    except ValueError:
                        erreurs_syntaxe["classe_invalide"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("classe_invalide")
                        Ctrl_ok = False

                    # --- Conversion bbox ---
                    try:
                        x, y, w, h = map(float, parts[1:])
                    except ValueError:
                        erreurs_syntaxe["valeurs_non_numeriques"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("valeurs_non_numeriques")
                        Ctrl_ok = False

                # --- Coordonnées négatives ou taille nulle ---
                if x < 0 or y < 0:
                    erreurs_syntaxe["coord_negatives"].append(f"{entry.name} (ligne {i})")
                    erreurs_ligne.append("coord_negatives")
                    Ctrl_ok = False
                if w <= 0 or h <= 0:
                    erreurs_syntaxe["taille_non_positive"].append(f"{entry.name} (ligne {i})")
                    erreurs_ligne.append("taille_non_positive")
                    Ctrl_ok = False
                if cls < 0:
                    erreurs_syntaxe["classe_negative"].append(f"{entry.name} (ligne {i})")
                    erreurs_ligne.append("classe_negative")
                    Ctrl_ok = False

                # --- BBox dupliquées ---
                bbox_tuple = (cls, x, y, w, h)
                if bbox_tuple in seen_boxes:
                    erreurs_syntaxe["bbox_dupliquees"].append(f"{entry.name} (ligne {i})")
                    erreurs_ligne.append("bbox_dupliquees")
                seen_boxes.add(bbox_tuple)

                # --- Collecte de toutes les bboxes pour statistiques ---
                all_bboxes.append((cls, x, y, w, h, entry.name))

                if erreurs_ligne:
                    rapport_detail[entry.name][i].extend(erreurs_ligne)

        if not has_content:
            erreurs_syntaxe["labels_vides"].append(entry.name)
            rapport_detail[entry.name][0].append("labels_vides")
            Ctrl_ok = False

    # --- Analyse images ---
    image_paths = [p for p in Path(images_dir).rglob("*") if p.suffix.lower() in ct.IMAGE_EXT]
    image_stems = {p.stem for p in image_paths}

    # labels orphelins
    orphan_labels = sorted(label_stems - image_stems)
    if orphan_labels:
        erreurs_syntaxe["labels_orphelins"] = orphan_labels
        for lbl in orphan_labels:
            rapport_detail[lbl][0].append("labels_orphelins")

    # images sans label
    images_sans_label = sorted(image_stems - label_stems)
    if images_sans_label:
        erreurs_syntaxe["images_sans_label"] = images_sans_label
        Ctrl_ok = False
        
    # images dupliquées
    image_names = [p.name for p in image_paths]
    duplicates = [k for k, v in Counter(image_names).items() if v > 1]
    if duplicates:
        erreurs_syntaxe["images_dupliquees"] = duplicates

    # --- Affichage des erreurs syntaxiques ---
    for key, values in erreurs_syntaxe.items():
        if values:
            util.display_and_save_errors(
                sorted(values),
                f"{key}.txt",
                key.replace("_", " ").capitalize()
            )

    return erreurs_syntaxe, all_bboxes, rapport_detail, Ctrl_ok

