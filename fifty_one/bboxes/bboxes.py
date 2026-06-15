import os
import re
from PIL import Image
from PIL import UnidentifiedImageError
from pathlib import Path
from collections import  defaultdict
from tqdm import tqdm

from tools import utility as util
from tools import rapport as rp
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from config import constants as ct


#==================================================================================

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return "valid"

    except FileNotFoundError:
        return "deleted"

    except (UnidentifiedImageError, OSError):
        return "invalid"

def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - h1/2, y1 + h1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - h2/2, y2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    union = w1*h1 + w2*h2 - inter_area
    if union == 0:
        return 0
    return inter_area / union

def round_bbox(cls, x, y, w, h, decimals=4):
    return (cls, round(x, decimals), round(y, decimals), round(w, decimals), round(h, decimals))

def round_coords(x, y, w, h, decimals=4):
    return (round(x, decimals), round(y, decimals), round(w, decimals), round(h, decimals))


#==================================================================================
#-----------------------------------------------------------------------------------
display = dc.DisplayColor()

def validate_yolo_dataset_detailed(DATASET_DIR, path_user, rapport, cfg):

    print()
    display.print("Début de l'analyse", colors['info'])
    display.print(" Analyse de la conformité", colors['info'])

    try:
        images_dir, labels_dir = util.get_dataset_paths(DATASET_DIR)

    except FileNotFoundError as e:
        display.print(str(e), colors['error'])
        util.sortie_de_programme()

    split_pattern = re.compile(r"[,\s]+")
    
    erreurs_syntaxe = defaultdict(list)
    rapport_detail = defaultdict(lambda: defaultdict(list))
    ctrl_ok = True
    label_stems = set()
    all_bboxes = []

    rp.suivi("Analyse conformité", rapport, "D")

    #====================================================================================================
    # Analyse des fichiers 'labels'
    for entry in tqdm(list(os.scandir(labels_dir)), # type: ignore 
                    desc=" - Labels",
                    unit=" fichier",
                    ncols=100,
                    position=0): 
        # for entry in os.scandir(labels_dir):
        if not entry.name.lower().endswith(".txt"):
            continue
        label_stems.add(Path(entry.name).stem)
        path = entry.path
        has_content = False

        try:
            with open(path, "r", encoding="utf-8") as f:

                seen_boxes = set()
                seen_coords_classes = {}            
                seen_boxes_list = []
                seen_boxes_by_class = defaultdict(list)

                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        # --- ligne vide ---
                        erreurs_syntaxe["lignes_vides"].append(f"{entry.name} (ligne {i})")     
                        rapport_detail[entry.name][i].append("lignes_vides")
                        ctrl_ok = False
                        continue

                    has_content = True
                    parts = split_pattern.split(line)
                    erreurs_ligne = []

                    cls, x, y, w, h = -1, -1, -1, 0, 0

                    # --- Colonnes ---
                    if len(parts) != 5:
                        erreurs_syntaxe["lignes_incorrectes"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("lignes_incorrectes")
                        ctrl_ok = False
                        continue 

                    # --- Classe ---
                    try:
                        cls = int(parts[0])
                        if ct.NB_CLASSES is not None and cls >= ct.NB_CLASSES :
                            # --- Classe hors plage ---
                            erreurs_syntaxe["classe_hors_plage"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("classe_hors_plage")
                            ctrl_ok = False
                    except ValueError:
                        # --- Classe invalides ---
                        erreurs_syntaxe["classe_invalide"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("classe_invalide")
                        ctrl_ok = False
                        continue

                    # Classe_ok : 
                    # --- Conversion bbox ---
                    float_ok = True
                    try:
                        x, y, w, h = map(float, parts[1:])
                    except ValueError:
                        # --- Valeurs non numériques ---
                        erreurs_syntaxe["valeurs_non_numeriques"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("valeurs_non_numeriques")
                        float_ok = False
                        ctrl_ok = False

                    if float_ok:
                        # --- Classe négative ---
                        if cls < 0:
                            erreurs_syntaxe["classe_negative"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("classe_negative")
                            ctrl_ok = False

                        # --- Coordonnées négatives ---
                        if x < 0 or y < 0:
                            erreurs_syntaxe["coord_negatives"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("coord_negatives")
                            ctrl_ok = False
                        # --- Coordonnées > 1 ---
                        if x > 1 or y > 1:
                            # util.quoi(" *" * 10)
                            erreurs_syntaxe["coord_sup_1"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("coord_sup_1")
                            ctrl_ok = False

                        # --- Taille négative ---
                        if w <= 0 or h <= 0:
                            erreurs_syntaxe["taille_negatives"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("taille_negatives")
                            ctrl_ok = False
                        # --- Taille > 1 ---    
                        if w > 1 or h > 1:
                            erreurs_syntaxe["taille_sup_1"].append(f"{entry.name} (ligne {i})")
                            erreurs_ligne.append("taille_sup_1")
                            ctrl_ok = False

                    #--- BBox dupliquées ---
                    bbox_tuple_rounded = round_bbox(cls, x, y, w, h)
                    if bbox_tuple_rounded in seen_boxes:
                        erreurs_syntaxe["bbox_dupliquees"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("bbox_dupliquees")
                        ctrl_ok = False
                    seen_boxes.add(bbox_tuple_rounded)

                    # --- Classes différentes avec mêmes coordonnées ---
                    coords_tuple = round_coords(x, y, w, h)
                    if coords_tuple not in seen_coords_classes:
                        seen_coords_classes[coords_tuple] = set()
                    seen_coords_classes[coords_tuple].add(cls)
                    if len(seen_coords_classes[coords_tuple]) > 1:
                        erreurs_syntaxe["bbox_classes_differentes"].append(f"{entry.name} (ligne {i})")
                        erreurs_ligne.append("bbox_classes_differentes")
                        ctrl_ok = False

                    # --- IoU suspect ---
                    current_box = (x, y, w, h)
                    for prev_box, prev_line in seen_boxes_by_class[cls]:
                        iou = bbox_iou(current_box, prev_box)
                        if iou > cfg["IOU_THRESHOLD"] :
                            erreurs_syntaxe["bbox_IoU_suspect"].append(
                                f"{entry.name} (classe {cls}, lignes {prev_line}-{i}) IoU={iou:.3f}"
                            )
                            erreurs_ligne.append("bbox_IoU_suspect")
                            ctrl_ok = False

                    # Ajout de la bbox courante
                    seen_boxes_by_class[cls].append((current_box, i))
                    seen_boxes_list.append((cls, current_box, i))

                    # --- Collecte bboxes ---
                    all_bboxes.append((cls, x, y, w, h, entry.name))
                    if erreurs_ligne:
                        rapport_detail[entry.name][i].extend(erreurs_ligne)

        except FileNotFoundError:
            nom = os.path.splitext(entry.name)[0]
            erreurs_syntaxe["labels_disparus"].append(nom)
            rapport_detail[nom][0].append("labels_disparus")
            ctrl_ok = False
            continue

        except UnicodeDecodeError:
            nom = os.path.splitext(entry.name)[0]
            erreurs_syntaxe["labels_invalides"].append(nom)
            rapport_detail[nom][0].append("labels_invalides")
            ctrl_ok = False
            continue


        if not has_content:
            erreurs_syntaxe["labels_vides"].append(os.path.splitext(entry.name)[0])
            rapport_detail[entry.name][0].append("labels_vides")
            ctrl_ok = False


    #====================================================================================================
    # Analyse des images
    image_paths = [p for p in Path(images_dir).glob("*") if p.suffix.lower() in ct.IMAGE_EXT] # type: ignore
    image_stems = {p.stem for p in image_paths}

    # labels orphelins
    orphan_labels = sorted(label_stems - image_stems)
    if orphan_labels:
        ctrl_ok = False
        erreurs_syntaxe["labels_orphelins"] = orphan_labels
        for lbl in orphan_labels:
            rapport_detail[lbl][0].append("labels_orphelins")


    # Vérification images invalides / corrompues
    images_invalides = []
    images_supprimees = []
    
    for p in tqdm(
            image_paths,
            desc=" - Images",
            unit=" image",
            ncols=100,
            position=0):

        result = check_image(p)

        if result == "invalid":
            images_invalides.append(p.stem)

        elif result == "deleted":
            images_supprimees.append(p.stem)

    # images sans label
    images_sans_label = sorted(image_stems - label_stems)
    if images_sans_label:
        erreurs_syntaxe["images_sans_label"] = images_sans_label
        ctrl_ok = False

    if images_invalides:
        erreurs_syntaxe["images_invalides"] = sorted(images_invalides)
        ctrl_ok = False

    if images_supprimees:
        erreurs_syntaxe["images_supprimees"] = sorted(images_supprimees)
        ctrl_ok = False


    #====================================================================================
    # --- Affichage erreurs ---
    for key, values in erreurs_syntaxe.items():
        if values:
            print()
            util.display_and_save_errors(
                cfg,
                path_user,
                sorted(values),
                f"{key}.txt",
                key.replace("_", " ").capitalize()
            )
    
    rp.suivi("Analyse conformité", rapport)
    
    #
    # all_bboxes : liste avec toutes les Bboxes
    # rapport_detail    : collections.defaultdict
    #

    return erreurs_syntaxe, ctrl_ok
