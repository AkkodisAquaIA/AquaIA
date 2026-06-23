
from config import constants as ct
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors

from tools import utility as util

#=======================================================================================================

def bbox_min_max(warnings, min_val, max_val, name="bbox"):
   
    # 1. Clamp individuel
    if min_val < 0.0 :
        warnings.append(f"{name}: min < 0.0 → forcé à 0.0")
        min_val = 0.0

    if max_val > 1.0 :
        warnings.append(f"{name}: max > 1.0 → forcé à 1.0")
        max_val = 1.0

    # 2. Swap si incohérent
    if min_val > max_val:
        warnings.append(f"{name}: min > max → inversion des valeurs")
        min_val, max_val = max_val, min_val

    return min_val, max_val

def normalize_min_max(warnings, val, name="bbox", min_allowed=0.0, max_allowed=1.0):
 
    # 1. Clamp individuel
    if val < min_allowed:
        warnings.append(f"{name}: min < {min_allowed} → forcé à {min_allowed}")
        val = min_allowed

    if val > max_allowed:
        warnings.append(f"{name}: max > {max_allowed} → forcé à {max_allowed}")
        val = max_allowed

    return val

def swap(warnings, val1, val2, name="bbox", inv=False):
   
    # 1. Swap si incohérent
    if inv :
        val1, val2 = val2, val1
    if val1 < val2:
        warnings.append(f"{name}: min > max → inversion des valeurs")
        val1, val2 = val2, val1

    return val1, val2

#--------------------------------------------------------------------------------------------------------
def controle(cfg):

    display = dc.DisplayColor()

    warnings = []

    cfg["IOU_THRESHOLD"] = normalize_min_max(warnings, cfg["IOU_THRESHOLD"], "PERCEIOU_THRESHOLDNTILE_WARNING", 0.50, 1.00)

    cfg["MIN_BBOX"], cfg["MAX_BBOX"] = bbox_min_max(warnings, cfg["MIN_BBOX"], cfg["MAX_BBOX"], "Bbox")
    cfg["MIN_BBOX_AREA"], cfg["MAX_BBOX_AREA"] = bbox_min_max(warnings, cfg["MIN_BBOX_AREA"], cfg["MAX_BBOX_AREA"], "Bbox_Area")

    cfg["PERCENTILE_WARNING"] = normalize_min_max(warnings, cfg["PERCENTILE_WARNING"], "PERCENTILE_WARNING", 70.0, 90.0)
    cfg["PERCENTILE_ERROR"] = normalize_min_max(warnings, cfg["PERCENTILE_ERROR"], "PERCENTILE_ERROR", 90.0, 99.0)

    cfg["MIN_BBOX_OVERFLOW_WARNING"] = normalize_min_max(warnings, cfg["MIN_BBOX_OVERFLOW_WARNING"], "MIN_BBOX_OVERFLOW_WARNING",  5.0, 20.0)
    cfg["MIN_BBOX_OVERFLOW_ERROR"] = normalize_min_max(warnings, cfg["MIN_BBOX_OVERFLOW_ERROR"], "MIN_BBOX_OVERFLOW_ERROR", 15.0, 40.0)

    cfg["RARE"] = normalize_min_max(warnings, cfg["RARE"], "RARE", 0.0, 100.0)
    cfg["DOMINANT"] = normalize_min_max(warnings, cfg["DOMINANT"], "DOMINANT", 0.0, 100.0)
    cfg["DOMINANT"], cfg["RARE"]  = swap(warnings, cfg["DOMINANT"], cfg["RARE"], "Class imbalance thresholds")


    profile = cfg.get("PROFILES", "normal").strip().lower()

    if profile not in ct.PROFILES:
        warnings.append(
            f"Profil inconnu '{profile}' : utilisation du profil normal"
        )
        profile = "normal"

    cfg["IMBALANCE"] = ct.PROFILES[profile]


    # -----------------------------------------------------------------------------------

    if len(warnings) != 0 :

        display.print('*------------ Erreur de saisies dans fichier de paramètres ------------', colors["warning"])
        for w in warnings:            
            display.print(w, colors["warning"])
        display.print('----------------------------------------------------------------------*', colors["warning"])
        print()


    if ct.LOAD_DIR:

        required_paths = {
            "DATASET_DIR": "dataset",
        }

        missing_config = False

        for key, label in required_paths.items():
            if not cfg.get(key):
                display.print(f"Répertoire du {label} non défini", colors["warning"])
                missing_config = True

        ct.LOAD_DIR = not missing_config

    if ct.LOAD_DIR:
        display.print(f"Répertoire du dataset : {cfg['DATASET_DIR']}", colors["ok"])
    else:
        display.print("Chargement via fichier de config désactivé\n  Chargement manuel des chemins!!", colors["warning"])
