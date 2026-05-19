

import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors


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
        min_val = min_allowed

    if val > max_allowed:
        warnings.append(f"{name}: max > {max_allowed} → forcé à {max_allowed}")
        max_val = max_allowed

    return val

def swap(warnings, val1, val2, name="bbox", inv=False):
   
    # 1. Swap si incohérent
    if inv :
        val1, val2 = val2, val1
    if val1 < val2:
        warnings.append(f"{name}: min > max → inversion des valeurs")
        val1, val2 = val2, val1

    return val1, val2

def est_negatif(warnings, val, name="bbox"):
    if val < 0 :
        warnings.append(f"{name} doit être positif : utilisation de la valeur absolue")
        return abs(val)
    return val



def controle(cfg):

    display = dc.DisplayColor()

    warnings = []

    cfg["IOU_THRESHOLD"] = normalize_min_max(warnings, cfg["IOU_THRESHOLD"], "PERCEIOU_THRESHOLDNTILE_WARNING", 0.50, 1.00)

    cfg["MIN_BBOX"], cfg["MAX_BBOX"] = bbox_min_max(warnings, cfg["MIN_BBOX"], cfg["MAX_BBOX"], "Bbox")
    cfg["MIN_BBOX_AREA"], cfg["MAX_BBOX_AREA"] = bbox_min_max(warnings, cfg["MIN_BBOX_AREA"], cfg["MAX_BBOX_AREA"], "Bbox_Area")


    cfg["PERCENTILE_WARNING"] = est_negatif(warnings, cfg["PERCENTILE_WARNING"], 'PERCENTILE_WARNING')
    cfg["PERCENTILE_ERROR"] = est_negatif(warnings, cfg["PERCENTILE_ERROR"], 'PERCENTILE_ERROR')

    cfg["PERCENTILE_WARNING"] = normalize_min_max(warnings, cfg["PERCENTILE_WARNING"], "PERCENTILE_WARNING", 70.0, 90.0)
    cfg["PERCENTILE_ERROR"] = normalize_min_max(warnings, cfg["PERCENTILE_ERROR"], "PERCENTILE_ERROR", 90.0, 99.0)

    cfg["MIN_BBOX_OVERFLOW_WARNING"] = normalize_min_max(warnings, cfg["MIN_BBOX_OVERFLOW_WARNING"], "MIN_BBOX_OVERFLOW_WARNING", 05.0, 20.0)
    cfg["MIN_BBOX_OVERFLOW_WARNING"] = normalize_min_max(warnings, cfg["MIN_BBOX_OVERFLOW_WARNING"], "MIN_BBOX_OVERFLOW_WARNING", 15.0, 40.0)

    cfg["DOMINANT"], cfg["RARE"]  = swap(warnings, cfg["DOMINANT"], cfg["RARE"], "Class imbalance thresholds")


    cfg["RATIO_OK"] = est_negatif(warnings, cfg["RATIO_OK"], 'RATIO_OK')
    cfg["RATIO_WARNING"] = est_negatif(warnings, cfg["RATIO_WARNING"], 'RATIO_WARNING')

    cfg["ENTROPY_OK"] = est_negatif(warnings, cfg["ENTROPY_OK"], 'ENTROPY_OK')
    cfg["ENTROPY_WARNING"] = est_negatif(warnings, cfg["ENTROPY_WARNING"], 'ENTROPY_WARNING')

    cfg["SCORE_OK"] = est_negatif(warnings, cfg["SCORE_OK"], 'SCORE_OK')
    cfg["SCORE_WARNING"] = est_negatif(warnings, cfg["SCORE_WARNING"], 'SCORE_WARNING')

    cfg["RATIO_OK"], cfg["RATIO_WARNING"] = swap(warnings, cfg["RATIO_OK"], cfg["RATIO_WARNING"],  "RATIO", inv=True)
    cfg["ENTROPY_OK"], cfg["ENTROPY_WARNING"] = swap(warnings, cfg["ENTROPY_OK"], cfg["ENTROPY_WARNING"], "ENTROPY")
    cfg["SCORE_OK"], cfg["SCORE_WARNING"] = swap(warnings, cfg["SCORE_OK"], cfg["SCORE_WARNING"], "SCORE")


    if len(warnings) != 0 :

        display.print('*------------ Erreur de saisies dans fichier de paramètres ------------', colors["warning"])
        for w in warnings:            
            display.print(w, colors["warning"])
        display.print('----------------------------------------------------------------------*', colors["warning"])
        