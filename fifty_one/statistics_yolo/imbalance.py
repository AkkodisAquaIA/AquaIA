import numpy as np

from tools import utility as util
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from config import constants as ct

display = dc.DisplayColor()

#============================================================================================

def imbalance_metrics(class_distribution, cfg):

    counts = np.array(list(class_distribution.values()), dtype=float)

    total = np.sum(counts)

    p = counts / total

    # Entropie de Shannon
    entropy = -np.sum(p * np.log2(p + 1e-12))

    # Entropie maximale
    max_entropy = np.log2(len(counts))

    # Score de déséquilibre
    imbalance_score = 1 - (entropy / max_entropy)

    return {
        "imbalance_score": imbalance_score
    }


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
    bare = util.draw_bar(value, bar_min, bar_max)   
    print(f"{'':25}  {bare}\n")

#----------------------------------------------------------------------------------------
def afficher_imbalance_avance(metrics, display, colors, cfg):

    print()
    display.print(f"Utilisation d'un profil {cfg["IMBALANCE"][2]}", colors['info'])
    print()

    imbalance_score = float(metrics["imbalance_score"])
    
    evaluate_metric(
        label="Score déséquilibre",
        value=imbalance_score,
        thresholds=(cfg["IMBALANCE"][1], cfg["IMBALANCE"][0]),
        statuses=(
            "Fort",
            "Modéré",
            "Faible"
        ),
        colors_map=colors,
        bar_min=0,
        bar_max=1,
        display=display,
        higher_is_better=False
    )

    print(
        "Score de déséquilibre par entropie normalisée : "
        "mesure l'écart entre la distribution observée et une "
        "répartition parfaitement uniforme des classes."
    )
    print(
        "Un score proche de 0 indique un dataset équilibré, "
    "un score proche de 1 indique un fort déséquilibre.\n"
    )


    # --- DIAGNOSTIC ---
    display.print("  - Diagnostic :", colors['info'])

    if imbalance_score < cfg["IMBALANCE"][0]:
        display.print(f"- Dataset globalement équilibré", colors['ok'])

    elif imbalance_score < cfg["IMBALANCE"][1]:
        display.print(f"- Dataset modérément déséquilibré", colors['warning'])
        
    else:
        display.print(f"- Dataset fortement déséquilibré !!! {ct.BELL}", colors['error'])
