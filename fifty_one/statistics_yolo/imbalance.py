import numpy as np

from config import process as pr
from tools import utility as util

#============================================================================================

def imbalance_metrics(class_distribution, cfg):
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

def afficher_imbalance_avance(metrics, display, colors, cfg):

    ratio = float(metrics["ratio"])
    entropy_norm = float(metrics["entropy_norm"])

    display.print("---------------- DATASET IMBALANCE ----------------", colors["info"])
    print('Ration max/min (plus petit = mieux) : indique le déséquilibre entre la classe la plus fréquente et la moins fréquente.')
    print('Entropie normalisée (plus grand = mieux) : mesure la diversité globale des classes, en tenant compte de leur distribution.')
    print('Score global : combinaison pondérée des deux métriques précédentes pour une évaluation synthétique du déséquilibre du dataset.\n')
    
    # --- RATIO (plus petit = mieux) ---
    evaluate_metric(
    label="Ratio max/min",
    value=min(ratio, 300),
    thresholds=(cfg["RATIO_WARNING"], cfg["RATIO_OK"]),
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
    thresholds=(cfg["ENTROPY_WARNING"], cfg["ENTROPY_OK"]),
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
        thresholds=(cfg["SCORE_WARNING"], cfg["SCORE_OK"]),
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

    if entropy_norm < 0.75:
        print("- Distribution globale déséquilibrée")

    if entropy_norm > 0.75 and ratio > 100:
        print("- Beaucoup de classes rares malgré une diversité correcte")
        print()
