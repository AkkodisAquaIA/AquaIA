
from collections import Counter, defaultdict

from tools import utility as util
from config import constants as ct
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from tools import graphe as gr


# ==============================================================
# COLOR UTIL
# ==============================================================

def colorize(value, color):
    r, g, b, _ = color
    rgb = f"\033[38;2;{r};{g};{b}m"
    return f"{rgb}{value}\033[0m"


# ==============================================================
# DATA PREPARATION
# ==============================================================

def build_anomaly_matrix(anomalies, types, id_to_type):
    matrix = defaultdict(lambda: defaultdict(int))

    for a in anomalies:
        matrix[a["image"]][a["type"]] += 1

    return matrix


def compute_totals(anomaly_matrix, id_to_type):
    totaux = {i: 0 for i in range(1, 7)}

    for img_data in anomaly_matrix.values():
        for i in range(1, 7):
            t = id_to_type[i]
            totaux[i] += img_data.get(t, 0)

    return totaux


# ==============================================================
# DISPLAY TABLE
# ==============================================================

def display_table(anomaly_matrix, sorted_images, totaux, id_to_type):
    col_width = 5

    header = f"{'Image':25} │ " + " │ ".join(f"{i:^{col_width}}" for i in range(1, 7))
    print(header)
    print("─" * len(header))

    for img, data in sorted_images[:ct.MAX_WORST_IMAGES]:

        line_parts = [f"{img:25}"]
        row_sum = 0

        for i in range(1, 7):
            t = id_to_type[i]
            count = data.get(t, 0)
            row_sum += count

            cell = f"{count:^{col_width}}"

            if count > 0:
                color = colors["error"] if i == 6 else colors["warning"]
                cell = colorize(cell, color)

            line_parts.append(cell)

        line_parts.append(f"{row_sum:^{col_width}}")
        print(" │ ".join(line_parts))


def display_totals(totaux, col_width=5):
    total_line = [f"{'TOTAL':25}"]

    for i in range(1, 7):
        total = totaux[i]
        cell = f"{total:^{col_width}}"

        if total > 0:
            color = colors["error"] if i == 6 else colors["warning"]
            cell = colorize(cell, color)

        total_line.append(cell)

    print(" │ ".join(total_line))


# ==============================================================
# SCORE
# ==============================================================

def compute_scores(anomalies, bbox_per_image, weights):
    max_weight = max(weights.values())

    scores = defaultdict(lambda: {
        "count": 0,
        "severity": 0,
        "bbox_total": 0,
        "score": 0.0
    })

    for a in anomalies:
        img = a["image"]
        t = a["type"]

        scores[img]["count"] += 1
        scores[img]["severity"] += weights.get(t, 0)

    for img, total_bbox in bbox_per_image.items():
        scores[img]["bbox_total"] = total_bbox

    for img, d in scores.items():
        if d["bbox_total"] == 0:
            continue

        error_ratio = d["count"] / d["bbox_total"]

        avg_severity = d["severity"] / d["count"] if d["count"] else 0
        normalized = avg_severity / max_weight if d["count"] else 0

        d["score"] = error_ratio * normalized

    return scores


# ==============================================================
# MAIN FUNCTION
# ==============================================================

def recherche_anomalie(stats, info_anomalie, path_user, file, cfg):

    display = dc.DisplayColor()

    anomalies = info_anomalie[0]
    resultats = info_anomalie[1]

    type_to_id = {
        "bbox_trop_petite": 1,
        "bbox_trop_grande": 2,
        "bbox_surface_trop_petite": 3,
        "bbox_surface_trop_grande": 4,
        "bbox_hors_limite_warning": 5,
        "bbox_hors_limite_error": 6,
    }

    id_to_type = {v: k for k, v in type_to_id.items()}

    display.print("Recherche d'anomalies", colors['titre'])

    if not anomalies:
        display.print(" - Aucune anomalie trouvée !!", colors['ok'])
        return

    types = list(type_to_id.keys())

    anomaly_matrix = build_anomaly_matrix(anomalies, types, id_to_type)
    totaux = compute_totals(anomaly_matrix, id_to_type)

    sorted_images = sorted(
        anomaly_matrix.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )

    # ==========================================================
    # TABLE
    # ==========================================================

    display.print("---------------- ANOMALIES ----------------------", colors['warning'])
    if len(anomalies) == 1: 
        display.print(f" - La pire image ", colors['warning'])
    elif len(anomalies) <= ct.MAX_WORST_IMAGES:
        display.print(f" - Les {len(anomalies)} pires images", colors['warning'])
    else:     
        display.print(f" - Les {ct.MAX_WORST_IMAGES} premières pires images", colors['warning'])
        display.print(f"    Liste compléte dans le rapport", colors['warning'])
    print()

    print("----------------------- LEGENDES -----------------------")
    print("1 : bbox_trop_petite        │ 2 : bbox_trop_grande")
    print("3 : surface_trop_petite     │ 4 : surface_trop_grande")
    print("5 : bbox_warning_hors_zone  │ 6 : bbox_error_hors_zone")
    print()

    display_table(anomaly_matrix, sorted_images, totaux, id_to_type)

    print("─" * 80)
    display_totals(totaux)

    total_defauts = sum(totaux.values())

    print(
        f"{'TOTAL DEFAUTS':25} "
        f"{util.format_nombre(total_defauts)}"
    )

    # ==========================================================
    # SCORES
    # ==========================================================

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

    scores = compute_scores(anomalies, bbox_per_image, weights)

    worst_images = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

    valid_images = [(img, d) for img, d in worst_images if d["score"] > 0]

    if valid_images:
        print("\n---------------- QUALITE DES IMAGES ----------------")

        for img, d in valid_images[:ct.MAX_WORST_IMAGES]:
            ratio = (d["count"] / d["bbox_total"]) * 100

            print(
                f"score={d['score']:.2f} "
                f"anomalies={d['count']:<3} "
                f"bbox={d['bbox_total']:<3} "
                f"ratio={ratio:6.2f}% "
                f"{img}"
            )

    # ==========================================================
    # STATS        
    # ==========================================================

    images_problematiques = sum(1 for d in scores.values() if d["count"] > 0)
    pct_images = (images_problematiques / stats["images"]) * 100 if stats["images"] else 0

    print()
    display.print(
        f"Images avec anomalies : {util.format_nombre(images_problematiques)} ({pct_images:.3f}%)",
        colors['warning']
    )

    display.print(
        f"Total anomalies : {util.format_nombre(len(anomalies))}",
        colors['warning']
    )

    dataset_score = (
        sum(d["score"] for d in scores.values()) / len(scores)
        if scores else 0
    )

    print(f"Score moyen du dataset : {dataset_score:.3f}\n")

    # ==========================================================
    # GRAPH + EXPORT
    # ==========================================================

    if not file and (util.answer_yes_or_no("Voulez-vous afficher le graphique")):
        display.print("Attente fermeture du graphe", colors['wait'])
        type_counts = Counter(a["type"] for a in anomalies)
        gr.histogram_anomalies(type_counts, "Nombre", cfg, anomalies)
