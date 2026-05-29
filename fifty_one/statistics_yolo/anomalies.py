

from collections import Counter, defaultdict

from tools import utility as util
from config import constants as ct
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from tools import graphe as gr


#==============================================================================


#==============================================================================


def recherche_anomalie(stats, info_anomalie, path_user, cfg):

    display = dc.DisplayColor()

    anomalies = info_anomalie[0]
    resultats = info_anomalie[1]
    afficher_hist = info_anomalie[2]

     # regroupement anomalies par image et type
    anomaly_images = defaultdict(lambda: defaultdict(int))
    for a in anomalies:
        img = a["image"]
        typ = a["type"]
        anomaly_images[img][typ] += 1

    # ---- 
    display.print("Recherche d'anomalies" , colors['titre'])
 
    if not anomaly_images :
        display.print('Aucune anomalie trouvé !!', colors['ok'])
        print()
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

            tag = f"-------------------------- {texte} ----------------------"
            print(tag)
            
            # Then limit to 10
            for img, d in valid_images[:ct.MAX_WORST_IMAGES]:
                ratio = (d['count'] / d['bbox_total']) * 100

                print(
                    f"score = {d['score']:.2f} : "
                    f"anomalies = {d['count']:<3} "
                    f"│ Nb bbox = {d['bbox_total']:<3} │ ratio {ratio:6.2f}% : "
                    f"{img:<25}"
                )

            print("-" * len(tag))
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

