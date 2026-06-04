from pathlib import Path

from tools import utility as util
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors

#============================================================================
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
    warnings = []

    def add_row(label, data, cmin, cmax):

        min_ok = data["min"] >= cmin
        max_ok = data["max"] <= cmax

        if not min_ok:
            warnings.append(
                f"{label}: valeur MIN ({data['min']:.4f}) < limite config ({cmin})"
            )

        if not max_ok:
            warnings.append(
                f"{label}: valeur MAX ({data['max']:.4f}) > limite config ({cmax})"
            )

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

    # --- Diagnostic global ---
    if warnings:
        display = dc.DisplayColor()
        print()
        display.print("Erreur : Dataset vs Configuration", colors["error"])

        print("Les valeurs suivantes sont hors limites par rapport au fichier de configuration :\n")

        for w in warnings:
            print(f" - {w}")
    else:
        display = dc.DisplayColor()
        print()
        display.print("Dataset conforme aux limites de configuration", colors["ok"])


def verifier_classes_dataset(class_distribution, class_names):
    n_classes = len(class_names)
    classes_presentes = set(class_distribution.keys())
    classes_yaml = set(range(n_classes))
    inutilisees = classes_yaml - classes_presentes
    manquantes = classes_presentes - classes_yaml
    valides = classes_presentes & classes_yaml
    return {"inutilisees": inutilisees, "manquantes": manquantes, "valides": valides}

#============================================================================


def afficher_info_general(stats, info_general, class_names, cfg):

    display = dc.DisplayColor()

    total_boxes = info_general[0]
    total_classes = info_general[1]
    class_distribution = info_general[2] 
    class_to_images = info_general[3]

    # --- résumé général ---
    display.print("Dataset Summary", colors["titre"])
    print(f"{'Images':18}: {stats['images']}")
    print(f"{'Labels':18}: {stats['labels']}")
    print(f"{'Bounding boxes':18}: {total_boxes}")
    print(f"{'BBox / image':18}: {total_boxes / stats['images']:.2f}")
    print(f"{'Classes':18}: {total_classes}")
    print()

    # --- statistiques BBOX ---
    display.print("Statistiques des BBOX", colors["titre"])
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
            missing_label = True

            display.print(
                f"Classes présentes dans labels mais absentes du YAML ({len(verif['manquantes'])}) :",
                colors["warning"]
            )

            for cls in sorted(verif["manquantes"]):
                fichiers = class_to_images.get(cls, [])
                print(f"Classe {cls}")

                if fichiers:
                    for f in fichiers:
                        print(f"   └── {Path(f).name}")
                else:
                    print("   └── Aucun fichier trouvé")

