
import os
from tools import system as syst

if syst.est_linux():
    os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")

from datetime import datetime
from pathlib import Path
from collections import defaultdict
import yaml
import shutil
from contextlib import redirect_stdout
import re
from io import StringIO


import fiftyone as fo
import fiftyone.core.labels as fol

from bboxes import bboxes as bb
from statistics_yolo import dataset_statistics_yolo as ds

from config import valid_conf as vc
from config.process import load_config
from config import constants as ct
from config.constants import DISPLAY_COLORS as colors

from tools import utility as util
import tools.display_color as dc
from tools import rapport as rp
from tools import graphe as gr
from tools import logo_win as lw
from tools import logo_linux as ll

# ansi_escape = re.compile(
#     r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
# )


#========================================================================================


def load_class_names(dataset_yaml_path):

    with open(dataset_yaml_path, "r", encoding="utf-8") as f:

        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, dict):

        names = [names[i] for i in sorted(names.keys())]

    return names

def statistique(DATASET_DIR, cfg, class_names, path_save, rapport):

     # ================= STATISTICS =================
    results = ds.dataset_statistics_yolo(DATASET_DIR, rapport, cfg)

    seuils = util.calibrer_seuils_overflow(
        results,
        warning_percentile= cfg["PERCENTILE_WARNING"],
        error_percentile= cfg["PERCENTILE_ERROR"],
        min_warning= cfg["MIN_BBOX_OVERFLOW_WARNING"],
        min_error= cfg["MIN_BBOX_OVERFLOW_ERROR"]
    )
    
    BBOX_OVERFLOW_WARNING = seuils['BBOX_OVERFLOW_WARNING']
    BBOX_OVERFLOW_ERROR   = seuils['BBOX_OVERFLOW_ERROR']

    outside_ratios = [a['outside_ratio_pct'] for a in results.get('anomalies',
                                            []) if 'outside_ratio_pct' in a]

    if outside_ratios :
        gr.bbox_overflow(cfg, outside_ratios, BBOX_OVERFLOW_WARNING, BBOX_OVERFLOW_ERROR) 

    resultat = ds.afficher_dataset_statistics(results, cfg, path_save, class_names)

    rp.ecrire_sortie_dans_rapport(
        rapport,
        ds.file_dataset_statistics,
        results,
        cfg,
        path_save,
        class_names
    )

    return resultat

def group_anomalies(anomalies):
    grouped = defaultdict(list)
    for a in anomalies:
        grouped[a["image"]].append(a)
    return grouped

def create_dataset(DATASET_DIR,  yaml_path=None, anomalies=None):

    display = dc.DisplayColor()

    dataset_name = (
                f"{Path(DATASET_DIR).name}"
                f"{'_def' if anomalies is not None else '_ok'}"
                )

    display.print("Création du dataset FiftyOne :", colors['info'])
    print(f"    '{dataset_name}'")

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    fo.close_app()

    # Mini barre
    progress = util.MiniProgressBar("Chargement dataset", width=20)
    progress.start()


    # =========================================================
    # CAS 1 : dataset classique YOLO
    # =========================================================
    if anomalies is None:

        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv5Dataset, # type: ignore
            dataset_dir=str(DATASET_DIR),
            yaml_path=str(yaml_path),
            name=dataset_name
        )

    # =========================================================
    # CAS 2 : dataset avec anomalies
    # =========================================================
    else:

        dataset = fo.Dataset(dataset_name)

        grouped = group_anomalies(anomalies)
        samples = []

        for img_name, image_anomalies in grouped.items():

            image_path = (
                Path(DATASET_DIR) / "images/train2017" / img_name
            )

            label_path = (
                Path(DATASET_DIR)
                / "labels/train2017"
                / img_name.replace(".jpg", ".txt")
            )

            if not image_path.exists() or not label_path.exists():
                print(f"Fichier manquant pour {img_name}")
                continue

            detections = []

            # Lire fichier YOLO
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:

                line = line.strip()

                # ignorer ligne vide
                if not line:
                    continue

                parts = line.split()

                x_center, y_center, width, height = map(
                    float,
                    parts[1:5]
                )

                # Associer bbox aux anomalies
                for anomaly in image_anomalies:

                    if (
                        abs(anomaly["width"] - width) < 1e-6
                        and abs(anomaly["height"] - height) < 1e-6
                    ):

                        # YOLO → FiftyOne
                        x = x_center - width / 2
                        y = y_center - height / 2

                        detection = fol.Detection(
                            label=anomaly["type"],
                            bounding_box=[x, y, width, height],
                            confidence=1.0,
                        )

                        detections.append(detection)

            sample = fo.Sample(filepath=str(image_path))
            sample["anomalies"] = fol.Detections(
                detections=detections
            )

            samples.append(sample)

        dataset.add_samples(samples)

    progress.stop()

    display.print(f"Dataset créé avec {len(dataset)} images", colors['ok'])  # type: ignore

    return dataset

def deplac_prob(DATASET_DIR, path_save,type_dep):

    display = dc.DisplayColor()


    type_d = type_dep.partition("_")[0]
    ext = ".txt" if type_d == "labels" else ".jpg"
        
    fichier_a_trouver = f"def_conf_{type_dep}.txt"

    fichiers = sorted(
            path_save.glob(fichier_a_trouver)
        )

    print()
    if fichiers:
        type_dep = (fichiers[-1])
        nom = resultat = " ".join(Path(type_dep).stem.split("_")[2:]).rstrip("_")

        display.print(f"Des {nom} ont été trouvé !", colors["warning"])
        if util.answer_yes_or_no(f"Voulez-vous déplacer ces {type_d}") :

            backup_dir = DATASET_DIR / type_d / "problems" 
            backup_dir.mkdir(exist_ok=True)

            with open(type_dep, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]

            for name in names:
                source_file = DATASET_DIR / type_d / "train2017" / f"{name}{ext}" 
                
                if source_file.exists():
                    destination = backup_dir / source_file.name
                    
                    try:
                        # os.chmod(source_file, 0o777)  # sécurité Windows
                        shutil.move(str(source_file), str(destination))
                    except Exception as e:
                        print(f"Erreur sur {source_file.name} : {e}")
                else:
                    print(f" - {source_file.name} : Introuvable/Supprimè")

def sup_file_def(path_file):

    # Recherche des fichiers correspondant au motif
    fichiers = list(Path(path_file).glob("def_conf_*.txt"))

    # Suppression si la liste n'est pas vide
    if fichiers:
        for fichier in fichiers:
            fichier.unlink()


#------------------------------------------------------------------------------------------------
def main():
    # ================= CONFIG =================
   
    fo.config.show_progress_bars = False 

    display = dc.DisplayColor()

    # Efface l'écran avant de commencer
    syst.clear_screen()

    # Display du logo et infos système
    if syst.est_windows():
        lw.splash_screen_circle("Image1.png") 
    else:
        ll.splash_screen_circle("Image1.png")

    display.print(ct.INFO_PROD, colors['aqua_light'])


    # Chargement & Vérification du fichier de Paramètrage
    try: 
        cfg = load_config()
    except Exception as e:
        display.print(f"Erreur de chargement du fichier de configuration :\n   {e}", colors['error'])
        print()
        util.sortie_de_programme()
        return
    
    print()
    vc.controle(cfg) # type: ignore
    print()

    
    # Contrôle répertoire de sauvegarde
    path_save = Path(cfg["SAVE_USER"])

    if not path_save.exists():
        path_save = Path.cwd() / "Report"
        path_save.mkdir(parents=True, exist_ok=True)

        display.print(
            f"Chemin de sauvegarde invalide : création du répertoire 'Report'{ct.BELL}",
            colors["error"]
        )
        color = colors["warning"]
    else:
        color = colors["ok"]

    display.print(f"Chemin de sauvegarde : {path_save}", color)


    print()
    # Report mode handling
    util.afficher_mode("Sauvegarde des Défauts :", cfg["REPORT_MODE"], path_save) # type: ignore
    if cfg["REPORT_MODE"] :
        sup_file_def(path_save)

    # Graphe mode handling
    util.afficher_mode("Sauvegarde des Graphiques :", cfg["SAVE_PLOT"], path_save) # type: ignore


    # Chargement du Répertoire du Dataset
    if ct.LOAD_DIR :
        DATASET_DIR : Path = Path(cfg["DATASET_DIR"])
    else :
        DATASET_DIR = util.get_path_color("Entrée le chemin du dataset")

    

    #  ------------------------------------------------------------------------------------------
    display.print("Démarrage du traitement", colors['titre'])

    # Chargement des noms de classes pour les stats
    DATASET_DIR  = Path(DATASET_DIR )

    # Recherche des fichiers .yaml
    # yaml_files = list(DATASET_DIR.glob("*.yaml")) + list(DATASET_DIR.glob("*.yml"))
    yaml_files = list(DATASET_DIR.glob("*.yaml"))
    dataset_yaml_ = ""

    # Gestion des fichiers .yaml
    # Si aucun fichier .yaml trouvé -> message d'erreur et sortie du programme
    if len(yaml_files) == 0:
        display.print(f"Aucun fichier .yaml trouvé dans : \n   {DATASET_DIR}", colors['error'])
        print()
        util.sortie_de_programme()
    
    # Si un seul fichier .yaml trouvé -> on l'utilise
    elif len(yaml_files) == 1:
        # Un seul fichier -> on l'utilise
        dataset_yaml_ =  yaml_files[0]

    # Si plusieurs fichiers .yaml trouvés -> demander lequel utiliser
    else:
        # Plusieurs fichiers -> demander lequel utiliser
        display.print("Plusieurs fichiers YAML trouvés :", colors['warning'])

        for i, file in enumerate(yaml_files, start=1):
            print(f"  - {i} : {file.name}")

        choix = util.selection(len(yaml_files)) 
        dataset_yaml_ = yaml_files[choix]

    # Chargement
    nom = Path(dataset_yaml_).name

    display.print(f"fichier .yaml utilisé : {nom}", colors['ok'])
    print()
    dataset_yaml =  DATASET_DIR / dataset_yaml_


    # Création d'un fichier de rapport
    rapport = rp.create_file_report(DATASET_DIR, path_save, nom, cfg)


    try:
        class_names = load_class_names(dataset_yaml)
    except Exception as e:
        display.print(f"dataset.yaml introuvable dans {DATASET_DIR}\n", colors['error'])
        util.sortie_de_programme()

    ctrl_ok = False
    # validation des labels avant création du dataset FiftyOne
    erreur, ctrl_ok = bb.validate_yolo_dataset_detailed(DATASET_DIR, path_save, rapport, cfg)
 
    # Affichage des erreurs détectées
    if not ctrl_ok:
        display.print("—" * 80, colors['error'])
        display.print(f"Erreurs détectées dans les images/labels. Arrêt du programme {ct.BELL}", colors['error'])
        total_errors = sum(len(v) for v in erreur.values())
        
        label1 = "Total Types           :"
        label2 = "Total warning/erreurs :"
                
        value1 = len(erreur)
        value2 = total_errors

        label_width = max(len(label1), len(label2))
        value_width = max(len(str(value1)), len(str(value2)))

        display.print(f"{label1:<{label_width}} {value1:>{value_width}}", colors['error'])
        display.print(f"{label2:<{label_width}} {value2:>{value_width}}", colors['error'])    
        print()
        util.afficher_bbox_erreurs_compact(erreur)
        display.print("—" * 80, colors['error'])

        rp.ecrire_sortie_dans_rapport(
            rapport,
            util.afficher_bbox_erreurs_compact,
            erreur
        )
 
        try:
            deplac_prob(DATASET_DIR, path_save, "labels_invalides")
            deplac_prob(DATASET_DIR, path_save, "labels_orphelins")
            deplac_prob(DATASET_DIR, path_save, "labels_vides")

            deplac_prob(DATASET_DIR, path_save, "images_invalides")
            deplac_prob(DATASET_DIR, path_save, "images_sans_label")
        except FileNotFoundError:
            display.print(f"   fichiers d'erreurs introuvables\n", colors['error'])
            util.sortie_de_programme()

    # Si aucune erreur détectée, continuer l'analyse du dataset et création du dataset FiftyOne
    else:    
        print()
        display.print("Aucune erreur détectée. Poursuite de l'analyse...\n", colors['ok'])

        def_image = statistique(DATASET_DIR, cfg, class_names, path_save, rapport) # type: ignore


        # ----- Résumé -----
        display.print("Résumé", colors['titre'])

        lauch_fifty = False        
        if not def_image:
            display.print("Dataset Ok ", colors['ok'])

            # TODO
            # display.print("Création du dataset pour FiftyOne", colors['titre'])
            # dataset = create_dataset(DATASET_DIR, yaml_path= dataset_yaml)

            # print()
            # lauch_fifty = util.answer_yes_or_no("Voulez-vous lancer Fifty_one") 
    
        # else:

            # TODO
            # display.print("Dataset non valide ", colors['warning'])
            # display.print("Création d'un dataset d'anomalies pour FiftyOne et lancement de celui-ci", colors['titre'])
            # dataset = create_dataset(DATASET_DIR,anomalies=def_image)
            # print()

        # TODO
        # # Lancement de l'interface FiftyOne
        # if def_image or lauch_fifty :
        #     util.launch_fiftyone_interface(dataset) # type: ignore

        
    print()
    util.sortie_de_programme()

#==========================================================================================
if __name__ == "__main__":
    main()
