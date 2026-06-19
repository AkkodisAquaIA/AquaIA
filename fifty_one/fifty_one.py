
import os
import re
import random
from tools import system as syst

if syst.est_linux():
    os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")

from pathlib import Path
from collections import defaultdict
import yaml
import shutil
from datetime import datetime
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

display = dc.DisplayColor()



#========================================================================================

def choix_logo():

    pattern = re.compile(r"^image_\d{2}\.png$")

    fichiers = [
    f
    for f in Path("logo").iterdir()
    if f.is_file() and pattern.match(f.name)
    ]

    return random.choice(fichiers) if fichiers else Path("logo/image_01.png")


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


    try:
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

    finally:
        progress.stop()

    display.print(f"Dataset créé avec {len(dataset)} images", colors['ok'])  # type: ignore

    return dataset


def deplacer_fichier(DATASET_DIR, dossier, extension, nom):
    source = DATASET_DIR / dossier / "train2017" / f"{nom}{extension}"
    destination = DATASET_DIR / dossier / "problems"
    destination.mkdir(exist_ok=True)

    if source.exists():
        shutil.move(source, destination / source.name)
    # else:
    #     print(f" / {source.name} : Déjà déplacé", end="")

def deplac_prob(DATASET_DIR, path_save,type_dep):

    type_d = type_dep.partition("_")[0]
    ext = ".txt" if type_d == "labels" else ".jpg"
        
    fichier_a_trouver = f"def_conf_{type_dep}.txt"
    fichiers = sorted(path_save.glob(fichier_a_trouver))

    if fichiers:
        type_dep = (fichiers[-1])

        with open(type_dep, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        for name in names:

            deplacer_fichier(DATASET_DIR, "images", ".jpg", name)
            deplacer_fichier(DATASET_DIR, "labels", ".txt", name)

def liste_def(repertoire):
    """
    Récupération des noms des fichiers qui présentent des problèmes
    """

    # Expression régulière pour capturer les noms de fichiers
    pattern = re.compile(r'^(\d+)\.txt', re.MULTILINE)
    
    liste_fichiers = []
    liste_fichiers_uniques = []


    for nom_fichier in os.listdir(repertoire):
        if nom_fichier.startswith("def_conf_") and nom_fichier.endswith(".txt"):
            chemin = repertoire / nom_fichier

            with open(chemin, "r", encoding="utf-8") as f:
                contenu = f.read()

            liste_fichiers.extend(pattern.findall(contenu))
            liste_fichiers_uniques = sorted(set(liste_fichiers))

    output_path =  repertoire / "def_conf_labels_xxxx.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in liste_fichiers_uniques :
            f.write(f"{item}\n")

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

    # Déactivation de la barre de progression de FiftyOne
    fo.config.show_progress_bars = False 

    # Efface l'écran avant de commencer
    syst.clear_screen()

    # Choix aléatoire d'un logo
    logo = choix_logo()

    # Display du logo et infos système
    if syst.est_windows():
        lw.splash_screen_circle(logo) 
    else:
        ll.splash_screen_circle(logo)

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
    path_save = Path(cfg["SAVE_USER"]) # type: ignore

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


    # Sauvegarde du fichier de configuration
    # Horodatage au format AAAAMMJJ_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source = Path("aqua_ia_conf.ini")
    destination_dir = Path(path_save)
    destination = destination_dir / f"{timestamp}_{source.name}"
    shutil.copy2(source, destination)

    # Affichage des états des modes de sauvegarde
    print()
  
    # Graphe mode handling
    util.afficher_mode("Sauvegarde des Graphiques :", cfg["SAVE_PLOT"]) # type: ignore


    # Chargement du Répertoire du Dataset
    if ct.LOAD_DIR :
        DATASET_DIR : Path = Path(cfg["DATASET_DIR"]) # type: ignore
    else :
        DATASET_DIR = util.get_path_color("Entrez le chemin du Dataset")
    

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
        display.print(f"Aucun fichier '.yaml' trouvé dans : \n   {DATASET_DIR}", colors['error'])
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

    display.print(f"fichier '.yaml' utilisé : {nom}", colors['ok'])
    print()
    dataset_yaml =  DATASET_DIR / dataset_yaml_


    # Création d'un fichier de rapport
    rapport = rp.create_file_report(DATASET_DIR, path_save, nom, cfg)


    # Validation du fichier .yaml
    tag, yaml_ok = "", True
    try:
       class_names = load_class_names(dataset_yaml)
    except yaml.YAMLError as e:
        tag = f"le fichier '.yaml' sélectionné  n'est pas conforme\n  {e} "
        display.print(tag, colors['error'])
        yaml_ok = False
    except FileNotFoundError as e:
        tag = f" '{dataset_yaml}' est introuvable dans {DATASET_DIR}\n  {e} "
        display.print(tag, colors['error'])
        yaml_ok = False

    if not yaml_ok :
        rp.repport_def(tag, rapport)
        util.sortie_de_programme()

    ctrl_ok = False
    # validation des labels avant création du dataset FiftyOne
    erreur, ctrl_ok = bb.validate_yolo_dataset_detailed(DATASET_DIR, path_save, rapport, cfg)
 

    # Gestion des erreurs de conformité
    # Déplacement des fichiers à problèmes dans les répertoires suivants
    #    images / problems pour les images
    #    labels / problems pour les labels

    print()
    if not ctrl_ok:
        print()
        display.print(" !!! Des erreurs de conformité ont été trouvées !!!\n"
                      "  les fichiers seront déplacés vers le répertoire 'problems' \n"
                      , colors['warning'])

        # Récupération des labels avec problèmes de conformité  
        liste_def(path_save)    

        # Déplacement des fichiers problèmatiques dans le répertoire 'problem'
        try:
            deplac_prob(DATASET_DIR, path_save, "labels_invalides")
            deplac_prob(DATASET_DIR, path_save, "labels_orphelins")
            deplac_prob(DATASET_DIR, path_save, "labels_vides")
            deplac_prob(DATASET_DIR, path_save, "labels_xxxx")

            deplac_prob(DATASET_DIR, path_save, "images_invalides")
            deplac_prob(DATASET_DIR, path_save, "images_sans_label")
            sup_file_def(path_save)
            print()
        except FileNotFoundError:
            tag = f"   fichiers d'erreurs introuvables"
            display.print(f"{tag}\n", colors['error'])
            rp.repport_def(tag, rapport)
            util.sortie_de_programme()
    else:
        display.print("Analyse de la conformité  terminée sans problème", colors['ok']) 
        print()

    # Calcul statistiques sur le Dataset
    def_image = statistique(DATASET_DIR, cfg, class_names, path_save, rapport) # type: ignore


    # --- Finalisation du Rapport ---------------------------------------------
    rp.finalisation_du_rapport(rapport, erreur, path_save)


    # Création d'un Dataset spécifique pour FiftyOne
    print()
    lauch_fifty = False  # valeur par défaut

    try:
        if not def_image:
            display.print("Dataset Ok", colors['ok'])
            display.print("Création du dataset pour FiftyOne", colors['titre'])

            dataset = create_dataset(DATASET_DIR, yaml_path=dataset_yaml)

            lauch_fifty = util.answer_yes_or_no("Voulez-vous lancer FiftyOne ?")

        else:
            display.print("Dataset non valide", colors['warning'])
            display.print(
                "Création d'un dataset d'anomalies pour FiftyOne et lancement de celui-ci",
                colors['titre']
            )

            dataset = create_dataset(DATASET_DIR, anomalies=def_image)
            lauch_fifty = True  # logique implicite : anomalies => lancement direct

    except ValueError:
        display.print(f"!!! Dataset non valide !!!\n{ct.BELL}", colors['error'])
        util.sortie_de_programme()
      
    print()
    # Lancement de l'interface FiftyOne
    if lauch_fifty :
        util.launch_fiftyone_interface(dataset) # type: ignore


    print()
    util.sortie_de_programme()

#==========================================================================================
if __name__ == "__main__":
    main()
