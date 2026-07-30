
import os
import re
import random
from tqdm import tqdm

from tools import system as syst
if syst.est_linux():
    os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")

from pathlib import Path
import yaml
import shutil
from datetime import datetime
import fiftyone as fo

# --- Bibliothèques utilisateur ---------------------------------------------------------
from config import valid_conf as vc
from config.process import load_config
from config import constants as ct

from bboxes import bboxes as bb
from statistics_yolo import dataset_statistics_yolo as ds

from tools import menu_color as menu_c
from tools import utility as util
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
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

def creation_file_yaml(DATASET_DIR ):
    labels_dir = DATASET_DIR / "labels" / "train2017"

    classes = set()

    fault_file = []

    txt_files = list(labels_dir.glob("*.txt"))

    for txt_file in tqdm(txt_files, 
                        desc="Analyse des labels",
                        unit=" labels",
                        ncols=100,
                        position=0):
        try : 
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        class_id = int(line.split()[0])
                        classes.add(class_id)
                    except (ValueError, IndexError):
                        pass

        except UnicodeDecodeError:
            
            tag = f" - Fichier ignoré : {txt_file.name} : (contenu non UTF-8 ou fichier non texte)" 
            fault_file.append(tag)

        except Exception as e:

            tag = f" - Fichier ignoré : {txt_file.name} ({e})" 
            fault_file.append(tag)


    if len(fault_file) != 0 :
        print()
        for f in fault_file :        

            display.print(f,colors['warning'])
        print()

    if not classes:
        display.print(
            "Aucune classe détectée dans les fichiers labels.",
            colors['error']
        )
    else:
        nc = max(classes) + 1

        contenu_defaut = {
            "path": ".",
            "train": "images/train2017",
            "val": "images/train2017",  # à adapter si nécessaire
            "nc": nc,
            "names": [f"classe_{i}" for i in range(nc)]
        }

        dataset_yaml_ = DATASET_DIR / "default.yaml"

        with open(dataset_yaml_, "w", encoding="utf-8") as f:
            yaml.dump(
                contenu_defaut,
                f,
                allow_unicode=True,
                sort_keys=False
            )

        display.print(
            f"Fichier créé avec {nc} classes détectées",
            colors['ok']
        )
        print()

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
        if util.answer_yes_or_no("Voulez-vous voir le graphe"):
            display.print("Attente fermeture du graphe", colors['wait'])
            gr.bbox_overflow(cfg, outside_ratios, BBOX_OVERFLOW_WARNING, BBOX_OVERFLOW_ERROR) 

    return   results

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


#----------------------------------------------------------------------------------------
def main():
    # ================= CONFIG =================
    # Déactivation de la barre de progression de FiftyOne
    fo.config.show_progress_bars = False 

    # Efface l'écran avant de commencer
    syst.clear_screen()

    # --- Affichage Logo et information générale ----------------------------------------
    try:
        # Choix aléatoire d'un logo
        logo = choix_logo()

        # affichage du logo
        if syst.est_windows():
            lw.splash_screen_circle(logo) 
        else:
            ll.splash_screen_circle(logo)
    except FileNotFoundError :
        print()
        tag = "Le répertoire 'logo' est introuvable ou vide"
        display.print(tag, colors['warning'])   

    tag = display.colored(ct.INFO_PROD,
                        colors['aqua_light'],
                        pref=False,
                        bold=True
                        )

    print(tag)


    # Chargement & Vérification du fichier de Paramètrage
    try: 
        cfg = load_config() 
    except Exception as e:
        display.print(f"Erreur de chargement du fichier de configuration :\n   {e}", colors['error'])
        print()
        util.sortie_de_programme()

    
    print()
    vc.controle(cfg) # type: ignore
    print()
    
    # Contrôle répertoire de sauvegarde
    path_save = Path(cfg["SAVE_USER"]) # type: ignore

    if not path_save.exists():
        fault_path = path_save
        path_save = Path.cwd() / "Report"
        path_save.mkdir(parents=True, exist_ok=True)

        display.print(
            f"Répertoire de sauvegarde '{fault_path}' invalide !!! {ct.BELL}",
            colors["error"]
        )
        color = colors["warning"]
    else:
        color = colors["ok"]

    display.print(f"Répertoire de sauvegarde utilisé : {path_save}", color)


    # Sauvegarde du fichier de configuration
    # Horodatage au format AAAAMMJJ_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source = Path("aqua_ia_conf.ini")
    destination_dir = Path(path_save)
    destination = destination_dir / f"{timestamp}_{source.name}"
    shutil.copy2(source, destination)

    # Affichage des états des modes de sauvegarde
    # Graphe mode handling
    print()
    util.afficher_mode("Sauvegarde des Graphiques :", cfg["SAVE_PLOT"]) # type: ignore


    # Chargement du Répertoire du Dataset
    if ct.LOAD_DIR :
        # Via le fichier de configuration
        DATASET_DIR : Path = Path(cfg["DATASET_DIR"]) # type: ignore
    else :
        # Saisie manuel
        DATASET_DIR = util.get_path_color("Entrez le chemin du Dataset")
    

    #  ------------------------------------------------------------------------------------------
    # Démarrage du traitement 
    display.header_title("Démarrage du traitement",
                        colors['aqua_light'],
                        Path(DATASET_DIR.name), # type: ignore
                        bold=True,
                        ) # type: ignore

    # Chargement des noms de classes pour les stats
    DATASET_DIR  = Path(DATASET_DIR )

    # Gestion des fichiers .yaml
    display.titre("Gestion des fichiers '.yaml'", colors['aqua']) # type: ignore

    # Recherche des fichiers .yaml
    yaml_files = list(DATASET_DIR.glob("*.yaml"))
    dataset_yaml_ = ""

    # Gestion des fichiers .yaml
    # Si aucun fichier .yaml trouvé -> Création d'un fichier '.yaml' par défaut
    if len(yaml_files) == 0:
  
        display.print(
            f"Aucun fichier '.yaml' trouvé dans :\n   {DATASET_DIR}\n - Création d'un fichier par défaut.",
            colors['warning']
        )

        creation_file_yaml(DATASET_DIR)
        dataset_yaml_= DATASET_DIR / "default.yaml"

    # Un seul fichier -> on l'utilise
    elif len(yaml_files) == 1:
        dataset_yaml_ =  yaml_files[0]

    # Sélection d'un fichier parmi une liste
    else:

        file_name = [f.stem for f in yaml_files]  
        menu_items = ["'.YAML' disponible"] + file_name 
        mm = menu_c.Menu("Dynamic",             # Menu créé dynamiquement
                        menu_items,             # Liste des items du menu
                        style= 'rounds',        # style du menu 
                        theme = menu_c.AQUA_IA) # Theme du menu  
     
        mm.display_menu()
        choice = mm.selection()
        dataset_yaml_ = yaml_files[choice - 1]

    # Chargement
    nom = Path(dataset_yaml_).stem

    display.print(f"fichier '.yaml' utilisé : {nom}", colors['ok'])
    print()
    dataset_yaml =  DATASET_DIR / dataset_yaml_

    # Création d'un fichier de rapport
    rapport = rp.create_file_report(DATASET_DIR, path_save, nom, cfg) # type: ignore


    # Validation du fichier .yaml
    tag, yaml_ok = "", True
    try:
       class_names = load_class_names(dataset_yaml)
    except yaml.YAMLError as e:
        print()
        tag = f"le fichier '.yaml' sélectionné  n'est pas conforme\n  {e} "
        display.print(tag, colors['error'])
        yaml_ok = False
    except FileNotFoundError as e:
        print()
        tag = f" '{dataset_yaml}' est introuvable dans {DATASET_DIR}\n  {e} "
        display.print(tag, colors['error'])
        yaml_ok = False

    if not yaml_ok :
        rp.repport_def(tag, rapport)
        print()
        util.sortie_de_programme()


    # ---  Analyse du Dataset 
    ctrl_ok = False
    erreur, ctrl_ok = bb.validate_yolo_dataset_detailed(DATASET_DIR, path_save, rapport, cfg) # type: ignore
 
    # --- Analyse des résultats ---------------------------------------------------------
    print()
    
    # --- Datast sain -------------------------------------------------------------------
    if ctrl_ok:
        display.print("Analyse de la conformité terminée sans problème", colors['ok']) 

    # --- Gestion des erreurs de conformité ---------------------------------------------
    else:
        print()
        display.print(" !!! Des erreurs de conformité ont été trouvées !!!\n"
                      "  les fichiers seront déplacés vers les répertoires 'problems' de 'images' & 'labels' \n"
                      , colors['warning'])

        # Récupération des labels avec problèmes de conformité  
        liste_def(path_save)    

        # Déplacement des fichiers problèmatiques dans les répertoires 'problem'
        try:
            deplac_prob(DATASET_DIR, path_save, "labels_invalides")
            deplac_prob(DATASET_DIR, path_save, "labels_orphelins")
            deplac_prob(DATASET_DIR, path_save, "labels_vides")
            deplac_prob(DATASET_DIR, path_save, "labels_xxxx")

            deplac_prob(DATASET_DIR, path_save, "images_invalides")
            deplac_prob(DATASET_DIR, path_save, "images_sans_label")

            # Suppresion des fichiers d'erreurs
            sup_file_def(path_save)
            print()
        except FileNotFoundError:
            tag = f"   fichiers d'erreurs introuvables"
            display.print(f"{tag}\n", colors['error'])
            rp.repport_def(tag, rapport)
            util.sortie_de_programme()


    print()    
    # --- Calcul statistiques sur le Dataset ---------------------------------------------
    results = statistique(DATASET_DIR, cfg, class_names, path_save, rapport) # type: ignore

    anomalies = results["anomalies"]
    print()
    tag = 'avec problèmes' if anomalies else 'sans problème'
    display.print(f"Analyse statistique terminé {tag}",
                  colors['warning' if anomalies else 'ok' ]
                  )


    #--- Mise à Jour du Rapport de Sortie ------------------------------------------------
    rp.ecrire_sortie_dans_rapport(
        rapport,
        ds.afficher_dataset_statistics,
        ct.FICHIER,
        DATASET_DIR, 
        results,
        cfg,  # type: ignore
        dataset_yaml,
        class_names) # type: ignore


    # --- Finalisation du Rapport --------------------------------------------------------
    rp.finalisation_du_rapport(rapport, erreur, path_save)

    # Attent validation de l'opérateur
    print()
    util.waiting_any_key("Appuyez sur 'Enter' pour continuer ... ")

    # --- Visualisation des résultats sur Ecran ------------------------------------------
    ds.afficher_dataset_statistics(ct.ECRAN,
                                   DATASET_DIR, 
                                   results, cfg,  # type: ignore
                                   dataset_yaml,
                                   class_names) # type: ignore


    print()
    util.sortie_de_programme()

#==========================================================================================
if __name__ == "__main__":
    main()
