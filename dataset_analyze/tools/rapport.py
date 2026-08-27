
from pathlib import Path
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout
import re
import shutil

from config import constants as ct
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from tools import utility as util

display = dc.DisplayColor()

#========================================================================================

def lige_de_liaison(file):
    file.write("\n")
    file.write("—" * ct.DISPLAY_WIDTH)
    file.write("\n")

def titre_rapport(file, texte):
    file.write(f" {texte} ".center(ct.DISPLAY_WIDTH, "—"))
    file.write("\n")

def suivi(texte, rapport, type=""):

    tag = "Commencée" if type == "D" else "Terminée " 
    # Récupération de l'heure de la fin de l'analyse
    timestamp = datetime.now().strftime("%Y-%m-%d à %H:%M:%S")

    # Ecriture dans le fichier rapport
    try:
        with open(rapport, "a", encoding="utf-8") as f:

            f.write(f"\n  - {texte}  : {tag} le {timestamp}")
            if type != "D":
                 f.write("\n")

    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])

    return datetime.now()

def temps_de_traitement(debut, fin, rapport):
     
    difference = fin - debut

    total_sec = int(difference.total_seconds())

    heures = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secondes = total_sec % 60

    try:
        with open(rapport, "a", encoding="utf-8") as f:
            f.write(f"  -  : {heures:02d}:{minutes:02d}:{secondes:02d}\n")
            
    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])

def repport_def(texte, rapport):

    # Récupération de l'heure de la fin de l'analyse
    timestamp = datetime.now().strftime("%Y-%m-%d à %H:%M:%S")

    try:
        with open(rapport, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("*" * 80 )
            f.write("\n  Programme arrêté pour cause d'erreur :")
            f.write(f"\n  - {texte}  :  le {timestamp}\n")
            f.write("*" * 80 )
            f.write("\n")

    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])

#----------------------------------------------------------------------------------------

def afficher_bbox_erreurs_compact(
    bbox_erreurs: dict[str, list[str]],
    largeur_max_ligne: int | None = None
    ) -> None:
        """
        Display bounding box errors grouped by category.

        The display automatically adapts to terminal width
        without cutting file names.

        Args:
            bbox_erreurs (Dict[str, List[str]]):
                Dictionary mapping error categories to image path lists.
            largeur_max_ligne (Optional[int]):
                Maximum line width. If None, half terminal width is used.
        """


        display = dc.DisplayColor()

        if not bbox_erreurs:
            display.print("Pas d'erreurs de boîte englobante détectées.", colors["ok"])
            return

        if largeur_max_ligne is None:
            largeur_max_ligne = shutil.get_terminal_size().columns // 2

        categorie_max_len: int = max(len(cat) for cat in bbox_erreurs.keys())
        indent: str = " " * (categorie_max_len + 3)
        separateur: str = " | "

        display.print(" --- Erreurs détectées:", colors["error"])

        for categorie, chemins in bbox_erreurs.items():
            if not chemins:
                continue

            display.print(
                f"{categorie.capitalize().ljust(categorie_max_len)} "
                f"({len(chemins)} images):",
                colors["error"]
            )

            nb_total = len(chemins)
            # noms_images = [Path(chemin).name for chemin in chemins[:ct.MAX_IMAGES_AFFICHEES]]
            noms_images = [Path(chemin).name for chemin in chemins]

            ligne: str = ""

            for nom in noms_images:
                element: str = nom if not ligne else separateur + nom

                # Check if adding the element exceeds allowed width
                if len(indent) + len(ligne) + len(element) > largeur_max_ligne:
                    print(f"{indent}{ligne}")
                    ligne = nom
                else:
                    ligne += element

            if ligne:
                print(f"{indent}{ligne}")

            # if nb_total > ct.MAX_IMAGES_AFFICHEES:
            #     display.print(
            #         f"        ... {nb_total - ct.MAX_IMAGES_AFFICHEES} additional images not shown",
            #         colors["warning"]
            #     )

            print()

#========================================================================================

def create_file_report(DATASET_DIR, path_save, nom, cfg):

    liste_nom = DATASET_DIR.name
    new_name = util.horodatage(liste_nom)+'.txt'
    rapport: Path = path_save / new_name 
    timestamp = datetime.now().strftime("%Y-%m-%d à %H:%M:%S") 
   
    try:
        with open(rapport, "w", encoding="utf-8") as f:
            f.write(ct.INFO_PROD)
            f.write("—" * ct.DISPLAY_WIDTH)
            f.write(f"\n - Nom du Dataset      : *** {liste_nom} ***")
            f.write(f"\n - Nom du fichier YAML : *** {nom} ***")
            f.write("\n")
            f.write(f"\n - Répertoire de sauvegarde : {path_save}")
            if not cfg["SAVE_PLOT"] :
                f.write("\n  - Pas de sauvegarde des graphiques")
            f.write("\n\n")
            f.write("—" * ct.DISPLAY_WIDTH)

    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])

    display.print(f"Fichier de Rapport : '{new_name}' create ", colors["ok"])

    return rapport
           
def ecrire_sortie_dans_rapport(rapport, fonction, *args, **kwargs):

    ansi_escape = re.compile(
        r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
        )

    buffer = StringIO()

    with redirect_stdout(buffer):
        fonction(*args, **kwargs)

    contenu = buffer.getvalue()
    contenu = ansi_escape.sub('', contenu)

    try:
        with open(rapport, "a", encoding="utf-8") as f:
            f.write(contenu)
            f.write("\n")
    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])

def finalisation_du_rapport(rapport, erreur, path_save):
         
    # --- Finalisation du Rapport ---------------------------------------------
    # --- Ecriture des défauts de conformitè ----------------------------------
    with open(rapport, "a", encoding="utf-8") as destination:
        titre_rapport(destination,"+ . . + . . +")
        titre_rapport(destination,"Listes complétes des Erreurs de Conformité & Anomalies")
        destination.write("\n")
        titre_rapport(destination,"Erreurs de Conformité")
        destination.write("\n")

    ecrire_sortie_dans_rapport(
            rapport,
            afficher_bbox_erreurs_compact,
            erreur
        )

    # --- Ecriture des défauts statistiques -----------------------------------
    repertoire = Path(path_save)

    # liste des fichiers 'erreurs_dataset' triés du plus récent au plus ancien 
    fichiers = sorted(
        repertoire.glob("*def_erreurs_dataset*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True 
    )
    
    # Lecture du fichier des erreurs le plus récent
    with open(fichiers[0], "r", encoding="utf-8") as source:
        contenu = source.read()

    # copie de celui-ci dans le rapport général
    with open(rapport, "a", encoding="utf-8") as destination:

        titre_rapport(destination,"Liste détaillée des Anomalies")
        destination.write("\n")
        destination.write(contenu)

        titre_rapport(destination,"Fin de Rapport")
