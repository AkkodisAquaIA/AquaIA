
from pathlib import Path
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout
import re

from config import constants as ct
from config.constants import DISPLAY_COLORS as colors
import tools.display_color as dc
from tools import utility as util

display = dc.DisplayColor()

#=========================================================================================)

def create_file_report(DATASET_DIR, path_save, nom, cfg):

    # display = dc.DisplayColor()

    liste_nom = DATASET_DIR.name
    new_name = util.horodatage(liste_nom)+'.txt'
    rapport: Path = path_save / new_name 
    timestamp = datetime.now().strftime("%Y-%m-%d à %H:%M:%S") 
   
    try:
        with open(rapport, "w", encoding="utf-8") as f:
            f.write(ct.INFO_PROD)
            f.write("—" * 120)
            f.write(f"\n - Nom du Dataset      : *** {liste_nom} ***")
            f.write(f"\n - Nom du fichier YAML : *** {nom} ***")
            f.write("\n")
            f.write(f"\n - Répertoire de sauvegarde : {path_save}")
            if not cfg["REPORT_MODE"] :
                f.write("\n  - Pas de sauvegarde des fichiers de défauts")
            if not cfg["SAVE_PLOT"] :
                f.write("\n  - Pas de sauvegarde des graphiques")
            f.write("\n\n")
            f.write("—" * 120)
            
    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])

    display.print(f"Fichier de Rapport : '{new_name}' create ", colors["ok"])

    return rapport

def suivi(texte, rapport, type=""):

    tag = "Commencé" if type == "D" else "Terminé" 
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



def ecrire_sortie_dans_rapport(rapport, fonction, *args, **kwargs):

    ansi_escape = re.compile(
        r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
        )


    display = dc.DisplayColor()

    buffer = StringIO()

    with redirect_stdout(buffer):
        fonction(*args, **kwargs)

    contenu = buffer.getvalue()
    contenu = ansi_escape.sub('', contenu)

    timestamp = datetime.now().strftime("%Y-%m-%d à %H:%M")

    try:
        with open(rapport, "a", encoding="utf-8") as f:
            f.write(contenu)
    except FileNotFoundError:
            display.print(f"Impossible de sauvegarder : {rapport}", colors['error'])


