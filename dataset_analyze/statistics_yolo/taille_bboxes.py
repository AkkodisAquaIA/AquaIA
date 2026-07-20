
import numpy as np

from config import constants as ct
from tools import utility as util
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from tools import graphe as gr

display = dc.DisplayColor()

#==============================================================================

def taille_bboxes(mode_aff, bbox_areas, cfg): 

    mode_affichage = mode_aff[0]
    etat_dataset = mode_aff[1]
    nom_dataset = mode_aff[2]


    display.header_title("(4) Distribution de la taille des BBoxes", colors[etat_dataset], nom_dataset)

    nb_bins = ct.BINS

    counts, edges = np.histogram(bbox_areas, bins=nb_bins)
    y_max = counts.max()

    print(f" - Nombre maximum d'occurrences : {util.format_nombre(y_max)}")
    print()

    y_max_affichage = y_max

    while True:

        display.print("Attente fermeture du graphe", colors['wait'])
        gr.histogram_taille_bbox(
            bbox_areas,
            "Distribution des tailles de BBox",
            "Aire bbox",
            "Nombre",
            cfg,
            y_max_affichage,
        )

        if not util.answer_yes_or_no("Voulez-vous modifier la valeur de 'y_max'"):
            break

        y_max_affichage = int(
            util.input_value("Entrer une valeur")
        )
