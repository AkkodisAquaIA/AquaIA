
from os import get_terminal_size

#==========================================================================================
# Logo
INFO_PROD = r"""
   ┌────────────────────────────────────────────┐
   │            Analyse d'un Dataset            │
   │                                            │    
   │                        by AKKODIS-Research │
   └────────────────────────────────────────────┘
"""

# Chargement du dataset
# True  -> Chargement via fichier de config 
# False -> Saisie manuelle des chemins 
LOAD_DIR = True

# System bell sound (used to alert on errors)
BELL = "\a"

# Width of the display for console output (in characters)
DISPLAY_WIDTH = min(180, get_terminal_size().columns)



ECRAN = False
FICHIER = True


#-----------------------------------------------------------------------------------
# Data Loading Configuration
#-----------------------------------------------------------------------------------

# Supported image file extensions
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


#-----------------------------------------------------------------------------------
# Display & UI Settings
#-----------------------------------------------------------------------------------

# Nombre maximal de classe
NB_CLASSES = 100


# imbalance
PROFILES = {
        "strict":   (0.05, 0.20, "strict"),
        "normal":   (0.10, 0.30, "normal"),
        "tolerant": (0.15, 0.40, "tolerant"),
    }


# Max number of worst images displayed in report
MAX_WORST_IMAGES = 12

# Max number of images displayed in summary (if too many, only a sample is shown)
MAX_IMAGES_AFFICHEES = 30

# Width of tqdm progress bars
TQDM_NCOLS = 150

# Number of items displayed per line in reports
N_PER_LINE = 7

# valeur du 'Bin' pour Bargraphe
BINS = 60


MENUS = {
    'MAIN': [
        "Sélection des informations à afficher",
        "Information générales",
        "Information sur les classes",
        "Images par classe",
        "Taille des BBoxes",
        "Anomalies",
        "Métriques de déséquilibre des classes",
        "Lancement Fifty_One",
        "Sortie"    ],
}
