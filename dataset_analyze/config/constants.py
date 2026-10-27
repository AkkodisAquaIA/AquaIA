
from os import get_terminal_size

#==========================================================================================
# Logo
INFO_PROD = r"""
   ┌────────────────────────────────────┐
   │    Agence de l'Eau Adour Garonne   │
   │        Analyse d'un Dataset        │
   │                                    │
   │                by AKKODIS-Research │
   └────────────────────────────────────┘
"""

# Mode débogage
DEBUG_MOODE = False

NB_COLONNES = 2


# Chargement du dataset
# True  -> Chargement via fichier de config 
# False -> Saisie manuelle des chemins 
LOAD_DIR = True



# System bell sound (used to alert on errors)
BELL = "\a"

# Width of the display for console output (in characters)
DISPLAY_WIDTH = min(180, get_terminal_size().columns)


TITLE_FILL = "—"   

ECRAN = False
FICHIER = True


#-----------------------------------------------------------------------------------
# Data Loading Configuration
#-----------------------------------------------------------------------------------

# Supported image file extensions
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


#-----------------------------------------------------------------------------------
# Display & UI Settings
#-----------------------------------------------------------------------------------

# Nombre maximal de classe
NB_CLASSES = 400


# imbalance
PROFILES = {
        "strict":   (0.05, 0.20, "strict"),
        "normal":   (0.10, 0.30, "normal"),
        "tolerant": (0.15, 0.40, "tolerant"),
    }


# Max number of worst images displayed in report
MAX_WORST_IMAGES = 5

# Max number of images displayed in summary (if too many, only a sample is shown)
MAX_IMAGES_AFFICHEES = 30

# Width of tqdm progress bars
TQDM_NCOLS = 150

# Number of items displayed per line in reports
N_PER_LINE = 3

# valeur du 'Bin' pour Bargraphe
BINS = 60


MENUS = {
    # Menu principal
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

    # Menu pour Augmentations de données
    'SYMETRIE': [
        "Symétrie Horizontale & Verticale",
        "Pas de symétrie",
        "Toutes",
        "Symétrie horizontale",
        "Symétrie verticale",
      ],
    'ROTATION': [
        "Rotation",
        "Pas de rotation",
        "Toutes",
        "Rotation simple",
        "Rotation + Flip horizontal",
        "Rotation + Flip horizontal + Contraste",
        "Rotation + Flip vertical",
        "Rotation + Flip vertical + Contraste",
      ],
    'ZOOM': [
        "Zoom",
        "Pas de zoom",
        "Tous",
        "Zoom simple",
        "Zoom + Contraste",
        "Zoom + Rotation",
        "Zoom + Rotation + Contraste",
        
      ],
    'DIVERS': [
        "Divers",
        "Pas de modification",
        "Toutes",
        "Contraste",
        "Bruit gaussien",
        "Contraste + Bruit gaussien",
      ],     

      # Menu infoq Images
      'SAUVEGARDE': [
        "Sauvegarde",
        "Bbox Rouge",
        "Bbox Verte",
        "Pas de Sauvegarde"
      ],

}
