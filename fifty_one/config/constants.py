#==========================================================================================
# Logo
INFO_PROD = r"""
   ┌────────────────────────────────────────────┐
   │  Indice macro-invertébrés par ADNe et IA   │  
   │       Agence de l'Eau Adour Garonne        │
   │                                            │
   │            Analyse d'un Dataset            │
   │                                            │   
   │ Version : 1.00                             │
   │                                            │ 
   │                        by AKKODIS-Research │
   └────────────────────────────────────────────┘
"""


# # Random seed for reproducibility (ensures consistent results across runs)
# SEED = 0

# Chargement du dataset
# True  -> Chargement via fichier de config 
# False -> Saisie manuelle des chemins 
LOAD_DIR = True

# System bell sound (used to alert on errors)
BELL = "\a"


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

# Max number of worst images displayed in report
MAX_WORST_IMAGES = 15

# Max number of images displayed in summary (if too many, only a sample is shown)
MAX_IMAGES_AFFICHEES = 30

# Width of tqdm progress bars
TQDM_NCOLS = 150

# Number of items displayed per line in reports
N_PER_LINE = 5

# valeur du 'Bin' pour Bargraphe
BINS = 60

#-----------------------------------------------------------------------------------
# Display Colors & Prefixes (RGB + label prefix)
#-----------------------------------------------------------------------------------

DISPLAY_COLORS = {
    # Standard statuses
    'error':   (204,  51,   0, "[X] "),    # Red           → critical error
    'warning': (204, 204,   0, "[!] "),    # Yellow/Orange → warning
    'input':   (153, 204,  51, "[?] "),    # Light green   → user input
    'ok':      ( 51, 153,   0, "[√] "),    # Green         → success
    'info':    ( 51, 102, 255, "[I] "),    # Blue          → informational message
    'wait':    (255, 153,  51, "[...] "),  # Orange        → processing/wait
    'goodbye': (255,  16, 240, "[<3] "),   # Purple        → exit message

    # Custom prefixes for specific message types
    'titre':   ( 0,  204, 153, "T"),        # Standard teal → ———— Titre ————

    # Aqua-IA themed colors (blue-green palette)
    'aqua_light': (102, 255, 204, "[~] "), # Light turquoise
    'aqua':       (  0, 204, 153, "[~] "), # Standard teal
    'aqua_dark':  (  0, 102, 102, "[~] "), # Dark blue-green
}


MENUS = {
    'MAIN': [
        "Sélection des informations à afficher",
        "Information générales",
        "Information sur les classes",
        "Images par classe",
        "Taille des BBoxes",
        "Anomalies",
        "Métriques de déséquilibre des classes",
        "Sortie"    ],
}


# symbols for frame creation
PATTERN ={
"double" : [".","╔", "╦", "╗",
                "╠", "╬", "╣",
                "╚", "╩", "╝",
            "═", "║"
           ],
"simple" : [".","┌", "┬", "┐",
                "├", "┼", "┤",
                "└", "┴", "┘",
            "─", "│"
           ],
"rounds" : [".","╭", "┬", "╮",
                "├", "┼", "┤",
                "╰", "┴", "╯",
            "─", "│"
           ],
"heavy": [".","┏", "┳", "┓",   
              "┣", "╋", "┫",   
              "┗", "┻", "┛",   
        "━", "┃"         
        ],           
"Unicode" : [".","┏", "┯", "┓",
                 "┣", "┿", "┫",
                 "┗", "┷", "┛",
        "━", "┃","│"
        ],
"ASCII" : ["." ,"+" , "+" , "+",
               "+", "+", "+",
               "+", "+", "+",
           "─", "│"
           ],
    }

