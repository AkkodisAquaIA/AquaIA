
#==========================================================================================
# Logo
INFO_PROD = r"""
┌────────────────────────────────────────────┐
│  Indice macro-invertébrés par ADNe et IA   │  
│       Agence de l'Eau Adour Garonne        │
│                                            │
│ Version : 0.05 Beta                        │
│                                            │  
│                        by AKKODIS-Research │
└────────────────────────────────────────────┘
"""

#-----------------------------------------------------------------------------------

# Mode Simulatio & Test
TEST_MODE = True

# Random seed for reproducibility
SEED = 42

# Sound system for Error.
BELL = "\a"

# Mode debug
"""
True  : The full error message is displayed.
False : The error message is displayed in a concise format.
"""
DEBUG_MODE = False


# Sound system for Error.
BELL = "\a"

# required imports 
REPORT_MODE  = True
PATH_USER = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Data/coco128"

# Constants and configuration for the FiftyOne project
BATCH_SIZE = 64

# Number of workers for data loading
NUM_WORKERS = 8

# supported image extensions
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


#   
IOU_THRESHOLD = 0.9

# Bounding box size limits (relative to image dimensions) 
MIN_BBOX =   0.00192186
MAX_BBOX =   1

# Bounding box area limits (relative to image area)
MIN_BBOX_AREA = MIN_BBOX ** 2 
MAX_BBOX_AREA = MAX_BBOX ** 2


# percentiles erreur et warnig en %
PERCILE_WARNING = 90
PERCILE_ERROR   = 99

# Valeur par défaut
# Tolérance maximale avant d’émettre un warning (en %)
BBOX_OVERFLOW_WARNING = 10  # de 10 à 20 %    
# Tolérance maximale absolue avant de bloquer l'image (en %)
BBOX_OVERFLOW_ERROR   = 30   # de 30 à 35%   



# Maximum number of classes (for class ID validation)
nb_classes = 100

# progress bar width
TQDM_NCOLS = 150

# Number of items to display per line in error reports
n_per_line = 5



#----------------------------------
# Code colors & prefixes for display
#----------------------------------
DISPLAY_COLORS = {
    'error':   (204,  51,  0,  "[X] "),   # Red
    'warning': (204, 204,  0,  "[!] "),   # Orange
    'input':   (153, 204, 51,  "[?] "),   # Light Green
    'ok':      ( 51, 153,  0,  "[√] "),   # Green
    'goodbye': (255,  16, 240, "[<3] "),  # Purple
    'info':    ( 51, 102, 255, "[I] "),   # Blue
    'wait':    (255, 153, 51,  "[...] "),  # Orange
    # Nuances bleu-vert pour Aqua-IA
    'aqua_light': (102, 255, 204, "[~] "), # Turquoise clair
    'aqua':       (  0, 204, 153, "[~] "), # Teal standard
    'aqua_dark':  (  0, 102, 102, "[~] "), # Bleu-vert foncé
}


