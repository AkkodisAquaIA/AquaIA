
#==========================================================================================
# Logo
logo = r"""
Version : 05 Beta                                                                                                             
             
"""

#-----------------------------------------------------------------------------------
# Mode debug
"""
True  : The full error message is displayed.
False : The error message is displayed in a concise format.
"""
DEBUG_MODE = False


# required imports 
REPORT_MODE  = False

# Constants and configuration for the FiftyOne project
BATCH_SIZE = 64

# Number of workers for data loading
NUM_WORKERS = 8

# Random seed for reproducibility
SEED = 42

# supported image extensions
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# Bounding box size limits (relative to image dimensions) 
MIN_BBOX = 0.001  
MAX_BBOX = 1.0

# Bounding box area limits (relative to image area)
MIN_BBOX_AREA = 0.001
MAX_BBOX_AREA = 1

# Tolérance maximale avant d’émettre un warning (en %)
BBOX_OVERFLOW_WARNING = 2       # 2 % → warning
# Tolérance maximale absolue avant de bloquer l'image (en %)
BBOX_OVERFLOW_ERROR   = 4       # 4 % → erreur

# Maximum number of classes (for class ID validation)
nb_classes = 100

# progress bar width
TQDM_NCOLS = 150

# Number of items to display per line in error reports
n_per_line = 5

# # Tolérance pour les bbox hors limites en pourcentage
# threshold_bounding_box = 0.5   

# Sound system for Error.
BELL = "\a"


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


