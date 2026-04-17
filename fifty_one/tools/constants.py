
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
# General Configuration
#-----------------------------------------------------------------------------------

# Simulation & test mode
# True  -> Enables test/simulation behavior
# False -> Runs in normal production mode
TEST_MODE = True

# Random seed for reproducibility (ensures consistent results across runs)
SEED = 42

# System bell sound (used to alert on errors)
BELL = "\a"

# Debug mode
# True  -> Full error messages (detailed stack traces)
# False -> Short and user-friendly error messages
DEBUG_MODE: bool = False


#-----------------------------------------------------------------------------------
# Reporting & Paths
#-----------------------------------------------------------------------------------

# Enable/disable report generation (e.g., anomaly logs)
REPORT_MODE: bool = True

# Enable/disable save plot
SAVE_PLOT: bool = True

# Root path to the dataset (user-specific)
PATH_USER = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Data/"


#-----------------------------------------------------------------------------------
# Data Loading Configuration
#-----------------------------------------------------------------------------------

# Batch size for data loading / processing
BATCH_SIZE = 64

# Number of parallel workers for data loading
NUM_WORKERS = 8

# Supported image file extensions
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

#-----------------------------------------------------------------------------------
# Detection / Evaluation Parameters
#-----------------------------------------------------------------------------------

# Intersection over Union (IoU) threshold (used for matching / evaluation)
IOU_THRESHOLD = 0.9

#-----------------------------------------------------------------------------------
# Bounding Box Constraints (YOLO format: normalized [0,1])
#-----------------------------------------------------------------------------------

# Minimum and maximum bounding box width/height (relative to image size)
MIN_BBOX = 0.0   # 0.00192186
MAX_BBOX = 1.0  # 0.99807814

# Minimum and maximum bounding box area (relative to image area)
MIN_BBOX_AREA = 0.0 # MIN_BBOX ** 2
MAX_BBOX_AREA = 1.0 #MAX_BBOX ** 2

#-----------------------------------------------------------------------------------
# Percentile Thresholds for Anomaly Detection
#-----------------------------------------------------------------------------------

# Percentile thresholds used to detect outliers
PERCENTILE_WARNING = 90   # Warning threshold
PERCENTILE_ERROR   = 99   # Critical threshold

#-----------------------------------------------------------------------------------
# Bounding Box Overflow Tolerance (in %)
#-----------------------------------------------------------------------------------

# Maximum tolerated overflow before raising a warning
# (bounding box partially outside image)
BBOX_OVERFLOW_WARNING = 2.0 # 32.150  # Typical range: 10–20%

# Maximum tolerated overflow before marking as error
# (bounding box significantly outside image)
BBOX_OVERFLOW_ERROR   = 5.0  # 32.155  # Typical range: 30–35%


#-----------------------------------------------------------------------------------
# Dataset Constraints
#-----------------------------------------------------------------------------------

# Maximum number of classes (used for class ID validation)
NB_CLASSES = 100

# Threshold (%) to consider a class as underrepresented
# Set to None to disable filtering
DOMINANT = 1
RARE = 0.50


RATIO_OK = 10
RATIO_WARNING = 50

ENTROPY_OK = 0.85
ENTROPY_WARNING = 0.65

SCORE_OK = 80
SCORE_WARNING = 50

#-----------------------------------------------------------------------------------
# Display & UI Settings
#-----------------------------------------------------------------------------------

# Width of tqdm progress bars
TQDM_NCOLS = 150

# Number of items displayed per line in reports
N_PER_LINE = 5

#-----------------------------------------------------------------------------------
# Display Colors & Prefixes (RGB + label prefix)
#-----------------------------------------------------------------------------------

DISPLAY_COLORS = {
    # Standard statuses
    'error':   (204,  51,   0, "[X] "),    # Red           → critical error
    'warning': (204, 204,   0, "[!] "),    # Yellow/Orange → warning
    'input':   (153, 204,  51, "[?] "),    # Light green   → user input
    'ok':      ( 51, 153,   0, "[√] "),    # Green         → success
    'goodbye': (255,  16, 240, "[<3] "),   # Purple        → exit message
    'info':    ( 51, 102, 255, "[I] "),    # Blue          → informational message
    'wait':    (255, 153,  51, "[...] "),  # Orange        → processing/wait

    # Aqua-IA themed colors (blue-green palette)
    'aqua_light': (102, 255, 204, "[~] "), # Light turquoise
    'aqua':       (  0, 204, 153, "[~] "), # Standard teal
    'aqua_dark':  (  0, 102, 102, "[~] "), # Dark blue-green
}
