
#==========================================================================================

#-----------------------------------------------------------------------------------
# General Configuration
#-----------------------------------------------------------------------------------

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
# Detection / Evaluation Parameters
#-----------------------------------------------------------------------------------

# Intersection over Union (IoU) threshold (used for matching / evaluation)
IOU_THRESHOLD = 0.9

#-----------------------------------------------------------------------------------
# Bounding Box Constraints (YOLO format: normalized [0,1])
#-----------------------------------------------------------------------------------

# Minimum and maximum bounding box width/height (relative to image size)
MIN_BBOX = 1.922e-03  # 0.00192186
MAX_BBOX = 0.9995      # 0.9980782

# Minimum and maximum bounding box area (relative to image area)
MIN_BBOX_AREA = 1.26e-5  # MIN_BBOX ** 2
MAX_BBOX_AREA = 0.98   # MAX_BBOX ** 2     0.9902

#-----------------------------------------------------------------------------------
# Percentile Thresholds for Anomaly Detection
#-----------------------------------------------------------------------------------

# Percentile thresholds used to detect outliers
# Percentile calibration
PERCENTILE_WARNING = 80   # 80
PERCENTILE_ERROR   = 95   # 95

# Minimum guaranteed thresholds (never go below this)
MIN_BBOX_OVERFLOW_WARNING = 15.0   # 15.0
MIN_BBOX_OVERFLOW_ERROR   = 30.0   # 30.0


#-----------------------------------------------------------------------------------
# Dataset Constraints
#-----------------------------------------------------------------------------------

# Maximum number of classes (used for class ID validation)
NB_CLASSES = 100

# Worst images to display in the report (based on anomaly score)
MAX_WORST_IMAGES = 10

# Threshold (%) to consider a class as underrepresented
# Set to None to disable filtering
DOMINANT = 1
RARE = 0.25


RATIO_OK = 10
RATIO_WARNING = 50

ENTROPY_OK = 0.85
ENTROPY_WARNING = 0.65

SCORE_OK = 80
SCORE_WARNING = 50
