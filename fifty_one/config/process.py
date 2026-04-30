import sys
from config.config_loader import Config
from pathlib import Path

import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from config import constants as ct


def load_config():

    display = dc.DisplayColor()
    
    print()

    fichier = Path("aqua_ia_conf.ini")
    if not fichier.is_file():
        display.print(f"Le fichier de paramètre 'aqua_ia_conf.ini' est introuvable !!! {ct.BELL}", colors['error'])
        exit(1)

    cfg = Config("aqua_ia_conf.ini")

    try:
        config = {
            # --- General ---
            "REPORT_MODE": cfg.get_bool("general", "REPORT_MODE"),
            "SAVE_PLOT": cfg.get_bool("general", "SAVE_PLOT"),
            "VEC_FIELD": cfg.get_str("general", "VEC_FIELD"),

            "PATH_USER": cfg.PATH_USER,

            # --- Détection ---
            "IOU_THRESHOLD": cfg.get_float("detection", "IOU_THRESHOLD"),

            # --- Bbox ---
            "MIN_BBOX": cfg.get_float("bbox", "MIN_BBOX"),
            "MAX_BBOX": cfg.get_float("bbox", "MAX_BBOX"),

            "MIN_BBOX_AREA": cfg.get_float("bbox", "MIN_BBOX_AREA"),
            "MAX_BBOX_AREA": cfg.get_float("bbox", "MAX_BBOX_AREA"),

            # --- Percentile ---
            "PERCENTILE_WARNING": cfg.get_float("percentiles","PERCENTILE_WARNING"),
            "PERCENTILE_ERROR": cfg.get_float("percentiles","PERCENTILE_ERROR"),

            "MIN_BBOX_OVERFLOW_WARNING": cfg.get_float("percentiles","MIN_BBOX_OVERFLOW_WARNING"),
            "MIN_BBOX_OVERFLOW_ERROR": cfg.get_float("percentiles","MIN_BBOX_OVERFLOW_ERROR"),

            # --- Dataset ---
            "DOMINANT": cfg.get_float("dataset","DOMINANT"),
            "RARE": cfg.get_float("dataset","RARE"),

            # --- Scoring ---
            "RATIO_OK": cfg.get_float("scoring","RATIO_OK"),
            "RATIO_WARNING": cfg.get_float("scoring","RATIO_WARNING"),

            "ENTROPY_OK": cfg.get_float("scoring","ENTROPY_OK"),
            "ENTROPY_WARNING": cfg.get_float("scoring","ENTROPY_WARNING"),

            "SCORE_OK": cfg.get_float("scoring","ENTROPY_OK"),
            "SCORE_WARNING": cfg.get_float("scoring","ENTROPY_WARNING"),

        }

        display.print("Fichier de Paramètres valide",colors["ok"])
        return config

    except Exception as e:
        display.print("---- ERREUR DE CONFIGURATION ----", colors["error"])
        display.print("-----                       ------", colors["error"])
        display.print(e, colors["error"]) # type: ignore
        display.print("----------------------------------", colors["error"])
        print()
        sys.exit(1)
