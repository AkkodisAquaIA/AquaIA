
from config.config_loader import Config
from pathlib import Path

from tools import utility as util
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from config import constants as ct


def load_config():

    display = dc.DisplayColor()
    
    print()

    fichier = util.get_path_color("Fichier de configuration")
    fichier = Path(fichier)

    cfg = Config(fichier)

    try:
        config = {
            # --- General ---
            "SAVE_PLOT": cfg.get_bool("general", "SAVE_PLOT"),

            "SAVE_USER": cfg.SAVE_USER,
            "DATASET_DIR":cfg.DATA_USER,

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

            # # --- Scoring ---
            "PROFILES": cfg.get_str("scoring", "PROFILES")

        }

        # --- Vérification des répertoires ---
        dirs_to_check = {
            "DATASET_DIR": config["DATASET_DIR"],
        }

        for name, path_str in dirs_to_check.items():
            path = Path(path_str)

            if not path.exists():
                raise FileNotFoundError(
                    f"Le répertoire '{path.name}' est introuvable !!! {ct.BELL}"
                )

            if not path.is_dir():
                raise NotADirectoryError(
                    f"'{name}' n'est pas un répertoire valide : {path}"
                )


        display.print("Fichier de Paramètres valide",colors["ok"])
        return config, fichier
        
    except Exception as e:
        display.print("---- Erreur dans le fichier de CONFIGURATION ----", colors["error"])
        display.print("-----                                       -----", colors["error"])
        display.print(e, colors["error"]) # type: ignore
        display.print("-------------------------------------------------", colors["error"])

        util.sortie_de_programme()

