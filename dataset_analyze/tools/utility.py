import os
import sys
import threading
import time
import numpy as np
import platform
from colorama import Fore, Style
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Tuple

import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from config import constants as ct

ColorSpec = Tuple[int, int, int, str]

#=====================================================================================================

class DatasetStats(TypedDict):
    images: int
    labels: int
    bounding_boxes: int
    bbox_width_mean: float
    bbox_height_mean: float
    bbox_area_mean: float
    bbox_width_min: float
    bbox_width_max: float
    bbox_height_min: float
    bbox_height_max: float

class Anomaly(TypedDict, total=False):
    type: str
    image: str
    outside_ratio_pct: float
    class_id: int

class ValidationResults(TypedDict):
    stats: DatasetStats
    class_distribution: dict[int, int]
    anomalies: list[Anomaly]

class MiniProgressBar:
    """
    Lightweight animated text progress bar displayed on a single console line.
    """

    def __init__(self, message: str = "Loading", width: int = 20) -> None:
        self.message: str = message
        self.width: int = width
        self.running: bool = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(
            target=self._animate,
            daemon=True
        )
        self.thread.start()

    def _animate(self) -> None:
        """Internal animation loop."""

        if platform.system().lower() == "windows":
            spinner = "|/-\\"
        else:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        
        i: int = 0
        progress: int = 0

        while self.running:
            fill: int = progress % (self.width + 1)
            bar: str = "#" * fill + "." * (self.width - fill)
            spin: str = spinner[i % len(spinner)]

            sys.stdout.write(f"\r{self.message} [{bar}] {spin}")
            sys.stdout.flush()

            progress += 1
            i += 1
            time.sleep(0.1)

    def stop(self) -> None:
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=1)

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

class TablePrinter:

    def __init__(self, columns):
        """
        columns = liste de dictionnaires :
        [
            {"title": "Mean", "width": 11, "align": ">",},
            ...
        ]
        align : "<" gauche | ">" droite | "^" centre
        """
        self.columns = columns

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _line(self, left, mid, right):
        line = left
        for i, col in enumerate(self.columns):
            line += "─" * col["width"]
            line += mid if i < len(self.columns) - 1 else right
        print(line)

    def _format_cell(self, text, width, align):
        return f"{text:{align}{width}}"

    def _color_if(self, text, width, align, ok):
        formatted = self._format_cell(text, width, align)
        if ok:
            return formatted
        return f"{Fore.RED}{Style.BRIGHT}{formatted}{Style.RESET_ALL}"

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def header(self):
        self._line("┌", "┬", "┐")

        row = "│"
        for col in self.columns:
            row += self._format_cell(
                col["title"],
                col["width"],
                "^"
            ) + "│"
        print(row)

        self._line("├", "┼", "┤")

    def row(self, values):
        """
        values = liste de tuples :
        [
            ("Width", True),
            ("0.1523", True),
            ("0.0152", True),
            ("0.1000", condition_ok),
        ]
        condition_ok → bool (True = normal, False = rouge)
        """
        row = "│"
        for (value, ok), col in zip(values, self.columns):
            cell = self._color_if(
                str(value),
                col["width"],
                col["align"],
                ok
            )
            row += cell + "│"
        print(row)

    def footer(self):
        self._line("└", "┴", "┘")

#------------------------------------------------------------------------------------------

# TODO A suprimer
def quoi(valeur):
    print(f"*-------------------\n {valeur} \n-------------------*")


def get_dataset_paths(dataset_dir, split="train2017"):
    dataset_dir = Path(dataset_dir)

    images_dir = dataset_dir / "images"   / split
    labels_dir = dataset_dir / "labels"   / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir introuvable : {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir introuvable : {labels_dir}")

    return images_dir, labels_dir

def horodatage(file_name: str, defaut="") -> str:

    # Generate a timestamp (format: YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 

    # Extract filename without extension and file extension
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix

    # Build a new filename including the timestamp
    filename_with_timestamp = f"{timestamp}_{defaut}_{stem}_{suffix}"

    return filename_with_timestamp

def calibrer_seuils_overflow(results : dict,
                             warning_percentile : float,
                             error_percentile : float,
                             min_warning : float,
                             min_error : float)-> dict[str, float]:
    """
    Automatically calibrate bounding box overflow thresholds based on anomaly statistics.

    This function computes warning and error thresholds for bounding box overflow
    using percentile values derived from detected anomalies. It extracts the
    `outside_ratio_pct` values from the provided results and determines threshold
    levels based on the specified percentiles. Minimum threshold values are enforced
    to guarantee baseline sensitivity.

    If no overflow anomalies are found, the function directly returns the provided
    minimum thresholds.

    Args:
        results (dict):
            A dictionary containing anomaly detection results. It is expected
            to include a key `"anomalies"` mapped to a list of dictionaries,
            each potentially containing an `"outside_ratio_pct"` value
            representing the percentage of bounding box overflow.

        warning_percentile (float):
            The percentile used to compute the warning threshold.

        error_percentile (float):
            The percentile used to compute the error threshold.

        min_warning (float):
            The minimum allowed value for the warning threshold.

        min_error (float):
            The minimum allowed value for the error threshold.

    Returns:
        dict[str, float]:
            A dictionary containing:
                - "BBOX_OVERFLOW_WARNING": Final warning threshold (percentage)
                - "BBOX_OVERFLOW_ERROR": Final error threshold (percentage)

    Notes:
        - The function ensures that the error threshold is strictly greater
          than the warning threshold. If necessary, a small safety margin
          is added to the error threshold.
        - If no valid `outside_ratio_pct` values are found, the minimum
          thresholds are returned without percentile computation.
    """

    display = dc.DisplayColor()

    outside_ratios = [
        a['outside_ratio_pct']
        for a in results.get('anomalies', [])
        if 'outside_ratio_pct' in a
    ]

    if not outside_ratios:
        print()
        display.print(" Aucun dépassement détecté, utilisation des seuils minimaux.", colors['ok'])
        return {
            "BBOX_OVERFLOW_WARNING": min_warning,
            "BBOX_OVERFLOW_ERROR": min_error
        }

    # Percentile-based thresholds
    warning_calculated = np.percentile(outside_ratios, warning_percentile)
    error_calculated   = np.percentile(outside_ratios, error_percentile)

    # Apply minimum guarantee
    warning_final = max(warning_calculated, min_warning)
    error_final   = max(error_calculated, min_error)

    # Security: ensure error > warning
    if error_final <= warning_final:
        error_final = warning_final + 5.0  # small safety margin

    print()
    display.print("Calibration automatique des seuils:", colors["warning"])
    print(f"  - Avertissement ({warning_percentile} percentile): "
          f"{warning_calculated:.2f}% → Final: {warning_final:.2f}%")
    print(f"  - Erreur   ({error_percentile} percentile): "
          f"{error_calculated:.2f}% → Final: {error_final:.2f}%")

    return {
        "BBOX_OVERFLOW_WARNING": warning_final,
        "BBOX_OVERFLOW_ERROR": error_final
    }

def rgb_to_ansi(rgb: tuple[int, int, int]) -> str:
    """Convert RGB color to ANSI escape code."""
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"

def get_color(color_key: str) -> ColorSpec:
    return colors.get(color_key, colors["info"])

def get_path_color(prompt: str, color_key: str = 'input') -> Path:
    """
    Prompt the user for a valid filesystem path.

    The prompt is displayed in the specified color if available
    in the DISPLAY_COLORS dictionary.

    Args:
        prompt (str): Message displayed to the user.
        color_key (str): Key used to retrieve RGB color.

    Returns:
        Path: Validated path entered by the user.
    """
    display: dc.DisplayColor = dc.DisplayColor()
    color: tuple[int, int, int] = get_color(color_key) # type: ignore

    while True:
        input_color: str = rgb_to_ansi(color)
        colored_prompt: str = f"{input_color}[?] {prompt}: {Style.RESET_ALL}"

        path_input: str = input(colored_prompt).strip()

        if os.path.exists(path_input):
            return Path(path_input)

        error_text: str = f"Chemin invalide: {path_input}. Veuillez réessayer."
        display.print(error_text, colors['error'])

def selection(maxi) -> int:
  
    display = dc.DisplayColor()
    color = colors['input']

    while True:
        try:
            #  # Convert the input color from DISPLAY_COLORS to ANSI
            input_color = rgb_to_ansi(color[:3])
            # # Displays the prompt in color
            prompt = "Quel est votre choix "
            colored_select = f"{input_color}[?] {prompt}: {Style.RESET_ALL}"

            select = int(input(colored_select).strip())

            if 1 <= select <= maxi  :
                return select - 1
            text = f"Sélection invalide. Veuillez réessayer. {ct.BELL}"
            display.print(text, colors['error'])

        # Input is not a number
        except ValueError:
            text = (
                    f"Ce n'est pas un nombre . "
                    f"Réessayez! {ct.BELL}"
                )
            display.print(text, colors['error'])

def answer_yes_or_no(message: str, default=False, color_key: str = 'input') -> bool:
    """
    This function returns
        - True for yes, y, oui, o.
        - False for no, non n.
        - The input does not take uppercase letters into account."
    Displays the prompt in the specified color.
    If the specified color key is invalid, the prompt will be displayed in Light Green.

    """

    display = dc.DisplayColor()

    color = get_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color) # type: ignore
        # Displays the prompt in color

        rep_auto = "(o/N)" if not default else "(O/n)"
        colored_prompt = f"{input_color}[?] {message} {rep_auto} ? : {Style.RESET_ALL}"

        reponse = input(colored_prompt).strip().lower()
        if reponse == "":
            return default
        if reponse in {'oui', 'o'}:
            return True
        if reponse in {'non', 'n'}:
            return False
        
        text = f"Réponse valide : (o/N) {ct.BELL}"
        display.print(text, colors['error'])

def waiting_any_key(message: str, color_key: str = 'wait') :
    """
    This function returnsis wainting any key
    """

    color = get_color(color_key)
    # Convert the input color from DISPLAY_COLORS to ANSI
    input_color = rgb_to_ansi(color) # type: ignore

    # Displays the prompt in color
    colored_prompt = f"{input_color}[wait...] {message}{Style.RESET_ALL}"

    _ = input(colored_prompt).strip().lower()

def input_value(message: str, color_key: str = 'input') -> int:
    """
    This function returns a inter > 0 
    """

    display = dc.DisplayColor()
    color = get_color(color_key)

    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color) # type: ignore
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} : {Style.RESET_ALL}"

        try:
            value = int(input(colored_prompt).strip())
            if value > 0:
                return value
            else: 
                text = (
                        f"la valeur doit être positive!! . "
                        f"Réessayez! {ct.BELL}"
                    )
            display.print(text, colors['error'])

        except ValueError:
            text = (
                        f"Ce n'est pas un nombre . "
                        f"Réessayez! {ct.BELL}"
                    )
            display.print(text, colors['error'])

def seuil():
    """
        Saisir deux seuils.
        Les seuils doivent être compris entre 0 & 100.
        on renvoit les deux seuils dans l'ordre croisant'
        """
    
    display = dc.DisplayColor()

    while True:
        display.print("Entrez seuil 'Rare' & 'Dominant'", colors["input"])
        entree = input(" - : ")
        seuil = entree.split()
        seuil.sort() # tri croisant des entrées

        if not valider_seuil(seuil):
            continue

        seuil_rare = float(seuil[0])
        seuil_dominant = seuil_rare if len(seuil) == 1 else float(seuil[1])

        return seuil_rare, seuil_dominant

def valider_seuil(seuil):
    """
    Controle la validitée des entrées
    """
    if len(seuil) != 2:
        print("Veuillez entrer 2 valeurs.")
        return False

    try:
        if  0.0 > float(seuil[0]) or float(seuil[0]) > 100.0:
            print("Veuillez entrer un seuil compris entre 0.0 et 100.0")
            return False
    except ValueError:
        print(f"{seuil[0]} n'est pas valide.")
        return False

    if len(seuil) == 2:
        try:
            float(seuil[1])
        except ValueError:
            print(f"{seuil[1]} n'est pas valide.")
            return False

    return True

def format_nombre(n):
    return f"{n:,}".replace(",", " ")

def afficher_mode(
    label: str,
    enabled: bool,
    ) -> None:
    
    display = dc.DisplayColor()

    display.print(
        f"{label} {'ON' if enabled else 'OFF'}",
        colors["ok" if enabled else "warning"]
    )

    print()

def draw_bar(value, vmin, vmax, length=50):
    """
    Barre visuelle normalisée
    """
    ratio = (value - vmin) / (vmax - vmin)
    ratio = max(0, min(1, ratio))

    filled = int(ratio * length)
    empty = length - filled

    scale = "█" * filled + "░" * empty
    
    return scale


def sortie_de_programme():

    display = dc.DisplayColor()
    print()
    display.print(f"Programme terminé. Au revoir !{ct.BELL}", colors['goodbye'])
    sys.exit(0)
