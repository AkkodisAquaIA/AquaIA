import os
import sys
import threading
import time
import numpy as np
import platform
from collections import Counter
from colorama import Fore, Style
from pathlib import Path
import shutil
import argparse
import fiftyone as fo
from collections import defaultdict
from datetime import datetime

from typing import TypedDict

from tools import system as syst
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors
from config import constants as ct
from config import process as pr

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
        """Start the progress bar animation in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
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
        """Stop the progress bar animation."""
        assert self.thread is not None
        self.running = False
        self.thread.join()

        # Clear the line
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

def horodatage(file_name: str) -> str:

    # Generate a timestamp (format: YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 

    # Extract filename without extension and file extension
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix

    # Build a new filename including the timestamp
    filename_with_timestamp = f"{timestamp}_{stem}_{suffix}"

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
        display.print(" No overflow detected, using minimum thresholds.", colors['ok'])
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

    display.print("Automatic threshold calibration:", colors["warning"])
    print(f"  - Warning ({warning_percentile} percentile): "
          f"{warning_calculated:.2f}% → Final: {warning_final:.2f}%")
    print(f"  - Error   ({error_percentile} percentile): "
          f"{error_calculated:.2f}% → Final: {error_final:.2f}%")

    return {
        "BBOX_OVERFLOW_WARNING": warning_final,
        "BBOX_OVERFLOW_ERROR": error_final
    }


# ------------------------------
# Function to launch FiftyOne web interface
# ------------------------------
def launch_fiftyone_interface(dataset: fo.Dataset) -> None:
    """
    Launches the FiftyOne web app for a given dataset.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset to visualize.
    """

    color = dc.DisplayColor()

    color.print("Launching FiftyOne web interface...", colors['info'])
    
    port = syst.get_free_port()
    session = None
    try:
        session = fo.launch_app(dataset, port=port, remote=False)
        color.print(f"tyOne web interface available at: http://127.0.0.1:{port}", colors['info'])
        color.print("Waiting for the web interface to close", colors['wait'], bold=True)
        color.print("Press CTRL+C to continue if needed.", colors['wait'], bold=True)

        # Wait until the session is closed
        try:
            session.wait()
        except KeyboardInterrupt:
            color.print("CTRL+C detected, continuing program...", colors['warning'])

    except Exception as e:
        color   .print("Failed to launch FiftyOne web interface.", colors['error'])
        print("Error:", e)

    finally:
        if session is not None:
            session.close()
            color.print("FiftyOne session closed, continuing program.", colors['info'])

def rgb_to_ansi(rgb: tuple[int, int, int]) -> str:
    """Convert RGB color to ANSI escape code."""
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"

def chck_color(color_key: str) -> tuple[int, int, int]:
    """
    Check if the color key exists in the DISPLAY_COLORS dictionary.
    If it does, return the corresponding RGB value.
    If not, return a default color (Light Green).
    """
    try:
        color = colors[color_key]
    except KeyError:
        color = (153, 204, 51)  # Default to Light Green

    # Check if the color is a valid RGB tuple.
    if not (isinstance(color, tuple) and len(color) == 3 and
             all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
        color = (153, 204, 51)

    return color

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
    color: tuple[int, int, int] = chck_color(color_key)

    while True:
        input_color: str = rgb_to_ansi(color)
        colored_prompt: str = f"{input_color}[?] {prompt}: {Style.RESET_ALL}"

        path_input: str = input(colored_prompt).strip()

        if os.path.exists(path_input):
            return Path(path_input)

        error_text: str = f"Invalid path: {path_input}. Please try again."
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

    color = chck_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} (o/N, défaut = N) ? : {Style.RESET_ALL}"

        reponse = input(colored_prompt).strip().lower()
        if reponse == "":
            return default
        if reponse in {'oui', 'o'}:
            return True
        if reponse in {'non', 'n'}:
            return False
        
        text = f"Réponse valide : (o/N) {ct.BELL}"
        display.print(text, colors['error'])

def titre_centre(texte, largeur=120, remplissage='—'):  # '—' 
    return f" {texte} ".center(largeur, remplissage)


# ------------------------------
# Function to display and save problematic items
# ------------------------------
def display_and_save_errors(
    cfg,
    path_user: Path,
    items: list[str],
    file_name: str,
    title: str,
    sort: bool = True,
    full_path: bool = False,
) -> None:
    """
    Display and optionally save problematic items to a file.

    Args:
        path_user (Path): Base directory where the report will be saved.
        items (List[str]): List of file paths or file names.
        file_name (str): Output report file name.
        title (str): Section title for display.
        sort (bool): Whether to sort items alphabetically.
        full_path (bool): If True, write full paths; otherwise file names only.
    """

    display: dc.DisplayColor = dc.DisplayColor()

    if not items:
        display.print(f"{title}: No issues detected.\n", colors['ok'])
        return

    if sort:
        items = sorted(items, key=lambda p: str(p))

    # Save to file if report mode is enabled
    if cfg["REPORT_MODE"]:

        new = horodatage(file_name)
        file_path: Path = path_user / new

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for x in items:
                    f.write(str(x) if full_path else Path(x).name)
                    f.write("\n")
        except FileNotFoundError:
             display.print(f"Impossible de sauvegarder : {file_path}", colors['error'])

        display.print(f" *** Fichier erreur : '{file_name}' create ", colors["warning"])


def afficher_bbox_erreurs_compact(
    bbox_erreurs: dict[str, list[str]],
    largeur_max_ligne: int | None = None
) -> None:
    """
    Display bounding box errors grouped by category.

    The display automatically adapts to terminal width
    without cutting file names.

    Args:
        bbox_erreurs (Dict[str, List[str]]):
            Dictionary mapping error categories to image path lists.
        largeur_max_ligne (Optional[int]):
            Maximum line width. If None, half terminal width is used.
    """

    display: dc.DisplayColor = dc.DisplayColor()

    if not bbox_erreurs:
        display.print("No bounding box errors detected.", colors["ok"])
        return

    if largeur_max_ligne is None:
        largeur_max_ligne = shutil.get_terminal_size().columns // 2

    categorie_max_len: int = max(len(cat) for cat in bbox_erreurs.keys())
    indent: str = " " * (categorie_max_len + 3)
    separateur: str = " | "

    display.print(" --- Detected errors:", colors["error"])

    for categorie, chemins in bbox_erreurs.items():
        if not chemins:
            continue

        display.print(
            f"{categorie.capitalize().ljust(categorie_max_len)} "
            f"({len(chemins)} images):",
            colors["error"]
        )

        noms_images: list[str] = [Path(chemin).name for chemin in chemins]

        ligne: str = ""

        for nom in noms_images:
            element: str = nom if not ligne else separateur + nom

            # Check if adding the element exceeds allowed width
            if len(indent) + len(ligne) + len(element) > largeur_max_ligne:
                print(f"{indent}{ligne}")
                ligne = nom
            else:
                ligne += element

        if ligne:
            print(f"{indent}{ligne}")

        print()


def afficher_distribution_classes(
    class_distribution: dict[int, int],
    classes_par_ligne: int = 4
) -> None:

    print("\n--- Distribution des classes ---")

    total = sum(class_distribution.values())

    items = sorted(class_distribution.items())

    largeur_colonne = 28  # largeur d'une colonne pour aligner

    for i in range(0, len(items), classes_par_ligne):

        ligne = items[i:i+classes_par_ligne]

        morceaux = []

        for cls, count in ligne:
            pct = (count / total) * 100
            txt = f"classe {cls:<3}: {count:<6} ({pct:5.2f}%)"
            morceaux.append(txt.ljust(largeur_colonne))

        print("".join(morceaux))


def afficher_dataset_statistics(
    resultats: ValidationResults,
    path_user: Path
) -> None:
    """
    Display YOLO dataset statistics in a structured format.

    Args:
        resultats (Dict[str, Any]):
            Dictionary containing:
                - "stats": dataset statistics
                - "class_distribution": class frequency mapping
                - "anomalies": anomaly list
        path_user (Path):
            Output directory (may be used later for reporting).
    """

    display: dc.DisplayColor = dc.DisplayColor()

    stats: DatasetStats = resultats["stats"]
    class_distribution: dict[int, int] = resultats["class_distribution"]
    anomalies: list[Anomaly] = resultats["anomalies"]

    display.print("YOLO Dataset Statistics", colors["info"])

    print("\n--- Dataset ---")

    print(f"{'Images':20}: {stats['images']}")
    print(f"{'Label files':20}: {stats['labels']}")
    print(f"{'Bounding boxes':20}: {stats['bounding_boxes']}")

    print("\n--- Bounding Boxes ---")

    print(f"{'Width mean':20}: {stats['bbox_width_mean']:.4f}")
    print(f"{'Height mean':20}: {stats['bbox_height_mean']:.4f}")
    print(f"{'Area mean':20}: {stats['bbox_area_mean']:.4f}")

    print(f"{'Width min':20}: {stats['bbox_width_min']:.4f}")
    print(f"{'Width max':20}: {stats['bbox_width_max']:.4f}")

    print(f"{'Height min':20}: {stats['bbox_height_min']:.4f}")
    print(f"{'Height max':20}: {stats['bbox_height_max']:.4f}")

    afficher_distribution_classes(class_distribution, classes_par_ligne=4)

    print("\n--- Suspicious Annotations ---")

    total: int = sum(class_distribution.values())

    if not anomalies:
        display.print("No anomalies detected", colors["ok"])
        return

    anomalies_count: Counter[str] = Counter(a[0] for a in anomalies) # type: ignore

    for type_anom, count in anomalies_count.items():
        pct: float = (count / total) * 100 if total > 0 else 0.0

        display.print(
            f"{type_anom:<20}: {count} ({pct:.3f}%)",
            colors["warning"]
        )


def save_anomalies_readable(
    anomalies: list[Anomaly],
    file_name: str,
    path_user: Path
) -> None:
    """
    Sauvegarde les anomalies dans un fichier texte lisible :
    - Résumé des anomalies par type
    - Images regroupées par type
    - Plusieurs images par ligne (configurable via ct.N_PER_LINE)
    """
    
    display = dc.DisplayColor()
    
    # Regroupement par type
    anomalies_by_type = defaultdict(set)
    for a in anomalies:
        typ = a.get("type")
        img_name = os.path.basename(a.get("image", "unknown"))
        if typ and img_name:
            anomalies_by_type[typ].add(img_name)

    # Tri alphabétique des images par type
    for typ in anomalies_by_type:
        anomalies_by_type[typ] = sorted(anomalies_by_type[typ]) # type: ignore

    # Generate a timestamp (format: YYYYMMDD_HHMMSS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
 
    new = horodatage(file_name)
    output_path =  path_user / new

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # --- Résumé ---
            f.write("=== RÉSUMÉ DES ANOMALIES ===\n")
            for typ, images in anomalies_by_type.items():
                f.write(f"{typ}: {len(images)} image(s)\n")
            f.write("\n")

            # --- Détails par type ---
            for typ, images in anomalies_by_type.items():
                f.write(f"--- {typ} ---\n")
                for i in range(0, len(images), ct.N_PER_LINE):
                    line_images = images[i:i+ ct.N_PER_LINE] # type: ignore
                    f.write(" | ".join(line_images) + "\n")
                f.write("\n")
    except FileNotFoundError:
             display.print(f"Impossible de sauvegarder : {output_path}", colors['error'])

    display.print(f" ****** '{file_name}' create *****", colors["warning"])        

def sortie_de_programme():
    display = dc.DisplayColor()
    display.print(f"Programme terminé. Au revoir !{ct.BELL}", colors['goodbye'])
    sys.exit(0)
