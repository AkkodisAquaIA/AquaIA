import os
import sys
import threading
import time
import numpy as np
import socket
import traceback
from collections import Counter
from colorama import init, Style
from pathlib import Path
import shutil
import argparse
import fiftyone as fo
from collections import defaultdict

# from typing import Tuple, List, Dict, Optional, Any
from typing import TypedDict

import tools.display_color as dc
from tools.constants import DISPLAY_COLORS as colors
from tools import constants as ct

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
        spinner: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
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
#------------------------------------------------------------------------------------------

#------------------------------
# Function to clear the console screen
#------------------------------
def clear_screen() -> None:
    """
    Clear the console screen depending on the operating system.

    Uses:
        - 'cls' on Windows
        - 'clear' on Unix-based systems (Linux / macOS)
    """
    os.system("cls" if os.name == "nt" else "clear")


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

    outside_ratios = [
        a['outside_ratio_pct']
        for a in results.get('anomalies', [])
        if 'outside_ratio_pct' in a
    ]

    if not outside_ratios:
        print("⚠ No overflow detected, using minimum thresholds.")
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

    print("Automatic threshold calibration:")
    print(f"  - Warning ({warning_percentile} percentile): "
          f"{warning_calculated:.2f}% → Final: {warning_final:.2f}%")
    print(f"  - Error   ({error_percentile} percentile): "
          f"{error_calculated:.2f}% → Final: {error_final:.2f}%")

    return {
        "BBOX_OVERFLOW_WARNING": warning_final,
        "BBOX_OVERFLOW_ERROR": error_final
    }


# ------------------------------
# Function to find a free port
# ------------------------------
def get_free_port() -> int:
    """
    Returns an available port on the local machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Let OS assign a free port
        return s.getsockname()[1]


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
    
    port = get_free_port()
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


# ------------------------------
# Function to parse command-line arguments
# ------------------------------
def read_param() -> tuple[Path, Path]:
    """
    Parses command-line arguments for base path and data folder.

    Returns:
        tuple[Path, Path]: base path and data folder path

    Raises:
        FileNotFoundError: If paths do not exist
        NotADirectoryError: If the data path is not a directory
    """
    parser = argparse.ArgumentParser(description="Program using a base path and a data folder")

    parser.add_argument(
        "-p", "--work_dir",
        required=True,
        help="Base path (e.g., C:/my_data)"
    )

    parser.add_argument(
        "-f", "--folder",
        required=True,
        help="Folder containing the data (inside the base path)"
    )

    args = parser.parse_args()

    base_path = Path(args.work_dir)
    data_path = base_path / args.folder

    # Validate paths
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path}")

    return base_path, data_path


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


# ------------------------------
# Function to display and save problematic items
# ------------------------------
def display_and_save_errors(
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
    if ct.REPORT_MODE:
        file_path: Path = path_user / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(str(x) if full_path else Path(x).name)
                f.write("\n")

        display.print(f" ****** '{file_name}' create *****\n", colors["warning"])


def format_and_display_error(texte: str, rep: str = "") -> None:
    """
    Display error information depending on DEBUG_MODE.

    In debug mode:
        - Full traceback is displayed and saved.

    In normal mode:
        - Only exception type and message are shown.

    Args:
        texte (str): Custom error message prefix.
        rep (str): Directory where fault report is stored (debug mode).
    """

    display: dc.DisplayColor = dc.DisplayColor()

    exc_type, exc_value, exc_traceback = sys.exc_info()

    if ct.DEBUG_MODE:
        tb: str = ''.join(traceback.format_exception(
            exc_type, exc_value, exc_traceback
        ))

        prompt: str = f"{texte}:\n{tb}"

        report: Path = Path(rep, "fault.txt")
        print(f"---- Error report saved to: {report}")

        with open(report, "a", encoding="utf-8") as f:
            f.write(f"{texte}:\n{tb}\n")

    else:
        assert exc_type is not None
        prompt = f"{texte}:\n{exc_type.__name__}: {exc_value}"

    display.print(prompt, colors['error'])


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

    display.print("Detected errors:", colors["error"])

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

    output_path =  path_user / file_name
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

    display.print(f" ****** '{file_name}' create *****\n", colors["warning"])        
