import os
import sys
import socket
import traceback
from collections import Counter
from pathlib import Path
import argparse
import fiftyone as fo

from typing import Tuple, List

import tools.display_color as dc
from tools.constants import DISPLAY_COLORS as colors
from tools import constants as ct

#==========================================================================================

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
def read_param() -> Tuple[Path, Path]:
    """
    Parses command-line arguments for base path and data folder.

    Returns:
        Tuple[Path, Path]: base path and data folder path

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


# ------------------------------
# Function to display and save problematic items
# ------------------------------
def display_and_save_errors(
    items: List[str],
    file_name: str,
    title: str,
    sort: bool = True,
    full_path: bool = False,
) -> None:
    """
    Display and save a list of problematic items in a file.

    Args:
        items (List[str]): List of file paths or names
        file_name (str): Output file name
        title (str): Title to display
        sort (bool): Whether to sort items before displaying
        full_path (bool): Display/write full paths if True, else only file names
        n_per_line (int): Number of items per line when printing
    """

    display = dc.DisplayColor()

    if not items:
        display.print(f"{title}: No issues detected.\n", colors['ok'])
        return

    if sort:
        items = sorted(items, key=lambda p: str(p))

    display.print(f"{title}: {len(items)} item(s) detected", colors['warning'], bold=True)
    for i in range(0, len(items), ct.n_per_line):
        line_items = items[i:i + ct.n_per_line]
        line = " | ".join(str(x) if full_path else Path(x).name for x in line_items)
        print(line)
    
    # Save to file
    if ct.REPORT_MODE:
        with open(file_name, "w") as f:
            for x in items:
                f.write(str(x) if full_path else Path(x).name)
                f.write("\n")

        print(f"List saved to '{file_name}'\n")
    else:
        print()



def format_and_display_error(texte : str, rep= "") -> None  :
    """
    Handles errors based on the specified level of detail.
    """

    display = dc.DisplayColor()

    # Retrieve the type, value, and traceback of the most recent exception
    exc_type, exc_value, exc_traceback = sys.exc_info()
 
    # --- Full mode (DEBUG) ---
    if ct.DEBUG_MODE:
        tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        prompt = f"{texte} :\n{tb}"

        report = Path(rep, "fault.txt"  )
        print(f' ---- Error report saved to: {report}     ')
        with open(report, "a", encoding="utf-8") as f:
            f.write(f"{texte} :\n {''.join(tb)}\n")

    # --- Simplified mode (without traceback) ---                                             
    else:
        # Only display the exception type and value
        prompt = f"{texte} :\n{exc_type.__name__}: {exc_value}"

    display.print(prompt, colors['error'])

    
def afficher_bbox_erreurs_compact(bbox_erreurs, noms_par_ligne=5):
    """
    Affiche les erreurs de bbox par catégorie, plusieurs noms d'images par ligne.
    
    bbox_erreurs : dict retourné par detect_bbox_problemes_detail()
    noms_par_ligne : nombre de noms d'images affichés par ligne
    """

    display = dc.DisplayColor()

    # Calculer largeur max pour aligner les catégories
    categorie_max_len = max(len(cat) for cat in bbox_erreurs.keys())
    
    display.print("Erreurs de bbox détectées :", colors["warning"])
    
    for categorie, chemins in bbox_erreurs.items():
        if not chemins:
            continue  # ignorer les catégories sans erreur
        
        # Titre catégorie + nombre d'images
        print(f"{categorie.capitalize().ljust(categorie_max_len)} ({len(chemins)} images) :")
        
        # Extraire uniquement les noms de fichiers
        noms_images = [os.path.basename(chemin) for chemin in chemins]
        
        # Affichage en blocs de plusieurs noms par ligne
        for i in range(0, len(noms_images), noms_par_ligne):
            ligne = "  ".join(noms_images[i:i+noms_par_ligne])
            print(f"{' ' * (categorie_max_len + 3)}{ligne}")
        
        print()  # ligne vide entre catégories



def afficher_distribution_classes(class_distribution, classes_par_ligne=4):

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



def afficher_dataset_statistics(resultats):

    display = dc.DisplayColor()

    stats = resultats["stats"]
    class_distribution = resultats["class_distribution"]
    anomalies = resultats["anomalies"]

    display.print("Statistiques du dataset YOLO", colors["info"])

    print("\n--- Dataset ---")

    print(f"{'Images':20}: {stats['images']}")
    print(f"{'Fichiers labels':20}: {stats['labels']}")
    print(f"{'Bounding boxes':20}: {stats['bounding_boxes']}")

    print("\n--- Bounding boxes ---")

    print(f"{'Largeur moyenne':20}: {stats['bbox_width_mean']:.4f}")
    print(f"{'Hauteur moyenne':20}: {stats['bbox_height_mean']:.4f}")
    print(f"{'Aire moyenne':20}: {stats['bbox_area_mean']:.4f}")

    print(f"{'Largeur min':20}: {stats['bbox_width_min']:.4f}")
    print(f"{'Largeur max':20}: {stats['bbox_width_max']:.4f}")

    print(f"{'Hauteur min':20}: {stats['bbox_height_min']:.4f}")
    print(f"{'Hauteur max':20}: {stats['bbox_height_max']:.4f}")

    afficher_distribution_classes(class_distribution, classes_par_ligne=4)


    print("\n--- Annotations suspectes ---")

    total = sum(class_distribution.values())
    if not anomalies:
        display.print("Aucune anomalie détectée", colors["ok"])
    else:

        anomalies_count = Counter(a[0] for a in anomalies)

        for type_anom, count in anomalies_count.items():
            pct = (count / total) * 100
            display.print(
                f"{type_anom:<20}: {count} ({pct:.3f}%)",
                colors["warning"]
            )



