
import os
from pathlib import Path
from collections import defaultdict
import fiftyone as fo
import fiftyone.core.labels as fol

from tools import system as syst
from tools import utility as util
from config import constants as ct
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors

display = dc.DisplayColor()

_DATASET_CACHE = {}

# =======================================================================================
def group_anomalies(anomalies):
    grouped = defaultdict(list)
    for a in anomalies:
        grouped[a["image"]].append(a)
    return grouped

def create_dataset(DATASET_DIR,  yaml_path=None, anomalies=None):

    is_def = bool(anomalies)
    dataset_name = f"{DATASET_DIR.stem}_{'def' if is_def else 'ok'}"

    display.print("Création du dataset FiftyOne :", colors['info'])
    print(f"    '{dataset_name}'")

    print()
    if not anomalies :   
        display.print(" - Dataset Ok", colors['ok'])
        display.print(f" - Création du dataset avec fichier '.yaml': {Path(yaml_path).stem}\n", colors['ok']) # stem or name depending on your needs
    else:
        display.print(" - Dataset non valide", colors['warning'])
        display.print(" - Création d'un dataset d'anomalies\n",colors['warning'])

    # Mini barre
    progress = util.MiniProgressBar("Chargement dataset", width=20)
    progress.start()

    try:
        # =========================================================
        # CAS 1 : dataset classique YOLO sans anomalies
        # =========================================================
        if not anomalies :
         
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.YOLOv5Dataset, # type: ignore
                dataset_dir=str(DATASET_DIR),
                yaml_path=str(yaml_path),
                name=dataset_name,
                overwrite=True
            )

        # =========================================================
        # CAS 2 : dataset avec anomalies
        # =========================================================
        else:

            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)

            dataset = fo.Dataset(dataset_name)

            grouped = group_anomalies(anomalies)
            samples = []

            for img_name, image_anomalies in grouped.items():

                image_path = (
                    Path(DATASET_DIR) / "images/train2017" / img_name
                )

                label_path = (
                    Path(DATASET_DIR)
                    / "labels/train2017"
                    / img_name.replace(".jpg", ".txt")
                )

                if not image_path.exists() or not label_path.exists():
                    print(f"Fichier manquant pour {img_name}")
                    continue

                detections = []

                # Lire fichier YOLO
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:

                    line = line.strip()

                    # ignorer ligne vide
                    if not line:
                        continue

                    parts = line.split()

                    x_center, y_center, width, height = map(
                        float,
                        parts[1:5]
                    )

                    # Associer bbox aux anomalies
                    for anomaly in image_anomalies:

                        if (
                            abs(anomaly["width"] - width) < 1e-6
                            and abs(anomaly["height"] - height) < 1e-6
                        ):

                            # YOLO → FiftyOne
                            x = x_center - width / 2
                            y = y_center - height / 2

                            detection = fol.Detection(
                                label=anomaly["type"],
                                bounding_box=[x, y, width, height],
                                confidence=1.0,
                            )

                            detections.append(detection)

                sample = fo.Sample(filepath=str(image_path))
                sample["anomalies"] = fol.Detections(
                    detections=detections
                )

                samples.append(sample)

            dataset.add_samples(samples)

    finally:
        progress.stop()

    display.print(f"Dataset créé avec {util.format_nombre(len(dataset))} images\n", colors['ok'])  # type: ignore

    return dataset

def is_valid_dataset(name):
    if not fo.dataset_exists(name):
        return False
    ds = fo.load_dataset(name)
    return fo.dataset_exists(name) and len(ds) > 0

def get_dataset(DATASET_DIR, dataset_yaml, anomalies):

    is_def = bool(anomalies)
    dataset_name = f"{DATASET_DIR.stem}_{'def' if is_def else 'ok'}"


    # 1. Cache Python (IMPORTANT)
    if dataset_name in _DATASET_CACHE:

        cached = _DATASET_CACHE[dataset_name]

        if len(cached) > 0:
            display.print("Dataset 'Fifty_One' déjà chargé depuis le cache :", colors['info'])
            return cached

    else:
        # 2. Dataset FiftyOne déjà existant
        if is_valid_dataset(dataset_name):

            dataset = fo.load_dataset(dataset_name)

        else:
 
            if fo.dataset_exists(dataset_name):
               fo.delete_dataset(dataset_name)

            dataset = create_dataset(
                DATASET_DIR,
                yaml_path=dataset_yaml,
                anomalies=anomalies
            )

    # 4. Cache mémoire
    _DATASET_CACHE[dataset_name] = dataset # type: ignore

    return dataset # type: ignore


def launch_fiftyone_interface(dataset: fo.Dataset) -> None:
    """
    Launches the FiftyOne web app for a given dataset.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset to visualize.
    """

    display = dc.DisplayColor()    

    display.print("Lancement de l'interface web FiftyOne...", colors['info'])
    
    port = syst.get_free_port()
    session = None
    try:
        session = fo.launch_app(dataset, port=port, remote=False)
        display.print(f"tyOne web interface accessible à l'adresse: http://127.0.0.1:{port}", colors['info'])
        display.print("Attente de la fermeture de l'interface web", colors['wait'], bold=True)
        display.print("Appuyez sur CTRL+C pour continuer si nécessaire.", colors['wait'], bold=True)

        # Wait until the session is closed
        try:
            session.wait()
        except KeyboardInterrupt:
            display.print("CTRL+C détecté, continuation du programme...", colors['warning'])

    except Exception as e:
        display.print("Échec du lancement de l'interface web FiftyOne.", colors['error'])
        print("Error:", e)

    finally:
        if session is not None:
            session.close()
            display.print("FiftyOne session fermée, continuation du programme.", colors['info'])

# =======================================================================================

def launch_fifty_one(mode_aff, data_fifty_one):

    mode_affichage = mode_aff[0]
    etat_dataset = mode_aff[1]
    nom_dataset = mode_aff[2]

    display.print("(7) Lancement de FiftyOne", colors[etat_dataset], nom_dataset) # type: ignore

    # if mode_affichage == ct.ECRAN :

    anomalies    = data_fifty_one[0]
    DATASET_DIR  = data_fifty_one[1]
    dataset_yaml = data_fifty_one[2]

    dataset = get_dataset(DATASET_DIR, dataset_yaml, anomalies)

    # Lancement de Fifty_One
    launch_fiftyone_interface(dataset) # type: ignore   


