
from pathlib import Path
from collections import defaultdict
import fiftyone as fo
import fiftyone.core.labels as fol

from tools import system as syst
from tools import utility as util
from config import constants as ct
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors

display = dc.DisplayColor()

# =======================================================================================
def group_anomalies(anomalies):
    grouped = defaultdict(list)
    for a in anomalies:
        grouped[a["image"]].append(a)
    return grouped

def create_dataset(DATASET_DIR,  yaml_path=None, anomalies=None):

    dataset_name = (
                f"{Path(DATASET_DIR).name}"
                f"{'_def' if anomalies is not None else '_ok'}"
                )

    display.print("Création du dataset FiftyOne :", colors['info'])
    print(f"    '{dataset_name}'")

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    fo.close_app()

    # Mini barre
    progress = util.MiniProgressBar("Chargement dataset", width=20)
    progress.start()


    try:
        # =========================================================
        # CAS 1 : dataset classique YOLO
        # =========================================================
        if anomalies is None:
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.YOLOv5Dataset, # type: ignore
                dataset_dir=str(DATASET_DIR),
                yaml_path=str(yaml_path),
                name=dataset_name
            )

        # =========================================================
        # CAS 2 : dataset avec anomalies
        # =========================================================
        else:

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

def lautch_fifty_one(data_fifty_one):

    anomalies    = data_fifty_one[0]
    DATASET_DIR  = data_fifty_one[1]
    dataset_yaml = data_fifty_one[2]

    try:
        if not anomalies:
            display.print(" - Dataset Ok", colors['ok'])
            display.print(f" - Création du dataset avec fichier : {Path(dataset_yaml).name}\n", colors['ok'])

            dataset = create_dataset(DATASET_DIR, yaml_path=dataset_yaml)

        else:
            display.print(" - Dataset non valide", colors['warning'])
            display.print(" - Création d'un dataset d'anomalies\n",colors['warning'])

            dataset = create_dataset(DATASET_DIR, anomalies=anomalies)
        
    except ValueError:
        display.print(f"!!! Dataset non valide !!!\n{ct.BELL}", colors['error'])
        util.sortie_de_programme()

    # Lancement de Fifty_One
    launch_fiftyone_interface(dataset) # type: ignore   
