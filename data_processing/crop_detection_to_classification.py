"""Découpe un dataset type YOLO et range les crops dans un dossier par classe."""

import argparse
import csv
import json
import math
import unicodedata
import warnings
from pathlib import Path

import yaml
from PIL import Image


SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_text(text):
    """Normalise seulement l'écriture Unicode pour comparer deux noms."""
    return unicodedata.normalize("NFC", str(text).strip())


def encode_class_name(class_name):
    """Raccourcit les trois premiers rangs taxonomiques à cinq caractères."""
    parts = normalize_text(class_name).split("_")
    for index in range(min(3, len(parts))):
        parts[index] = parts[index][:5]
    return "_".join(parts)


def load_class_ids(yaml_path=None, json_path=None):
    """Lit les classes du YAML, ou du JSON de secours si le YAML manque."""
    class_ids = {}
    class_sources = {}

    if yaml_path is not None:
        with yaml_path.open("r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        if not isinstance(yaml_data, dict) or "names" not in yaml_data:
            raise ValueError("Le fichier YAML doit contenir une section 'names'.")

        yaml_names = yaml_data["names"]
        if isinstance(yaml_names, list):
            yaml_names = dict(enumerate(yaml_names))
        if not isinstance(yaml_names, dict):
            raise ValueError("La section 'names' du YAML doit être une liste ou un dictionnaire.")

        for raw_id, raw_name in yaml_names.items():
            class_id = int(raw_id)
            class_ids[class_id] = normalize_text(raw_name)
            class_sources[class_id] = "data.yaml"

        return class_ids, class_sources

    if json_path is None:
        raise FileNotFoundError("Aucun data.yaml trouvé. Utilisez --classes-json comme solution de secours.")

    with json_path.open("r", encoding="utf-8") as file:
        json_data = json.load(file)

    if not isinstance(json_data, list):
        raise ValueError("Le fichier JSON doit contenir une liste de classes.")

    for class_id, item in enumerate(json_data):
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"Classe JSON invalide à l'index {class_id}.")
        class_ids[class_id] = normalize_text(item["name"])
        class_sources[class_id] = "json"

    return class_ids, class_sources


def load_taxonomy_prefixes(json_path):
    """Charge et vérifie le registre des trois préfixes taxonomiques."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    prefixes = data.get("prefixes")
    if data.get("prefix_length") != 5 or not isinstance(prefixes, list) or len(prefixes) != 3:
        raise ValueError("Le registre doit contenir trois tables de préfixes de cinq lettres.")

    for rank_prefixes in prefixes:
        if not isinstance(rank_prefixes, dict) or not rank_prefixes:
            raise ValueError("Chaque rang du registre doit être une table non vide.")
        for prefix, full_name in rank_prefixes.items():
            if prefix != full_name[:5]:
                raise ValueError(f"Préfixe incohérent dans le registre : {prefix} -> {full_name}")

    return prefixes


def decode_class_name(encoded_name, prefixes):
    """Développe les trois premiers taxons et conserve famille, genre et espèce."""
    parts = normalize_text(encoded_name).split("_")
    if len(parts) < 3:
        return None, "nom_incomplet"

    decoded_parts = []
    expanded = False

    for index in range(3):
        token = parts[index]
        rank_prefixes = prefixes[index]

        if token in rank_prefixes:
            decoded_parts.append(rank_prefixes[token])
            expanded = True
        elif token in rank_prefixes.values():
            decoded_parts.append(token)
        else:
            return None, "prefixe_inconnu"

    standard_name = "_".join([*decoded_parts, *parts[3:]])
    status = "converti_par_prefixes" if expanded else "deja_complet"
    return standard_name, status


def build_conversion_table(class_ids, class_sources, prefixes):
    """Convertit les noms réellement utilisés par les annotations."""
    rows = []
    id_to_standard = {}

    for class_id in sorted(class_ids):
        encoded_name = class_ids[class_id]
        standard_name, status = decode_class_name(encoded_name, prefixes)

        if standard_name is not None:
            id_to_standard[class_id] = standard_name

        rows.append(
            {
                "class_id": class_id,
                "nom_encode": encoded_name,
                "nom_standard": standard_name or "",
                "source_nom": class_sources[class_id],
                "statut": status,
            }
        )

    return rows, id_to_standard


def save_conversion_table(rows, output_path):
    """Enregistre la table qui permet de passer d'un nom à l'autre."""
    fieldnames = ["class_id", "nom_encode", "nom_standard", "source_nom", "statut"]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_images(image_dir):
    """Retourne les images indexées par leur nom sans extension."""
    images = {}

    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.stem in images:
            raise ValueError(f"Deux images ont le même nom sans extension : {path.stem}")
        images[path.stem] = path

    return images


def read_yolo_labels(label_path):
    """Lit les lignes : classe, centre x, centre y, largeur, hauteur."""
    boxes = []

    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}, ligne {line_number} : cinq valeurs sont attendues.")

        class_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        values = [x_center, y_center, box_width, box_height]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{label_path}, ligne {line_number} : coordonnées hors de [0, 1].")

        boxes.append((class_id, x_center, y_center, box_width, box_height))

    return boxes


def yolo_box_to_pixels(box, image_width, image_height):
    """Convertit une bounding box YOLO normalisée en coordonnées pixels."""
    _, x_center, y_center, box_width, box_height = box

    left = math.floor((x_center - box_width / 2) * image_width)
    top = math.floor((y_center - box_height / 2) * image_height)
    right = math.ceil((x_center + box_width / 2) * image_width)
    bottom = math.ceil((y_center + box_height / 2) * image_height)

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    if right <= left or bottom <= top:
        raise ValueError("La bounding box a une largeur ou une hauteur nulle.")

    return left, top, right, bottom


def crop_split(split, images_dir, labels_dir, output_dir, class_ids, id_to_standard):
    """Traite toutes les annotations d'un split."""
    images = list_images(images_dir)
    label_paths = sorted(labels_dir.glob("*.txt"))
    manifest = []
    ignored = []

    for label_path in label_paths:
        image_path = images.get(label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"Image introuvable pour {label_path}")

        boxes = read_yolo_labels(label_path)
        counters = {}

        with Image.open(image_path) as image:
            for box in boxes:
                class_id = box[0]
                if class_id not in class_ids:
                    raise ValueError(f"Classe {class_id} absente du fichier de classes : {label_path}")
                if class_id not in id_to_standard:
                    ignored.append(
                        {
                            "split": split,
                            "image_source": str(image_path),
                            "label_source": str(label_path),
                            "class_id": class_id,
                            "nom_encode": class_ids[class_id],
                            "raison": "classe_absente_du_referentiel",
                        }
                    )
                    continue

                standard_name = id_to_standard[class_id]
                counters[standard_name] = counters.get(standard_name, 0) + 1
                number = counters[standard_name]

                class_dir = output_dir / standard_name
                class_dir.mkdir(parents=True, exist_ok=True)

                image_name = image_path.stem.replace(" ", "_")
                crop_name = f"{image_name}_LPL_{number}{image_path.suffix.lower()}"
                crop_path = class_dir / crop_name

                if crop_path.exists():
                    raise FileExistsError(f"Le crop existe déjà : {crop_path}")

                pixel_box = yolo_box_to_pixels(box, image.width, image.height)
                crop = image.crop(pixel_box)
                crop.save(crop_path)

                manifest.append(
                    {
                        "split": split,
                        "image_source": str(image_path),
                        "label_source": str(label_path),
                        "class_id": class_id,
                        "nom_encode": class_ids[class_id],
                        "nom_standard": standard_name,
                        "x_center": box[1],
                        "y_center": box[2],
                        "largeur": box[3],
                        "hauteur": box[4],
                        "crop": str(crop_path),
                    }
                )

    return manifest, ignored, len(images), len(label_paths)


def save_manifest(rows, output_path):
    """Enregistre la provenance de chaque crop."""
    fieldnames = [
        "split",
        "image_source",
        "label_source",
        "class_id",
        "nom_encode",
        "nom_standard",
        "x_center",
        "y_center",
        "largeur",
        "hauteur",
        "crop",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ignored_annotations(rows, output_path):
    """Enregistre les annotations qui n'ont pas pu être classées."""
    fieldnames = [
        "split",
        "image_source",
        "label_source",
        "class_id",
        "nom_encode",
        "raison",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def crop_directory_list(
    directories,
    output_dir,
    taxonomy_path,
    yaml_path=None,
    json_path=None,
):
    """Traite une liste de couples (nom, dossier images, dossier labels)."""
    for _, images_dir, labels_dir in directories:
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'images manquant : {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Dossier de labels manquant : {labels_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Le dossier de sortie n'est pas vide : {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids, class_sources = load_class_ids(yaml_path=yaml_path, json_path=json_path)
    prefixes = load_taxonomy_prefixes(taxonomy_path)
    conversion_rows, id_to_standard = build_conversion_table(class_ids, class_sources, prefixes)

    save_conversion_table(conversion_rows, output_dir / "table_conversion_classes.csv")

    manifest = []
    ignored = []
    total_images = 0
    total_labels = 0

    for split, split_images, split_labels in directories:
        split_rows, ignored_rows, image_count, label_count = crop_split(
            split,
            split_images,
            split_labels,
            output_dir,
            class_ids,
            id_to_standard,
        )
        manifest.extend(split_rows)
        ignored.extend(ignored_rows)
        total_images += image_count
        total_labels += label_count

    save_manifest(manifest, output_dir / "manifest_crops.csv")
    ignored_path = output_dir / "annotations_ignorees.csv"
    save_ignored_annotations(ignored, ignored_path)

    if ignored:
        warnings.warn(
            f"{len(ignored)} annotation(s) ignorée(s). Consultez le fichier : {ignored_path}",
            RuntimeWarning,
            stacklevel=2,
        )

    print(f"Images trouvées : {total_images}")
    print(f"Fichiers de labels trouvés : {total_labels}")
    print(f"Images sans labels : {total_images - total_labels}")
    print(f"Crops créés : {len(manifest)}")
    print(f"Annotations ignorées : {len(ignored)}")
    print(f"Classes chargées depuis : {yaml_path or json_path}")
    print(f"Dossier de sortie : {output_dir}")

    return manifest


def crop_directories(
    images_dir,
    labels_dir,
    output_dir,
    taxonomy_path,
    yaml_path=None,
    json_path=None,
):
    """Traite simplement un dossier d'images et son dossier de labels."""
    directories = [("dataset", images_dir, labels_dir)]
    return crop_directory_list(
        directories=directories,
        output_dir=output_dir,
        taxonomy_path=taxonomy_path,
        yaml_path=yaml_path,
        json_path=json_path,
    )


def crop_splits(dataset_dir, output_dir, taxonomy_path, yaml_path=None, json_path=None):
    """Traite les dossiers train, val et test d'un dataset complet."""
    directories = []

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        directories.append((split, images_dir, labels_dir))

    return crop_directory_list(
        directories=directories,
        output_dir=output_dir,
        taxonomy_path=taxonomy_path,
        yaml_path=yaml_path,
        json_path=json_path,
    )


def find_parent_file(start_dir, filename):
    """Cherche un fichier dans un dossier puis dans ses dossiers parents."""
    for directory in [start_dir, *start_dir.parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Découpe les bounding boxes YOLO et range les crops par classe.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Dossier contenant les images sources",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Dossier contenant les annotations TXT",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Nouveau dossier de classification",
    )
    parser.add_argument(
        "--classes-json",
        type=Path,
        default=None,
        help="JSON de secours à utiliser seulement en l'absence de data.yaml",
    )
    parser.add_argument(
        "--classes-yaml",
        type=Path,
        default=None,
        help="data.yaml optionnel, recherché automatiquement près des labels",
    )
    parser.add_argument(
        "--taxonomy-prefixes",
        type=Path,
        default=None,
        help="Registre JSON des préfixes taxonomiques",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()

    yaml_path = args.classes_yaml or find_parent_file(labels_dir, "data.yaml")
    json_path = args.classes_json

    if yaml_path is None and json_path is None:
        raise FileNotFoundError("Aucun data.yaml trouvé. Utilisez --classes-json comme solution de secours.")

    taxonomy_path = args.taxonomy_prefixes
    if taxonomy_path is None:
        taxonomy_path = Path(__file__).parent / "resources" / "taxonomy_prefixes.json"

    crop_directories(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        taxonomy_path=taxonomy_path.resolve(),
        yaml_path=yaml_path.resolve() if yaml_path else None,
        json_path=json_path.resolve() if json_path else None,
    )


if __name__ == "__main__":
    main()
