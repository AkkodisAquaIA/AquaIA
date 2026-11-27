
#!/usr/bin/env python3

import argparse
import json
import zipfile
from pathlib import Path


def extract_image_names(annotations_dir: Path):
    """
    Parcourt tous les fichiers ZIP présents dans le dossier Annotations,
    recherche les fichiers instances_default.json et extrait les noms
    des images présents dans la section "images".
    """

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Le dossier n'existe pas : {annotations_dir}"
        )

    if not annotations_dir.is_dir():
        raise NotADirectoryError(
            f"Le chemin indiqué n'est pas un dossier : {annotations_dir}"
        )

    # Recherche récursive de tous les fichiers ZIP
    zip_files = sorted(annotations_dir.rglob("*.zip"))

    if not zip_files:
        print(f"[ATTENTION] Aucun fichier .zip trouvé dans : {annotations_dir}")
        return []

    print("=" * 70)
    print("EXTRACTION DES NOMS D'IMAGES")
    print("=" * 70)
    print(f"Dossier : {annotations_dir}")
    print(f"Nombre de ZIP trouvés : {len(zip_files)}")
    print()

    all_image_names = []

    for zip_path in zip_files:

        print(f"[ZIP] {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:

                # Recherche de tous les instances_default.json,
                # quelle que soit leur position dans le ZIP
                json_files = [
                    name
                    for name in zf.namelist()
                    if Path(name).name == "instances_default.json"
                ]

                if not json_files:
                    print("      [ATTENTION] Aucun instances_default.json trouvé.")
                    continue

                for json_path in json_files:

                    print(f"      JSON : {json_path}")

                    try:
                        with zf.open(json_path) as json_file:
                            data = json.load(json_file)

                    except json.JSONDecodeError as e:
                        print(f"      [ERREUR JSON] {e}")
                        continue

                    # Vérification de la présence de "images"
                    if "images" not in data:
                        print("      [ATTENTION] Clé 'images' absente du JSON.")
                        continue

                    image_names = []

                    for image in data["images"]:

                        file_name = image.get("file_name")

                        if file_name:
                            image_names.append(file_name)
                            all_image_names.append(file_name)

                    print(f"      Images trouvées : {len(image_names)}")

        except zipfile.BadZipFile:
            print("      [ERREUR] Fichier ZIP invalide.")

        except Exception as e:
            print(f"      [ERREUR] {e}")

        print()

    return all_image_names


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extrait les noms d'images contenus dans les "
            "instances_default.json présents dans des fichiers ZIP."
        )
    )

    parser.add_argument(
        "annotations_dir",
        type=Path,
        help="Chemin vers le dossier Annotations"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("image_names.txt"),
        help="Fichier texte de sortie (défaut : image_names.txt)"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------

    image_names = extract_image_names(args.annotations_dir)

    if not image_names:
        print("Aucun nom d'image trouvé.")
        return

    # ---------------------------------------------------------
    # Suppression des doublons
    # ---------------------------------------------------------

    unique_image_names = sorted(set(image_names))

    duplicates = len(image_names) - len(unique_image_names)

    # ---------------------------------------------------------
    # Sauvegarde
    # ---------------------------------------------------------

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for image_name in unique_image_names:
            f.write(image_name + "\n")

    # ---------------------------------------------------------
    # Résumé
    # ---------------------------------------------------------

    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    print(f"Nombre total de références d'images : {len(image_names)}")
    print(f"Nombre d'images uniques             : {len(unique_image_names)}")
    print(f"Nombre de doublons                  : {duplicates}")

    print()
    print(f"Liste enregistrée dans : {args.output.resolve()}")


if __name__ == "__main__":
    main()

