import json
from pathlib import Path


def generate_cvat_labels(classes_path, output_file="cvat_labels.json"):
    classes_dir = Path(classes_path)

    if not classes_dir.exists():
        raise FileNotFoundError(f"Le chemin n'existe pas : {classes_path}")

    if not classes_dir.is_dir():
        raise NotADirectoryError(f"Ce n'est pas un dossier : {classes_path}")

    labels = []

    for folder in sorted(classes_dir.iterdir()):
        if folder.is_dir():
            labels.append({"name": folder.name, "type": "any", "attributes": []})

    data = labels

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"JSON généré : {output_file}")
    print(f"{len(labels)} classes trouvées.")


if __name__ == "__main__":
    classes_path = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/PERLA_cropped_Sarah_20042026_vuLPL"
    output_file = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/cvat/cvat_labels.json"

    generate_cvat_labels(classes_path, output_file)
