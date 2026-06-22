import json
from pathlib import Path
from openpyxl import Workbook


def abbreviate_class_name(class_name: str) -> str:
    parts = class_name.split("_")

    # Cas 1
    if len(parts) >= 6 and parts[-5:] == ["Classis", "Order", "Familia", "Genus", "sp"]:
        return "_".join(parts[:-5])

    # Cas 2
    if len(parts) >= 5 and parts[3] == "Familia" and parts[-2:] == ["Genus", "sp"]:
        prefix = parts[0][:2] + parts[1][:2]
        third = parts[2]
        return f"{prefix}_{third}_FaGesp"

    # Cas 3
    if len(parts) >= 4 and parts[-2:] == ["Genus", "sp"]:
        prefix_parts = parts[:-3]
        family_part = parts[-3]
        prefix = "".join(part[:2] for part in prefix_parts)
        return f"{prefix}_{family_part}_Gesp" if prefix else f"{family_part}_Gesp"

    # Cas général
    if len(parts) <= 2:
        return class_name

    prefix_parts = parts[:-2]
    last_parts = parts[-2:]
    prefix = "".join(part[:2] for part in prefix_parts)

    return f"{prefix}_{'_'.join(last_parts)}"


def generate_excel_and_check(classes_dir, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "classes"

    ws.append(["nom_classe", "nom_abrege"])

    abbreviations = {}
    duplicates = {}

    for folder in sorted(classes_dir.iterdir()):
        if folder.is_dir():
            class_name = folder.name
            abbreviated_name = abbreviate_class_name(class_name)

            ws.append([class_name, abbreviated_name])

            if abbreviated_name in abbreviations:
                duplicates.setdefault(abbreviated_name, set()).update([class_name, abbreviations[abbreviated_name]])
            else:
                abbreviations[abbreviated_name] = class_name

    wb.save(output_file)

    print(f"Excel généré : {output_file}")

    if duplicates:
        print("\n⚠️ Doublons détectés :")
        for abbr, classes in duplicates.items():
            print(f"{abbr} -> {list(classes)}")
    else:
        print("\n✅ Toutes les abréviations sont uniques !")


def generate_cvat_labels(classes_dir, output_file="cvat_labels.json"):
    labels = []

    for folder in sorted(classes_dir.iterdir()):
        if folder.is_dir():
            short_name = abbreviate_class_name(folder.name)

            labels.append(
                {
                    "name": short_name,
                    "type": "rectangle",  # adapté pour détection
                    "attributes": [],
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)

    print(f"JSON CVAT généré : {output_file}")
    print(f"{len(labels)} classes trouvées.")


def main():

    classes_path = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/PERLA_cropped_Sarah_20042026_vuLPL"
    excel_output = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/cvat/classes_abbreviations.xlsx"
    json_output = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/cvat/cvat_labels.json"

    classes_dir = Path(classes_path)

    if not classes_dir.exists():
        raise FileNotFoundError(f"Le chemin n'existe pas : {classes_path}")

    if not classes_dir.is_dir():
        raise NotADirectoryError(f"Ce n'est pas un dossier : {classes_path}")

    generate_excel_and_check(classes_dir, excel_output)
    generate_cvat_labels(classes_dir, json_output)


if __name__ == "__main__":
    main()
