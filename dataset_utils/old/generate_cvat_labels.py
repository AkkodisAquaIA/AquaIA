import json
from pathlib import Path
from openpyxl import load_workbook


def generate_cvat_labels(excel_path, output_file="cvat_labels.json"):
    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {excel_path}")

    # Ouverture du fichier Excel
    wb = load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active

    class_names = []

    # Colonne C = colonne 3
    # min_row=2 permet d'ignorer l'en-tête
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=3, values_only=True):
        class_name = row[0]

        if class_name is not None:
            class_name = str(class_name).strip()

            if class_name:
                class_names.append(class_name)

    wb.close()

    # ---------------------------------------------------------
    # Vérification de l'unicité des labels
    # ---------------------------------------------------------
    duplicates = sorted({name for name in class_names if class_names.count(name) > 1})

    if duplicates:
        print("\nERREUR : des labels sont présents plusieurs fois :")

        for name in duplicates:
            print(f"  - {name} ({class_names.count(name)} occurrences)")

        raise ValueError(f"{len(duplicates)} label(s) dupliqué(s) détecté(s). Le JSON n'a pas été généré.")

    print(f"OK : les {len(class_names)} labels sont uniques.")

    # ---------------------------------------------------------
    # Création des labels CVAT
    # ---------------------------------------------------------
    labels = [{"name": class_name, "type": "any", "attributes": []} for class_name in class_names]

    # Génération du JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)

    print(f"JSON généré : {output_file}")
    print(f"{len(labels)} classes trouvées.")


if __name__ == "__main__":
    excel_path = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/cvat/New_labels_cvat.xlsx"

    output_file = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/cvat/cvat_new_labels.json"

    generate_cvat_labels(excel_path, output_file)
