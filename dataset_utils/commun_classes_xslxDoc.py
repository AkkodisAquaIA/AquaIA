from pathlib import Path
from openpyxl import load_workbook, Workbook


def read_first_column(excel_path):
    """
    Lit la première colonne d'un fichier Excel.
    La première ligne est considérée comme un en-tête.
    """
    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {excel_path}")

    wb = load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active

    classes = []

    # Colonne A, à partir de la ligne 2
    for row in ws.iter_rows(
        min_row=2,
        min_col=1,
        max_col=1,
        values_only=True
    ):
        value = row[0]

        if value is not None:
            class_name = str(value).strip()

            if class_name:
                classes.append(class_name)

    wb.close()

    return classes


def compare_classes(file1, file2, output_file):

    # Lecture des deux fichiers
    classes1 = read_first_column(file1)
    classes2 = read_first_column(file2)

    # Conversion en set pour éviter les doublons
    set1 = set(classes1)
    set2 = set(classes2)

    # Classes communes
    common_classes = sorted(set1 & set2)

    # Classes présentes uniquement dans chaque fichier
    only_file1 = sorted(set1 - set2)
    only_file2 = sorted(set2 - set1)

    # ---------------------------------------------------------
    # Affichage du résumé
    # ---------------------------------------------------------

    print("\n========================================")
    print("RÉSUMÉ")
    print("========================================")

    print(f"Classes uniques fichier 1 : {len(set1)}")
    print(f"Classes uniques fichier 2 : {len(set2)}")
    print(f"Classes en commun          : {len(common_classes)}")
    print(f"Uniquement fichier 1       : {len(only_file1)}")
    print(f"Uniquement fichier 2       : {len(only_file2)}")

    print("\nClasses en commun :")

    for class_name in common_classes:
        print(f"  - {class_name}")

    # ---------------------------------------------------------
    # Création du fichier Excel de sortie
    # ---------------------------------------------------------

    wb = Workbook()

    # Feuille 1 : classes communes
    ws_common = wb.active
    ws_common.title = "Classes communes"

    ws_common.append(["Classe"])

    for class_name in common_classes:
        ws_common.append([class_name])

    # Feuille 2 : uniquement fichier 1
    ws_file1 = wb.create_sheet("Uniquement fichier 1")

    ws_file1.append(["Classe"])

    for class_name in only_file1:
        ws_file1.append([class_name])

    # Feuille 3 : uniquement fichier 2
    ws_file2 = wb.create_sheet("Uniquement fichier 2")

    ws_file2.append(["Classe"])

    for class_name in only_file2:
        ws_file2.append([class_name])

    # Sauvegarde
    wb.save(output_file)

    print(f"\nFichier généré : {output_file}")


if __name__ == "__main__":

    FILE1 = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/AQUA-IA_dataset/AQUA-IA_dataset_Nb_images_per_class.xlsx"
    FILE2 = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/AG_dataset_Nb_images_per_class.xlsx"

    OUTPUT_FILE = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/AQUA-IA_dataset/AQUA-IA_dataset_AG_common_classes.xlsx"

    compare_classes(
        FILE1,
        FILE2,
        OUTPUT_FILE
    )