import shutil
from pathlib import Path
from openpyxl import load_workbook


def copy_folders_from_excel(excel_path, source_path, output_path):
    excel_path = Path(excel_path)
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"Fichier Excel introuvable : {excel_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {source_path}")

    if not source_path.is_dir():
        raise NotADirectoryError(f"Ce n'est pas un dossier : {source_path}")

    # Création du dossier de sortie s'il n'existe pas
    output_path.mkdir(parents=True, exist_ok=True)

    # Lecture du fichier Excel
    wb = load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active

    folder_names = []

    # Première colonne = colonne A
    # min_row=2 suppose que la première ligne est un en-tête
    for row in ws.iter_rows(
        min_row=2,
        min_col=1,
        max_col=1,
        values_only=True
    ):
        value = row[0]

        if value is not None:
            folder_name = str(value).strip()

            if folder_name:
                folder_names.append(folder_name)

    wb.close()

    copied = []
    not_found = []
    already_existing = []

    for folder_name in folder_names:

        source_folder = source_path / folder_name
        destination_folder = output_path / folder_name

        # Dossier non trouvé
        if not source_folder.exists() or not source_folder.is_dir():
            not_found.append(folder_name)
            continue

        # Dossier déjà présent dans le dossier de sortie
        if destination_folder.exists():
            already_existing.append(folder_name)
            continue

        # Copie du dossier complet
        shutil.copytree(source_folder, destination_folder)

        copied.append(folder_name)

        print(f"Copié : {folder_name}")

    # ---------------------------------------------------------
    # Résumé
    # ---------------------------------------------------------

    print("\n========================================")
    print("RÉSUMÉ")
    print("========================================")

    print(f"Noms lus dans l'Excel : {len(folder_names)}")
    print(f"Dossiers copiés : {len(copied)}")
    print(f"Dossiers introuvables : {len(not_found)}")
    print(f"Dossiers déjà présents : {len(already_existing)}")

    if not_found:
        print("\nDossiers NON TROUVÉS :")

        for folder_name in not_found:
            print(f"  - {folder_name}")

    if already_existing:
        print("\nDossiers déjà présents dans la sortie :")

        for folder_name in already_existing:
            print(f"  - {folder_name}")


if __name__ == "__main__":

    excel_path = (
        "/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/classes_presentes_dans_taxons.xlsx"
    )

    source_path = (
        "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/Fusion_PERLA_AQUAIA_2sept"
    )

    output_path = (
        "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AG_dataset"
    )

    copy_folders_from_excel(
        excel_path,
        source_path,
        output_path
    )