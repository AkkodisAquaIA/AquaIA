from pathlib import Path
import os

from authentication import build_headers

from graph_client import (
    get_site_info,
    get_site_drives,
    find_drive_by_name,
    list_subfolders,
    list_files,
    download_file_content,
)


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMG_EXTENSIONS


def download_sharepoint_imagefolder(
    SITE_URL: str,
    FOLDER_PATH: str,
    DOCUMENTS_DRIVE_NAME: str,
    local_dir: str,
    force_redownload: bool = False,
) -> None:
    """
    Télécharge un dataset SharePoint structuré comme ImageFolder vers un dossier local.

    Structure attendue sur SharePoint :
        FOLDER_PATH/
            class_1/
                img1.jpg
                img2.jpg
            class_2/
                img3.jpg
                ...

    Structure créée en local :
        local_dir/
            class_1/
                img1.jpg
                img2.jpg
            class_2/
                img3.jpg
                ...

    Args:
        SITE_URL: URL du site SharePoint
        FOLDER_PATH: chemin du dossier dataset dans la bibliothèque
        DOCUMENTS_DRIVE_NAME: nom de la bibliothèque, ex. "Documents"
        local_dir: dossier local de destination
        force_redownload: si True, retélécharge même les fichiers déjà présents
    """
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    headers = build_headers()

    # 1. Résoudre le site
    site = get_site_info(SITE_URL, headers)
    site_id = site["id"]

    # 2. Résoudre la bibliothèque
    drives = get_site_drives(site_id, headers)
    drive = find_drive_by_name(drives, DOCUMENTS_DRIVE_NAME)
    drive_id = drive["id"]

    # 3. Récupérer les sous-dossiers = classes
    class_folders = list_subfolders(site_id, drive_id, FOLDER_PATH, headers)

    if not class_folders:
        raise RuntimeError(
            f"Aucun sous-dossier trouvé dans '{FOLDER_PATH}'. "
            f"Structure attendue de type ImageFolder."
        )

    total_downloaded = 0
    total_skipped = 0

    for folder in class_folders:
        class_name = folder["name"]
        class_folder_path = f"{FOLDER_PATH.strip('/')}/{class_name}"
        local_class_dir = local_root / class_name
        local_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Classe: {class_name}")

        files = list_files(site_id, drive_id, class_folder_path, headers)

        for item in files:
            file_name = item.get("name", "")
            if not is_image_file(file_name):
                continue

            file_id = item["id"]
            local_file_path = local_class_dir / file_name

            if local_file_path.exists() and not force_redownload:
                total_skipped += 1
                print(f"  [SKIP] {local_file_path}")
                continue

            print(f"  [DOWNLOAD] {file_name}")
            content = download_file_content(drive_id, file_id, headers)

            with open(local_file_path, "wb") as f:
                f.write(content)

            total_downloaded += 1

    print("\n[OK] Téléchargement terminé.")
    print(f"Fichiers téléchargés : {total_downloaded}")
    print(f"Fichiers ignorés     : {total_skipped}")
    print(f"Dossier local        : {local_root.resolve()}")



if __name__ == "__main__":
    SITE_URL = "https://tobumo.sharepoint.com/teams/FRProjetDecarbonation-AQUA-IA"
    DOCUMENTS_DRIVE_NAME = "Documents"

    download_sharepoint_imagefolder(
        SITE_URL=SITE_URL,
        FOLDER_PATH="AQUA-IA/Data/datasets/train",
        DOCUMENTS_DRIVE_NAME=DOCUMENTS_DRIVE_NAME,
        local_dir="/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/sharepoint_dataset_test/train", #"/mnt/data/train",
    )

    download_sharepoint_imagefolder(
        SITE_URL=SITE_URL,
        FOLDER_PATH="AQUA-IA/Data/datasets/val",
        DOCUMENTS_DRIVE_NAME=DOCUMENTS_DRIVE_NAME,
        local_dir="/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/sharepoint_dataset_test/val", #"/mnt/data/val",
    )