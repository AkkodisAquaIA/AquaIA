#!/usr/bin/env python3
from pathlib import Path

# ======================
# CONFIG
# ======================
ROOT_PATH = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/PERLA"
SUFFIX = "_PERLA"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def rename_images_with_suffix(root_path):
    root = Path(root_path)

    if not root.is_dir():
        raise FileNotFoundError(f"{root_path} n'existe pas")

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        print(f"\n📂 Dossier: {folder.name}")

        for img_path in folder.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                # Si déjà suffixé → on skip
                if img_path.stem.endswith(SUFFIX):
                    continue

                new_name = img_path.stem + SUFFIX + img_path.suffix
                new_path = folder / new_name

                # gestion collision
                counter = 1
                while new_path.exists():
                    new_name = f"{img_path.stem}{SUFFIX}_{counter}{img_path.suffix}"
                    new_path = folder / new_name
                    counter += 1

                img_path.rename(new_path)
                print(f"{img_path.name} -> {new_name}")

    print("\n✅ Renommage terminé")


if __name__ == "__main__":
    rename_images_with_suffix(ROOT_PATH)
