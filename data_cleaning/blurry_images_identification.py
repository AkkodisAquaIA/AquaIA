#!/usr/bin/env python3
import cv2
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ======================
# CONFIG
# ======================
DATASET_PATH = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/FIN_Benthic_cleaned"
THRESHOLD = 10  # seuil de flou


def blur_score(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()


def move_blurry_images(dataset_path, threshold):

    dataset_path = Path(dataset_path)
    outliers_root = dataset_path / "outliers_flou"

    outliers_root.mkdir(exist_ok=True)

    class_folders = [p for p in dataset_path.iterdir() if p.is_dir() and p.name != "outliers_flou"]

    print("Nombre de classes :", len(class_folders))

    total_blurry = 0

    for class_folder in class_folders:

        print(f"\nClasse : {class_folder.name}")

        images = [p for p in class_folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS]

        blurry_count = 0

        for img_path in images:

            score = blur_score(img_path)
            print(f"flou -> {img_path.name} | score={score:.2f}")

            if score is None:
                continue

            if score < threshold:

                dst_dir = outliers_root / class_folder.name
                dst_dir.mkdir(parents=True, exist_ok=True)

                dst_path = dst_dir / img_path.name

                shutil.move(str(img_path), str(dst_path))

                blurry_count += 1
                total_blurry += 1

                print(f"flou -> {img_path.name} | score={score:.2f}")

        print(f"images floues déplacées : {blurry_count}/{len(images)}")

    print("\n=================================")
    print("Total images floues :", total_blurry)
    print("Dossier outliers :", outliers_root)


if __name__ == "__main__":
    move_blurry_images(DATASET_PATH, THRESHOLD)