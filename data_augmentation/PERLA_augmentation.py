import shutil
from pathlib import Path

import cv2
import albumentations as A

DOSSIER_ENTREE = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/PERLA"
DOSSIER_SORTIE = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/PERLA_augmented"
OBJECTIF_PAR_CLASSE = 20
COPIER_ORIGINAUX = True
TAILLE_IMAGE = (224, 224)
EXTENSIONS_VALIDES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def est_image(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONS_VALIDES


def creer_dossier(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def construire_pipeline():
    transforms = []

    if TAILLE_IMAGE is not None:
        h, w = TAILLE_IMAGE
        transforms.append(A.Resize(height=h, width=w))

    transforms.extend(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(p=0.15),
        ]
    )

    return A.Compose(transforms)


def lire_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Impossible de lire {path}")
    return image


def sauvegarder(path: Path, image):
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Impossible de sauvegarder {path}")


def equilibrer_dataset():
    entree = Path(DOSSIER_ENTREE)
    sortie = Path(DOSSIER_SORTIE)
    creer_dossier(sortie)

    pipeline = construire_pipeline()

    for classe_dir in sorted(entree.iterdir()):
        if not classe_dir.is_dir():
            continue

        images = [p for p in classe_dir.iterdir() if p.is_file() and est_image(p)]
        if not images:
            continue

        sortie_classe = sortie / classe_dir.name
        creer_dossier(sortie_classe)

        # copier originaux
        nb_existantes = 0
        if COPIER_ORIGINAUX:
            for img_path in images:
                shutil.copy2(img_path, sortie_classe / img_path.name)
                nb_existantes += 1

        # combien faut-il générer ?
        reste = max(0, OBJECTIF_PAR_CLASSE - nb_existantes)
        if reste == 0:
            continue

        i = 1
        idx_source = 0
        while reste > 0:
            img_path = images[idx_source % len(images)]
            image = lire_image(img_path)
            image_aug = pipeline(image=image)["image"]

            nom = f"{img_path.stem}_aug_balanced_{i}{img_path.suffix.lower()}"
            sauvegarder(sortie_classe / nom, image_aug)

            i += 1
            idx_source += 1
            reste -= 1

        print(f"Classe {classe_dir.name} équilibrée à {OBJECTIF_PAR_CLASSE} images.")


if __name__ == "__main__":
    equilibrer_dataset()
