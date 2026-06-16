from pathlib import Path
import cv2
import numpy as np

def score_image(path):
    img = cv2.imread(str(path))

    if img is None:
        return None

    # résolution
    h, w = img.shape[:2]
    resolution = w * h

    # netteté (variance du Laplacien)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # taille fichier
    size = path.stat().st_size

    # score combiné (à ajuster)
    score = resolution * 0.5 + sharpness * 1000 + size * 0.0001

    return {
        "fichier": path.name,
        "resolution": resolution,
        "sharpness": sharpness,
        "size": size,
        "score": score
    }


def meilleure_image(dossier):
    dossier = Path(dossier)
    resultats = []

    for f in dossier.iterdir():
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            res = score_image(f)
            if res:
                resultats.append(res)

    resultats.sort(key=lambda x: x["score"], reverse=True)
    return resultats


images = meilleure_image("/home/sarah.laroui/Bureau/AQUA-IA/Docs/protocole_prise_photos/Test_photos_Isabelle_20avril2026/session2/ISO_160_F14_vit_diff/tiff_output")

for img in images:
    print(img)