import shutil
from pathlib import Path

# =====================
# CONFIG
# =====================
SRC_ROOT = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN-Benthic/Cropped images"
DST_ROOT = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Dossier_test"

EXTENSIONS = {".png"}

Path(DST_ROOT).mkdir(parents=True, exist_ok=True)

for subdir in Path(SRC_ROOT).iterdir():
    if not subdir.is_dir():
        continue

    # Liste des images dans le sous-dossier
    images = [p for p in subdir.iterdir() if p.suffix.lower() in EXTENSIONS]

    if not images:
        print(f"[SKIP] Aucun fichier image dans {subdir.name}")
        continue

    # On prend la première image (tri pour stabilité)
    img = sorted(images)[0]

    # Nouveau nom : <nom_dossier>_<nom_image>
    new_name = f"{subdir.name}_{img.name}"
    dst_path = Path(DST_ROOT) / new_name

    if dst_path.exists():
        print(f"[SKIP] Existe déjà: {dst_path}")
        continue

    shutil.copy2(img, dst_path)
    print(f"[OK] {img.name} -> {new_name}")

print("Copie terminée")
