import os
import shutil
import re
from pathlib import Path

# =====================
# CONFIG
# =====================
SRC_ROOT = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN Benthic2/IDA/Images"
DST_ROOT = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN Benthic2/IDA/Images_per_class"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

os.makedirs(DST_ROOT, exist_ok=True)

# Regex pour extraire le nom de classe depuis le dossier
# Ex: algae12 -> algae | coral7 -> coral
CLASS_REGEX = re.compile(r"^([A-Za-z_]+)\d*$")


def extract_class_from_folder(folder_name):
    match = CLASS_REGEX.match(folder_name)
    return match.group(1) if match else None


for specimen_dir in Path(SRC_ROOT).iterdir():
    if not specimen_dir.is_dir():
        continue

    class_name = extract_class_from_folder(specimen_dir.name)
    if class_name is None:
        print(f"[WARN] Classe non détectée pour le dossier: {specimen_dir.name}")
        continue

    dst_class_dir = Path(DST_ROOT) / class_name
    dst_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in specimen_dir.iterdir():
        if img_path.suffix.lower() not in EXTENSIONS:
            continue

        dst_path = dst_class_dir / img_path.name

        if dst_path.exists():
            print(f"[SKIP] existe déjà: {dst_path}")
            continue

        shutil.copy2(img_path, dst_path)

print("Regroupement des images par classe terminé")
