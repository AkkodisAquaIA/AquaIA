import pandas as pd
from pathlib import Path

# === PARAMÈTRES ===
dossier_racine = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/NOT_USED/AQUA-IA_dataset_mars2026")
fichier_sortie = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/AQUA-IA_dataset/AQUA-IA_dataset_Nb_images_per_class.xlsx"

# Extensions d'images à prendre en compte
extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")

# === TRAITEMENT ===
resultats = []

for sous_dossier in dossier_racine.iterdir():
    if sous_dossier.is_dir():
        nb_images = sum(
            1 for f in sous_dossier.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        resultats.append({
            "Classe": sous_dossier.name,
            "Nombre d'images": nb_images
        })

# === DATAFRAME ===
df = pd.DataFrame(resultats)

# Optionnel : trier par nombre d’images décroissant
df = df.sort_values(by="Nombre d'images", ascending=False)

# === EXPORT EXCEL ===
df.to_excel(fichier_sortie, index=False)

print(f"Fichier généré : {fichier_sortie}")