from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AG_dataset_genus_09-29")
OUTDIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Docs/FinBenthic_1-2_PERLA_AG_per_genre")

# Extensions d'images acceptées
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp", ".webp"
}

# ============================================================
# IDENTIFICATION DU DATASET
# ============================================================

def get_dataset(filename):
    """
    Détermine le dataset d'origine à partir du nom du fichier.
    """

    name = filename.lower()

    # IMPORTANT : FIN-Benthic2 doit être testé avant FIN-Benthic
    if "fin-benthic2" in name:
        return "FIN-Benthic2"

    elif "fin-benthic" in name:
        return "FIN-Benthic"

    elif "perla" in name:
        return "PERLA"

    return "INCONNU"


# ============================================================
# CONSTRUCTION DE LA MATRICE
# ============================================================

rows = []

# Chaque sous-dossier correspond à une classe
for class_dir in sorted(DATASET_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    counts = {
        "Classe": class_name,
        "PERLA": 0,
        "FIN-Benthic": 0,
        "FIN-Benthic2": 0,
        "INCONNU": 0
    }

    for image_path in class_dir.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        dataset = get_dataset(image_path.name)

        counts[dataset] += 1

    rows.append(counts)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

# Nombre total d'images par classe
df["TOTAL"] = (
    df["PERLA"]
    + df["FIN-Benthic"]
    + df["FIN-Benthic2"]
    + df["INCONNU"]
)

# Tri par nom de classe
df = df.sort_values("Classe").reset_index(drop=True)


# ============================================================
# AFFICHAGE
# ============================================================

print("\n=== MATRICE CLASSE × DATASET ===\n")

print(df.to_string(index=False))


# ============================================================
# RÉSUMÉ PAR DATASET
# ============================================================

print("\n=== TOTAL PAR DATASET ===\n")

for dataset in ["PERLA", "FIN-Benthic", "FIN-Benthic2", "INCONNU"]:
    print(f"{dataset:15s} : {df[dataset].sum():6d} images")

print(f"{'TOTAL':15s} : {df['TOTAL'].sum():6d} images")


# ============================================================
# NOMBRE DE DATASETS REPRÉSENTÉS PAR CLASSE
# ============================================================

dataset_columns = ["PERLA", "FIN-Benthic", "FIN-Benthic2"]

df["Nb_datasets"] = (df[dataset_columns] > 0).sum(axis=1)


# ============================================================
# EXPORT CSV
# ============================================================

output_csv = OUTDIR / "matrice_class_per_dataset.csv"

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nMatrice sauvegardée dans :\n{output_csv}")