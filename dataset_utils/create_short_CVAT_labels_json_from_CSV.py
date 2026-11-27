#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("/home/sarah.laroui/Bureau/AQUA-IA/Docs/FinBenthic_1-2_PERLA_AG_per_genre/AG_dataset_genus_Nb_images_per_class.xlsx")
OUTPUT_FILE = Path("/home/sarah.laroui/Bureau/AQUA-IA/Docs/FinBenthic_1-2_PERLA_AG_per_genre/labels_AG_dataset_genus_09-29.json")


# Première colonne du fichier Excel = nom de la classe
CLASS_COL = 0


# ============================================================
# FONCTION DE TRANSFORMATION
# ============================================================


def shorten_label(label):
    """
    Raccourcit uniquement les deux premiers niveaux
    taxonomiques à 5 caractères :

    Embranchement_Classe_Ordre_Famille_Genre_Espece

    Exemple :

    Arthropoda_Insecta_Trichoptera_Polycentropodidae_
    Polycentropus_flavomaculatus

    devient :

    Arthr_Insec_Trichoptera_Polycentropodidae_
    Polycentropus_flavomaculatus
    """

    label = str(label).strip()

    parts = label.split("_")

    if len(parts) < 2:
        print(f"WARNING : nomenclature invalide : '{label}'")
        return None

    # Raccourcissement Embranchement
    parts[0] = parts[0][:5]

    # Raccourcissement Classe
    parts[1] = parts[1][:5]

    # Reconstruction du label
    short_label = "_".join(parts)

    return short_label


# ============================================================
# LECTURE DU FICHIER EXCEL
# ============================================================

print("=" * 70)
print("CRÉATION DU FICHIER JSON POUR CVAT")
print("=" * 70)

print(f"\nLecture de : {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

class_column = df.columns[CLASS_COL]

print(f"Colonne utilisée : {class_column}")
print(f"Nombre de classes dans l'Excel : {len(df)}")


# ============================================================
# CRÉATION DES LABELS
# ============================================================

labels = []

for original_label in df[class_column]:
    # Ignore les cellules vides
    if pd.isna(original_label):
        continue

    short_label = shorten_label(original_label)

    if short_label is None:
        continue

    labels.append({"name": short_label, "type": "any", "attributes": []})


# ============================================================
# VÉRIFICATION DES DOUBLONS
# ============================================================

label_names = [item["name"] for item in labels]

duplicates = sorted({label for label in label_names if label_names.count(label) > 1})


if duplicates:
    print("\n" + "=" * 70)
    print("WARNING : LABELS EN DOUBLE")
    print("=" * 70)

    for label in duplicates:
        print(f"  - {label}")

else:
    print("\nAucun doublon détecté.")


# ============================================================
# STATISTIQUES SUR LA LONGUEUR DES LABELS
# ============================================================

if labels:
    lengths = [len(item["name"]) for item in labels]

    print("\nLongueur des labels :")

    print(f"  Minimum : {min(lengths)} caractères")

    print(f"  Maximum : {max(lengths)} caractères")

    print(f"  Moyenne : {sum(lengths) / len(lengths):.1f} caractères")

    # Recherche du label le plus long
    longest_label = max(labels, key=lambda x: len(x["name"]))

    print("\nLabel le plus long :")

    print(f"  {longest_label['name']}")

    print(f"  -> {len(longest_label['name'])} caractères")


# ============================================================
# EXPORT JSON
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=4)


# ============================================================
# RÉSUMÉ
# ============================================================

print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)

print(f"Nombre de labels créés : {len(labels)}")
print(f"Fichier JSON           : {OUTPUT_FILE}")
