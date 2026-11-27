#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/Liste_LPL_taxons_AG_cleaned_09-29.xlsx")

OUTPUT_EXCEL = Path("/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/classes_AG_ALL_taxons_LPL.xlsx")
OUTPUT_JSON = Path("/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/classes_AG_ALL_taxons.json")

# Dossier dans lequel seront créés les dossiers de classes
OUTPUT_CLASSES_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AG_dataset_genus_09-29")


# ============================================================
# VALEURS À REMPLACER
# ============================================================

REPLACEMENTS = {
    # Genre
    "sans genre spécifié": "Genus",
    # Famille
    "sans famille spécifiée": "Familia",
    # Ordre
    "sans ordre spécifié": "Order",
    "sans ordre taxo": "Order",
    # Classe
    "sans classe spécifiée": "Classis",
}


# ============================================================
# FONCTIONS
# ============================================================


def clean_taxon(value):
    """
    Nettoie une valeur taxonomique et applique les remplacements.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Comparaison insensible à la casse
    replacement = REPLACEMENTS.get(value.lower())

    if replacement is not None:
        return replacement

    return value


def create_full_label(row):
    """
    Crée :
    Embranchement_Classe_Ordre_Famille_Genre
    """

    parts = [
        row["EMBRANCHEMENT"],
        row["CLASSE"],
        row["ORDRE"],
        row["FAMILLE"],
        row["GENRE"],
    ]

    return "_".join(parts)


def create_short_label(row):
    """
    Raccourcit uniquement Embranchement et Classe
    à leurs 5 premiers caractères.

    Exemple :

    Annelida_Clitellata_Arhynchobdellida_Erpobdellidae_Trocheta

    devient :

    Annel_Clite_Arhynchobdellida_Erpobdellidae_Trocheta
    """

    parts = [
        row["EMBRANCHEMENT"][:5],
        row["CLASSE"][:5],
        row["ORDRE"],
        row["FAMILLE"],
        row["GENRE"],
    ]

    return "_".join(parts)


# ============================================================
# LECTURE DU FICHIER EXCEL
# ============================================================

print(f"Lecture de : {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

print(f"Nombre de lignes initiales : {len(df)}")


# ============================================================
# VÉRIFICATION DES COLONNES
# ============================================================

required_columns = [
    "REGNE",
    "EMBRANCHEMENT",
    "CLASSE",
    "ORDRE",
    "FAMILLE",
    "GENRE",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError("Colonnes manquantes dans le fichier Excel : " + ", ".join(missing_columns))


# ============================================================
# NETTOYAGE DES VALEURS TAXONOMIQUES
# ============================================================

taxonomy_columns = [
    "EMBRANCHEMENT",
    "CLASSE",
    "ORDRE",
    "FAMILLE",
    "GENRE",
]

for column in taxonomy_columns:
    df[column] = df[column].apply(clean_taxon)


# ============================================================
# VÉRIFICATION DES VALEURS VIDES
# ============================================================

print("\nVérification des valeurs manquantes...")

for column in taxonomy_columns:
    empty_mask = df[column] == ""
    n_empty = empty_mask.sum()

    if n_empty > 0:
        print(f"WARNING : {n_empty} valeur(s) vide(s) dans la colonne {column}")


# ============================================================
# CRÉATION DES LABELS COMPLETS
# ============================================================

df["label"] = df.apply(create_full_label, axis=1)


# ============================================================
# SUPPRESSION DES DOUBLONS
# ============================================================

n_before = len(df)

df_unique = df.drop_duplicates(subset=["label"]).copy()

n_duplicates = n_before - len(df_unique)

if n_duplicates > 0:
    print(f"\n{n_duplicates} doublon(s) supprimé(s).")


# ============================================================
# CRÉATION DES LABELS COURTS
# ============================================================

df_unique["short_label"] = df_unique.apply(create_short_label, axis=1)

df_unique["length"] = df_unique["short_label"].str.len()


# ============================================================
# VÉRIFICATION DES COLLISIONS
# ============================================================

collision_mask = df_unique["short_label"].duplicated(keep=False)

collisions = df_unique[collision_mask]

if not collisions.empty:
    print("\n" + "=" * 60)
    print("WARNING : COLLISIONS DE LABELS COURTS")
    print("=" * 60)

    for short_label, group in collisions.groupby("short_label"):
        print(f"\nLabel court : {short_label}")

        for label in group["label"]:
            print(f"  -> {label}")

else:
    print("\nAucune collision après raccourcissement.")


# ============================================================
# EXPORT EXCEL
# ============================================================

excel_output = df_unique[
    [
        "label",
        "short_label",
        "length",
    ]
].copy()

excel_output.columns = [
    "Classe",
    "Classe_short",
    "Nombre_caracteres",
]

excel_output.to_excel(OUTPUT_EXCEL, index=False)


# ============================================================
# EXPORT JSON
# ============================================================

json_classes = df_unique["short_label"].tolist()

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_classes, f, ensure_ascii=False, indent=4)


# ============================================================
# CRÉATION DES DOSSIERS DE CLASSES
# ============================================================

print("\nCréation des dossiers de classes...")

# Création du dossier principal
OUTPUT_CLASSES_DIR.mkdir(parents=True, exist_ok=True)

created_count = 0

for label in df_unique["label"]:
    class_directory = OUTPUT_CLASSES_DIR / label

    # exist_ok=True permet de ne pas générer d'erreur
    # si le dossier existe déjà
    class_directory.mkdir(parents=True, exist_ok=True)

    created_count += 1

print(f"{created_count} dossier(s) de classes disponibles dans : {OUTPUT_CLASSES_DIR}")


# ============================================================
# STATISTIQUES
# ============================================================

print("\n" + "=" * 60)
print("RÉSULTATS")
print("=" * 60)

print(f"Nombre de classes : {len(df_unique)}")

if len(df_unique) > 0:
    print(f"Longueur minimale : {df_unique['length'].min()} caractères")

    print(f"Longueur maximale : {df_unique['length'].max()} caractères")

    print(f"Longueur moyenne  : {df_unique['length'].mean():.1f} caractères")

    longest = df_unique.loc[df_unique["length"].idxmax()]

    print("\nLabel le plus long :")
    print(f"  {longest['short_label']}")
    print(f"  -> {longest['length']} caractères")


# ============================================================
# RÉCAPITULATIF
# ============================================================

print("\n" + "=" * 60)
print("FICHIERS / DOSSIERS CRÉÉS")
print("=" * 60)

print(f"Excel            : {OUTPUT_EXCEL}")
print(f"JSON             : {OUTPUT_JSON}")
print(f"Dossier classes  : {OUTPUT_CLASSES_DIR}")

print("\nTerminé.")
