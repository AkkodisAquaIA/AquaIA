import argparse
from pathlib import Path
import pandas as pd
import shutil


OLD_COL = "Classes IA (Famille_Genre_espece)"
PHYLUM_COL = "EMBRANCHEMENT"
CLASS_COL = "CLASSE"
ORDER_COL = "ORDRE"


def normalize_value(value, default):
    """
    Nettoie une valeur issue du tableau Excel.
    Si vide / NaN, retourne la valeur par défaut.
    """
    if pd.isna(value):
        return default

    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return default

    return value


def normalize_old_class_name(value):
    """
    Nettoie la clé de correspondance de l'ancienne nomenclature.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    return value if value else None


def build_new_name(row):
    """
    Construit le nouveau nom :
    Embranchement_Classe_Ordre_Famille_Genre_espece

    La partie Famille_Genre_espece est gardée telle quelle
    depuis la colonne OLD_COL.
    """
    phylum = normalize_value(row.get(PHYLUM_COL), "UnknownPhylum")
    class_name = normalize_value(row.get(CLASS_COL), "Classis")
    order = normalize_value(row.get(ORDER_COL), "Order")

    old_name = normalize_old_class_name(row.get(OLD_COL))
    if old_name is None:
        return None

    return f"{phylum}_{class_name}_{order}_{old_name}"


def load_mapping(excel_path):
    """
    Charge le fichier Excel et construit un mapping :
    ancien_nom -> nouveau_nom
    """
    df = pd.read_excel(excel_path)

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()

    required_cols = [OLD_COL, PHYLUM_COL, CLASS_COL, ORDER_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier Excel : {missing}")

    # Gère le cas fréquent des cellules fusionnées dans Excel
    df[required_cols] = df[required_cols].ffill()

    mapping = {}
    duplicates = []

    for _, row in df.iterrows():
        old_name = normalize_old_class_name(row[OLD_COL])
        if not old_name:
            continue

        new_name = build_new_name(row)
        if not new_name:
            continue

        if old_name in mapping and mapping[old_name] != new_name:
            duplicates.append(old_name)

        mapping[old_name] = new_name

    if duplicates:
        raise ValueError("Plusieurs correspondances incohérentes trouvées pour : " + ", ".join(sorted(set(duplicates))))

    return mapping


def rename_class_folders(root_dir, mapping, dry_run=False, copy_instead_of_rename=False):
    """
    Renomme ou copie les dossiers du dossier racine selon le mapping.
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dossier invalide : {root}")

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        print("Aucun sous-dossier trouvé.")
        return

    renamed = 0
    skipped = 0

    for class_dir in sorted(class_dirs):
        old_name = class_dir.name

        if old_name not in mapping:
            print(f"[SKIP] Aucun mapping trouvé pour : {old_name}")
            skipped += 1
            continue

        new_name = mapping[old_name]
        target_dir = root / new_name

        if class_dir.resolve() == target_dir.resolve():
            print(f"[SKIP] Déjà au bon nom : {old_name}")
            skipped += 1
            continue

        if target_dir.exists():
            print(f"[SKIP] Le dossier cible existe déjà : {target_dir.name}")
            skipped += 1
            continue

        if dry_run:
            action = "COPIE" if copy_instead_of_rename else "RENAME"
            print(f"[DRY-RUN][{action}] {old_name} -> {new_name}")
        else:
            if copy_instead_of_rename:
                shutil.copytree(class_dir, target_dir)
                print(f"[COPY] {old_name} -> {new_name}")
            else:
                class_dir.rename(target_dir)
                print(f"[RENAME] {old_name} -> {new_name}")
            renamed += 1

    print(f"\nTerminé. Modifiés : {renamed} | Ignorés : {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Renomme des dossiers/classes d'images à partir d'un tableau Excel.")
    parser.add_argument("--root-dir", required=True, help="Dossier racine contenant les dossiers/classes d'images")
    parser.add_argument("--excel", required=True, help="Chemin vers le fichier Excel")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les changements sans rien modifier")
    parser.add_argument("--copy", action="store_true", help="Copie les dossiers au lieu de les renommer")

    args = parser.parse_args()

    mapping = load_mapping(args.excel)
    rename_class_folders(root_dir=args.root_dir, mapping=mapping, dry_run=args.dry_run, copy_instead_of_rename=args.copy)


if __name__ == "__main__":
    main()
