import argparse
from pathlib import Path
import shutil


def build_short_name(folder_name: str) -> str | None:
    """
    Convertit :
    Embranchement_Classe_Ordre_Famille_Genre_espece
    en :
    Famille_Genre_espece

    Retourne None si le nom n'a pas assez de segments.
    """
    parts = folder_name.strip().split("_")

    if len(parts) < 6:
        return None

    family, genus, species = parts[-3], parts[-2], parts[-1]
    return f"{family}_{genus}_{species}"


def rename_class_folders(root_dir: str, dry_run: bool = False, copy_instead_of_rename: bool = False) -> None:
    """
    Renomme ou copie les dossiers d'un répertoire racine
    vers la nomenclature Famille_Genre_espece.
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
        new_name = build_short_name(old_name)

        if new_name is None:
            print(f"[SKIP] Nom invalide ou incomplet : {old_name}")
            skipped += 1
            continue

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
            action = "COPY" if copy_instead_of_rename else "RENAME"
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
    parser = argparse.ArgumentParser(description="Renomme des dossiers/classes de Embranchement_Classe_Ordre_Famille_Genre_espece vers Famille_Genre_espece.")
    parser.add_argument("--root-dir", required=True, help="Dossier racine contenant les dossiers/classes")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les changements sans rien modifier")
    parser.add_argument("--copy", action="store_true", help="Copie les dossiers au lieu de les renommer")

    args = parser.parse_args()

    rename_class_folders(root_dir=args.root_dir, dry_run=args.dry_run, copy_instead_of_rename=args.copy)


if __name__ == "__main__":
    main()
