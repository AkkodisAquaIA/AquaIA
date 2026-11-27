from pathlib import Path
import shutil


def merge_folders(source: Path, destination: Path):
    """
    Fusionne le contenu de source dans destination.
    En cas de fichier de même nom, le fichier source est renommé
    pour éviter tout écrasement.
    """

    for item in source.iterdir():
        target = destination / item.name

        # Si c'est un dossier
        if item.is_dir():
            if target.exists():
                merge_folders(item, target)
                item.rmdir()
            else:
                shutil.move(str(item), str(target))

        # Si c'est un fichier
        else:
            if target.exists():
                # Éviter d'écraser un fichier existant
                stem = item.stem
                suffix = item.suffix
                i = 1

                while target.exists():
                    target = destination / f"{stem}_duplicate_{i}{suffix}"
                    i += 1

                print(
                    f"    ATTENTION : fichier dupliqué détecté : {item.name}\n"
                    f"    -> renommé en : {target.name}"
                )

            shutil.move(str(item), str(target))


def rename_coeloptera_folders(root_path):
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"Le dossier n'existe pas : {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Ce n'est pas un dossier : {root}")

    renamed = 0
    merged = 0

    # list(...) pour figer la liste avant de commencer à renommer
    folders = list(root.iterdir())

    for folder in folders:

        if not folder.is_dir():
            continue

        parts = folder.name.split("_")

        # Nomenclature :
        # Embranchement_Classe_Ordre_Famille_Genre_Espèce
        if len(parts) != 6:
            print(
                f"ATTENTION : nomenclature inattendue, dossier ignoré : "
                f"{folder.name}"
            )
            continue

        # Vérification de l'ordre
        if parts[2] != "Coeloptera":
            continue

        old_name = folder.name

        parts[2] = "Coleoptera"

        new_name = "_".join(parts)
        new_path = root / new_name

        # -----------------------------------------------------
        # Cas 1 : le dossier cible existe déjà
        # -> fusion
        # -----------------------------------------------------
        if new_path.exists():

            print("\nFUSION détectée :")
            print(f"  Source      : {old_name}")
            print(f"  Destination : {new_name}")

            merge_folders(folder, new_path)

            # Supprimer le dossier source s'il est maintenant vide
            if folder.exists():
                try:
                    folder.rmdir()
                except OSError:
                    print(
                        f"ATTENTION : le dossier source n'a pas pu être "
                        f"supprimé complètement : {folder}"
                    )

            print("  -> Fusion terminée.")

            merged += 1

        # -----------------------------------------------------
        # Cas 2 : le dossier cible n'existe pas
        # -> renommage simple
        # -----------------------------------------------------
        else:

            folder.rename(new_path)

            print(f"RENOMMÉ :")
            print(f"  {old_name}")
            print(f"  -> {new_name}")

            renamed += 1

    print("\n====================================")
    print("Résumé")
    print("====================================")
    print(f"Dossiers renommés : {renamed}")
    print(f"Dossiers fusionnés : {merged}")


if __name__ == "__main__":

    root_path = (
        "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/Fusion_PERLA_AQUAIA_24juin"
    )

    rename_coeloptera_folders(root_path)