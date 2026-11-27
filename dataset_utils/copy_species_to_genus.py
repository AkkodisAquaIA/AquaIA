#!/usr/bin/env python3

from pathlib import Path
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

# Dossier contenant les dossiers par ESPÈCE :
#
# Embranchement_Classe_Ordre_Famille_Genre_Espece/
#
SOURCE_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/Fusion_PERLA_AQUAIA_2sept")


# Dossier contenant les dossiers par GENRE :
#
# Embranchement_Classe_Ordre_Famille_Genre/
#
DESTINATION_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AG_dataset_genus_09-29")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


# ============================================================
# FONCTIONS
# ============================================================

def get_genus_from_species_folder(folder_name):
    """
    Transforme :

    Embranchement_Classe_Ordre_Famille_Genre_Espece

    en :

    Embranchement_Classe_Ordre_Famille_Genre
    """

    parts = folder_name.split("_")

    if len(parts) < 6:
        return None

    return "_".join(parts[:5])


def get_unique_destination_path(destination):
    """
    Évite d'écraser une image existante.

    image.jpg
    image_1.jpg
    image_2.jpg
    ...
    """

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    counter = 1

    while True:

        new_destination = (
            destination.parent
            / f"{stem}_{counter}{suffix}"
        )

        if not new_destination.exists():
            return new_destination

        counter += 1


# ============================================================
# VÉRIFICATION DES PATHS
# ============================================================

if not SOURCE_DIR.is_dir():
    raise FileNotFoundError(
        f"Dossier source introuvable : {SOURCE_DIR}"
    )

if not DESTINATION_DIR.is_dir():
    raise FileNotFoundError(
        f"Dossier destination introuvable : {DESTINATION_DIR}"
    )


# ============================================================
# RÉCUPÉRATION DES DOSSIERS GENRES EXISTANTS
# ============================================================

existing_genus_directories = {
    folder.name: folder
    for folder in DESTINATION_DIR.iterdir()
    if folder.is_dir()
}


print("=" * 70)
print("COPIE DES IMAGES ESPÈCES -> GENRES")
print("=" * 70)

print(f"\nSource      : {SOURCE_DIR}")
print(f"Destination : {DESTINATION_DIR}")

print(
    f"\n{len(existing_genus_directories)} "
    f"dossiers de classes disponibles."
)


# ============================================================
# COMPTEURS
# ============================================================

species_count = 0
matched_species_count = 0
ignored_species_count = 0
invalid_folder_count = 0
copied_images_count = 0

ignored_genus = set()


# ============================================================
# PARCOURS DES DOSSIERS ESPÈCES
# ============================================================

for species_directory in sorted(SOURCE_DIR.iterdir()):

    if not species_directory.is_dir():
        continue

    species_count += 1

    species_name = species_directory.name

    # --------------------------------------------------------
    # Récupération du genre
    # --------------------------------------------------------

    genus_name = get_genus_from_species_folder(
        species_name
    )

    if genus_name is None:

        print(
            f"\nWARNING : nomenclature invalide :\n"
            f"  {species_name}"
        )

        invalid_folder_count += 1
        continue


    # --------------------------------------------------------
    # Vérification que le dossier genre existe
    # --------------------------------------------------------

    if genus_name not in existing_genus_directories:

        print(
            f"\nIGNORÉ : classe absente du dataset destination"
            f"\n  Espèce : {species_name}"
            f"\n  Genre  : {genus_name}"
        )

        ignored_species_count += 1
        ignored_genus.add(genus_name)

        # IMPORTANT :
        # aucune image n'est copiée
        continue


    # --------------------------------------------------------
    # Le genre existe -> copie autorisée
    # --------------------------------------------------------

    destination_directory = (
        existing_genus_directories[genus_name]
    )

    matched_species_count += 1


    # --------------------------------------------------------
    # Recherche des images
    # --------------------------------------------------------

    images = [
        file
        for file in species_directory.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]


    print(
        f"\nOK : {species_name}"
        f"\n  -> {genus_name}"
        f"\n  -> {len(images)} image(s) à copier"
    )


    # --------------------------------------------------------
    # Copie
    # --------------------------------------------------------

    for image_path in images:

        destination = (
            destination_directory
            / image_path.name
        )

        # Protection contre les noms identiques
        destination = get_unique_destination_path(
            destination
        )

        shutil.copy2(
            image_path,
            destination
        )

        copied_images_count += 1


# ============================================================
# RÉSUMÉ
# ============================================================

print("\n")
print("=" * 70)
print("RÉSUMÉ")
print("=" * 70)

print(
    f"Dossiers espèces analysés      : "
    f"{species_count}"
)

print(
    f"Espèces copiées                : "
    f"{matched_species_count}"
)

print(
    f"Espèces ignorées               : "
    f"{ignored_species_count}"
)

print(
    f"Dossiers invalides             : "
    f"{invalid_folder_count}"
)

print(
    f"Images copiées                 : "
    f"{copied_images_count}"
)


# ============================================================
# AFFICHAGE DES GENRES IGNORÉS
# ============================================================

if ignored_genus:

    print("\n" + "=" * 70)
    print("CLASSES ABSENTES DE LA DESTINATION")
    print("AUCUNE IMAGE N'A ÉTÉ COPIÉE POUR CES CLASSES")
    print("=" * 70)

    for genus in sorted(ignored_genus):
        print(f"  - {genus}")


print("\nTerminé.")