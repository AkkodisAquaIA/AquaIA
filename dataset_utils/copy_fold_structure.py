import os


def copy_folder_structure(src_path, dst_path):
    # Vérifie que le dossier source existe
    if not os.path.exists(src_path):
        print(f"Le chemin source n'existe pas : {src_path}")
        return

    # Crée le dossier destination s'il n'existe pas
    os.makedirs(dst_path, exist_ok=True)

    # Parcours de l'arborescence
    for root, dirs, files in os.walk(src_path):
        # Calcul du chemin relatif depuis la source
        relative_path = os.path.relpath(root, src_path)

        # Reconstitution du chemin dans la destination
        target_dir = os.path.join(dst_path, relative_path)

        # Création du dossier correspondant
        os.makedirs(target_dir, exist_ok=True)

        # Création des sous-dossiers
        for d in dirs:
            os.makedirs(os.path.join(target_dir, d), exist_ok=True)

    print("Structure des dossiers copiée avec succès !")


# Exemple d'utilisation
if __name__ == "__main__":
    source = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/PERLA_cropped_Sarah_20042026_vuLPL"
    destination = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Nicolas_folder"

    copy_folder_structure(source, destination)
