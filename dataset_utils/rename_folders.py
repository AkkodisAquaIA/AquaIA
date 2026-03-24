import os

def rename_folders(path):
    # Vérifie que le chemin existe
    if not os.path.isdir(path):
        print("Le chemin spécifié n'existe pas.")
        return

    for folder_name in os.listdir(path):
        old_path = os.path.join(path, folder_name)

        # On ne traite que les dossiers
        if os.path.isdir(old_path):
            parts = folder_name.split("_")

            # Vérifie qu'il y a au moins 3 parties
            if len(parts) >= 3:
                if parts[2] == "all":
                    parts[2] = "sp"
                    new_folder_name = "_".join(parts)
                    new_path = os.path.join(path, new_folder_name)

                    # Renomme seulement si le nom change
                    if new_folder_name != folder_name:
                        os.rename(old_path, new_path)
                        print(f"Renommé : {folder_name} → {new_folder_name}")


if __name__ == "__main__":
    path = r"/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/FIN-Benthic"  # 🔁 Remplace par ton chemin
    rename_folders(path)