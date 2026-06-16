import os

def check_folder_names(path):
    # Vérifie que le chemin existe
    if not os.path.isdir(path):
        print("Chemin invalide")
        return

    # Parcours des dossiers
    for name in os.listdir(path):
        full_path = os.path.join(path, name)

        # On ne traite que les dossiers
        if os.path.isdir(full_path):
            words = name.split("_")
            count = len(words)

            # On veut exactement 6 mots → sinon on affiche
            if count != 6:
                diff = count - 6
                print(f"{name} → {count} mots ({diff:+} par rapport à 6)")


# Exemple d'utilisation
path = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/PERLA_cropped"
check_folder_names(path)