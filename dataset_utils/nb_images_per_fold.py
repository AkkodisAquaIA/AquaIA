import os


def count_total_images(root_path, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    total = 0

    for root, dirs, files in os.walk(root_path):
        total += sum(1 for f in files if f.lower().endswith(extensions))

    return total


# Exemple d'utilisation
path = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026_splited/test"
total_images = count_total_images(path)

print(f"Nombre total d'images : {total_images}")
