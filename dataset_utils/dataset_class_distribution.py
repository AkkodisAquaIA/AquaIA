import os
import matplotlib.pyplot as plt
import pandas as pd


def extraire_noms_dossiers(fichier_excel, colonne):
    """
    fichier_excel : chemin vers le fichier Excel
    colonne : nom de la colonne contenant les noms de dossiers
    """

    # Lire le fichier Excel
    df = pd.read_excel(fichier_excel)

    # Vérifier que la colonne existe
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le fichier.")

    # Extraire les noms (en supprimant les valeurs vides)
    liste_dossiers = df[colonne].dropna().astype(str).tolist()

    return liste_dossiers


def plot_images_per_class(root_path, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    """
    Histogramme du nombre d'images par classe.
    Si le dossier n'existe pas → 0 image.
    """

    counts = []
    c = 0

    for cls in CLASSES_IA:
        folder_path = os.path.join(root_path, cls)

        if os.path.isdir(folder_path):
            n_images = sum(1 for f in os.listdir(folder_path) if f.lower().endswith(extensions))
            c += 1
        else:
            n_images = 0

        counts.append(n_images)

    print("nb class =", c)

    plt.figure(figsize=(18, 7))

    bars = plt.bar(range(len(CLASSES_IA)), counts)

    plt.xticks(ticks=range(len(CLASSES_IA)), labels=CLASSES_IA, rotation=90, ha="center", fontsize=4)

    plt.xlabel("Classes IA")
    plt.ylabel("Nombre d'images")
    plt.title("Nombre d'images par classe")

    # --- ajout du nombre d'images verticalement ---
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


# 🔹 Exemple d'utilisation
fichier = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/AQUA-IA_dataset/AQUA-IA_dataset_Nb_images_per_class.xlsx"
colonne_dossiers = "Classe"

CLASSES_IA = extraire_noms_dossiers(fichier, colonne_dossiers)

print(CLASSES_IA)

plot_images_per_class("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/NOT_USED/AQUA-IA_dataset_mars2026")
