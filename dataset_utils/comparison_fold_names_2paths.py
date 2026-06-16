import os

import os

# Extensions d'images à prendre en compte
EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

def compter_images(dossier):
    count = 0
    for fichier in os.listdir(dossier):
        chemin = os.path.join(dossier, fichier)
        if os.path.isfile(chemin):
            _, ext = os.path.splitext(fichier)
            if ext.lower() in EXTENSIONS_IMAGES:
                count += 1
    return count

def comparer_dossiers(path1, path2):
    # Vérification des chemins
    if not os.path.exists(path1):
        print(f"❌ Le chemin n'existe pas : {path1}")
        return
    if not os.path.exists(path2):
        print(f"❌ Le chemin n'existe pas : {path2}")
        return

    # Récupérer uniquement les dossiers
    dossiers1 = {d for d in os.listdir(path1) if os.path.isdir(os.path.join(path1, d))}
    dossiers2 = {d for d in os.listdir(path2) if os.path.isdir(os.path.join(path2, d))}

    # Comparaisons
    uniquement_dans_1 = dossiers1 - dossiers2
    uniquement_dans_2 = dossiers2 - dossiers1
    communs = dossiers1 & dossiers2

    # # Résultats
    # print("\n Dossiers uniquement dans le premier chemin :")
    # for d in sorted(uniquement_dans_1):
    #     print(f" - {d}")

    # print("\n Dossiers uniquement dans le second chemin :")
    # for d in sorted(uniquement_dans_2):
    #     print(f" - {d}")
    print(f"\n📊 Nombre de dossiers en commun : {len(communs)}")
    print("\nDossiers communs :")
    for d in sorted(communs):
        chemin1 = os.path.join(path1, d)
        chemin2 = os.path.join(path2, d)

        nb1 = compter_images(chemin1)
        nb2 = compter_images(chemin2)

        print(f"{d}")
        print(f"   - Images dans path1 : {nb1}")
        print(f"   - Images dans path2 : {nb2}")


    


if __name__ == "__main__":
    # 👉 MODIFIE CES CHEMINS
    path1 = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/PERLA_cropped"
    path2 = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026"

    comparer_dossiers(path1, path2)