import os

# Chemin vers ton dossier d'images
dossier = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/AQUA-IA_dataset/Fusion/Dataset/Elmidae_Elmis_aenea_adult"

# Extensions d'images à traiter
extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")

for fichier in os.listdir(dossier):
    if fichier.lower().endswith(extensions):
        ancien_chemin = os.path.join(dossier, fichier)
        
        nom, extension = os.path.splitext(fichier)
        nouveau_nom = f"{nom}_adult{extension}"
        nouveau_chemin = os.path.join(dossier, nouveau_nom)
        
        os.rename(ancien_chemin, nouveau_chemin)
        print(f"Renommé : {fichier} -> {nouveau_nom}")

print("Terminé !")