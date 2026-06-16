import shutil
import pandas as pd
from pathlib import Path

# === PARAMÈTRES ===
fichier_excel = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/PERLA/PERLA_taxo_info.xlsx"
dossier_images = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/PERLA_IMAGES_individus_entiers")
dossier_sortie = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/PERLA")
fichier_excel_sortie = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/PERLA/PERLA_taxo_info_classeIA.xlsx"


# === LECTURE EXCEL ===
df = pd.read_excel(fichier_excel)

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# Vérification des colonnes nécessaires
colonnes_requises = ["filename", "EMBRANCHEMENT", "CLASSE", "ORDRE", "FAMILLE", "GENRE", "ESPECE"]
for col in colonnes_requises:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante dans le fichier Excel : {col}")

# Ajout de la colonne si elle n'existe pas
if "Classe IA" not in df.columns:
    df["Classe IA"] = ""

# Fonction pour nettoyer les noms de dossiers
def nettoyer_nom(valeur, fallback=None):
    if pd.isna(valeur) or str(valeur).strip() == "":
        return fallback if fallback else "Inconnu"
    valeur = str(valeur).strip()
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        valeur = valeur.replace(char, "_")
    return valeur

# === TRAITEMENT ===
for index, row in df.iterrows():
    filename = row["filename"]

    if pd.isna(filename) or str(filename).strip() == "":
        print(f"Ligne {index + 2}: filename vide, ignorée")
        df.at[index, "Classe IA"] = "Non traité"
        continue

    filename = str(filename).strip()

    embranchement = nettoyer_nom(row["EMBRANCHEMENT"], fallback="Inconnu")
    classe = nettoyer_nom(row["CLASSE"], fallback="Classis")
    ordre = nettoyer_nom(row["ORDRE"], fallback="Order")
    famille = nettoyer_nom(row["FAMILLE"], fallback="Familia")
    genre = nettoyer_nom(row["GENRE"], fallback="Genus")
    espece = nettoyer_nom(row["ESPECE"], fallback="sp")

    nom_dossier = f"{embranchement}_{classe}_{ordre}_{famille}_{genre}_{espece}"
    dossier_cible = dossier_sortie / nom_dossier
    dossier_cible.mkdir(parents=True, exist_ok=True)

    image_source = dossier_images / filename
    image_destination = dossier_cible / filename

    if image_source.exists():
        shutil.copy2(image_source, image_destination)
        df.at[index, "Classe IA"] = nom_dossier
        print(f"Copié : {filename} -> {dossier_cible}")
    else:
        df.at[index, "Classe IA"] = "Image introuvable"
        print(f"Image introuvable : {image_source}")

# === SAUVEGARDE EXCEL ===
df.to_excel(fichier_excel_sortie, index=False)

print(f"Terminé. Fichier mis à jour enregistré ici : {fichier_excel_sortie}")