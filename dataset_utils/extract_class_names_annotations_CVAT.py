from pathlib import Path
import zipfile
import json

# Dossier contenant les fichiers ZIP
ANNOTATIONS_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Livraison_27aout/Annotations")

# Ensemble pour stocker les classes sans doublons
classes = set()

# Parcours de tous les fichiers ZIP
for zip_path in ANNOTATIONS_DIR.glob("*.zip"):

    print(f"Lecture : {zip_path.name}")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:

            # Recherche de instances_default.json dans le ZIP
            json_files = [
                name for name in z.namelist()
                if name.endswith("instances_default.json")
            ]

            if not json_files:
                print("  ⚠ Aucun instances_default.json trouvé")
                continue

            for json_file in json_files:

                with z.open(json_file) as f:
                    data = json.load(f)

                # Association category_id -> nom de classe
                category_map = {
                    category["id"]: category["name"]
                    for category in data.get("categories", [])
                }

                # IDs des catégories réellement utilisées
                used_category_ids = {
                    annotation["category_id"]
                    for annotation in data.get("annotations", [])
                    if "category_id" in annotation
                }

                # Récupération des noms des classes utilisées
                for category_id in used_category_ids:
                    if category_id in category_map:
                        classes.add(category_map[category_id])

    except zipfile.BadZipFile:
        print(f"  ❌ ZIP invalide : {zip_path.name}")

    except Exception as e:
        print(f"  ❌ Erreur avec {zip_path.name} : {e}")


# ============================================================
# Résultats
# ============================================================

classes = sorted(classes)

print("\n============================")
print(f"Nombre de classes utilisées : {len(classes)}")
print("============================")

for class_name in classes:
    print(class_name)

print("\nListe Python :")
print(classes)