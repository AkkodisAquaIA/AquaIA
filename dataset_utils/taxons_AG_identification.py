import pandas as pd
from pathlib import Path


# ============================================================
# PARAMETRES
# ============================================================

TAXONS_FILE = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/AquaIA_BDD_Listes__MAJ_31082026_envoyée_Sarah_310826.xlsx"
CLASSES_FILE = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/cvat/New_labels_cvat.xlsx"

OUTPUT_CLASSES_PRESENTES = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/classes_presentes_dans_taxons.xlsx"
OUTPUT_TAXONS_ABSENTS = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/ADOUR-GARONNE/taxons_absents_des_classes.xlsx"


# Colonnes utilisées dans le tableau taxonomique
TAXONOMIC_COLUMNS = [
    "EMBRANCHEMENT",
    "CLASSE",
    "ORDRE",
    "FAMILLE",
    "GENRE",
]


# Valeurs signifiant que le niveau taxonomique n'est pas renseigné.
# Ces niveaux seront IGNORÉS lors de la comparaison.
MISSING_TAXON_VALUES = {
    "sans ordre taxo",
    "sans ordre spécifié",
    "sans genre spécifié",
    "sans famille spécifié",
    "sans classe taxo",
}


# ============================================================
# FONCTIONS
# ============================================================


def normalize(value):
    """
    Normalise une valeur pour faciliter les comparaisons :
    - conversion en texte
    - suppression des espaces avant/après
    - comparaison insensible à la casse
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value.casefold()


def is_missing_taxon(value):
    """
    Retourne True lorsque le niveau taxonomique
    doit être ignoré lors de la comparaison.
    """

    value = normalize(value)

    if value is None:
        return True

    return value in MISSING_TAXON_VALUES


def parse_class_name(class_name):
    """
    Sépare un nom de classe suivant la nomenclature :

    Embranchement_Classe_Ordre_Famille_Genre_Espèce

    Retourne un dictionnaire.
    """

    class_name = str(class_name).strip()
    parts = class_name.split("_")

    if len(parts) != 6:
        return None

    return {
        "EMBRANCHEMENT": parts[0],
        "CLASSE": parts[1],
        "ORDRE": parts[2],
        "FAMILLE": parts[3],
        "GENRE": parts[4],
        "ESPECE": parts[5],
    }


def taxon_matches_class(taxon_row, parsed_class):
    """
    Vérifie si une classe correspond à une ligne du tableau taxonomique.

    Les niveaux contenant par exemple :
        Sans Ordre Taxo
        Sans Ordre Spécifié
        Sans Genre Spécifié
        Sans Famille Spécifié
        Sans Classe Taxo

    sont ignorés.
    """

    for column in TAXONOMIC_COLUMNS:
        taxon_value = taxon_row[column]

        # Niveau non renseigné -> on ne le compare pas
        if is_missing_taxon(taxon_value):
            continue

        class_value = parsed_class[column]

        if normalize(taxon_value) != normalize(class_value):
            return False

    return True


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================


def compare_taxons_and_classes(
    taxons_file,
    classes_file,
    output_classes_presentes,
    output_taxons_absents,
):
    taxons_file = Path(taxons_file)
    classes_file = Path(classes_file)

    # --------------------------------------------------------
    # Vérification des fichiers
    # --------------------------------------------------------

    if not taxons_file.exists():
        raise FileNotFoundError(f"Le fichier taxonomique n'existe pas : {taxons_file}")

    if not classes_file.exists():
        raise FileNotFoundError(f"Le fichier des classes n'existe pas : {classes_file}")

    # --------------------------------------------------------
    # Lecture des fichiers Excel
    # --------------------------------------------------------

    taxons_df = pd.read_excel(taxons_file)
    classes_df = pd.read_excel(classes_file)

    print("Colonnes lues dans le tableau 1 :", taxons_df.columns.tolist())

    # --------------------------------------------------------
    # Vérification des colonnes taxonomiques
    # --------------------------------------------------------

    missing_columns = [column for column in TAXONOMIC_COLUMNS if column not in taxons_df.columns]

    if missing_columns:
        raise ValueError("Colonnes absentes du tableau taxonomique : " + ", ".join(missing_columns))

    # --------------------------------------------------------
    # Première colonne du tableau des classes
    # --------------------------------------------------------

    class_column = classes_df.columns[0]

    print(f"Colonne contenant les classes : {class_column}")

    # --------------------------------------------------------
    # Lecture et vérification des classes
    # --------------------------------------------------------

    parsed_classes = []

    invalid_classes = []

    for value in classes_df[class_column]:
        if pd.isna(value):
            continue

        class_name = str(value).strip()

        parsed = parse_class_name(class_name)

        if parsed is None:
            invalid_classes.append(class_name)
            continue

        parsed_classes.append(
            {
                "class_name": class_name,
                "taxonomy": parsed,
            }
        )

    # --------------------------------------------------------
    # Avertissement nomenclature incorrecte
    # --------------------------------------------------------

    if invalid_classes:
        print("\nATTENTION : certaines classes ne respectent pas la nomenclature :")
        print("Embranchement_Classe_Ordre_Famille_Genre_Espèce\n")

        for class_name in invalid_classes:
            print(f"  - {class_name}")

    print(f"\n{len(parsed_classes)} classes valides trouvées.")

    # --------------------------------------------------------
    # Recherche des correspondances
    # On conserve UNE SEULE correspondance par classe :
    # la plus précise taxonomiquement
    # --------------------------------------------------------

    matched_classes = set()
    taxon_has_match = [False] * len(taxons_df)

    match_details = []

    for class_info in parsed_classes:
        class_name = class_info["class_name"]
        parsed_class = class_info["taxonomy"]

        possible_matches = []

        # Recherche de toutes les lignes compatibles
        for position, (_, taxon_row) in enumerate(taxons_df.iterrows()):
            if taxon_matches_class(taxon_row, parsed_class):
                # Score de précision :
                # +1 pour chaque niveau taxonomique réellement renseigné
                specificity_score = 0

                for column in TAXONOMIC_COLUMNS:
                    if not is_missing_taxon(taxon_row[column]):
                        specificity_score += 1

                possible_matches.append(
                    {
                        "position": position,
                        "taxon_row": taxon_row,
                        "score": specificity_score,
                    }
                )

        # ----------------------------------------------------
        # Si plusieurs correspondances :
        # garder uniquement la plus précise
        # ----------------------------------------------------

        if possible_matches:
            best_match = max(possible_matches, key=lambda x: x["score"])

            taxon_row = best_match["taxon_row"]
            position = best_match["position"]

            matched_classes.add(class_name)
            taxon_has_match[position] = True

            match_details.append(
                {
                    "Classe": class_name,
                    "REGNE": taxon_row.get("REGNE", ""),
                    "EMBRANCHEMENT": taxon_row.get("EMBRANCHEMENT", ""),
                    "CLASSE": taxon_row.get("CLASSE", ""),
                    "ORDRE": taxon_row.get("ORDRE", ""),
                    "FAMILLE": taxon_row.get("FAMILLE", ""),
                    "GENRE": taxon_row.get("GENRE", ""),
                    "Commentaires": taxon_row.get("Commentaires", ""),
                    "Fréquent en Bassin AG": taxon_row.get("Fréquent en Bassin AG", ""),
                    "Niveau_precision": best_match["score"],
                }
            )

    # ========================================================
    # TABLEAU 1
    # Classes trouvées dans le tableau taxonomique
    # ========================================================

    classes_presentes_df = pd.DataFrame(match_details)

    if not classes_presentes_df.empty:
        classes_presentes_df = classes_presentes_df.drop_duplicates().sort_values("Classe").reset_index(drop=True)

    # ========================================================
    # TABLEAU 2
    # Taxons du tableau de référence sans classe correspondante
    # ========================================================

    taxons_absents_df = taxons_df.loc[[index for index, has_match in enumerate(taxon_has_match) if not has_match]].copy()

    taxons_absents_df = taxons_absents_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    classes_presentes_df.to_excel(output_classes_presentes, index=False)

    taxons_absents_df.to_excel(output_taxons_absents, index=False)

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n========================================")
    print("RÉSUMÉ")
    print("========================================")

    print(f"Nombre de classes analysées : {len(parsed_classes)}")

    print(f"Classes trouvées dans le tableau taxonomique : {len(matched_classes)}")

    print(f"Classes sans correspondance : {len(parsed_classes) - len(matched_classes)}")

    print(f"Nombre de taxons de référence : {len(taxons_df)}")

    print(f"Taxons sans classe correspondante : {len(taxons_absents_df)}")

    print("\nFichiers générés :")

    print(f"  1. Classes présentes : {output_classes_presentes}")

    print(f"  2. Taxons absents : {output_taxons_absents}")

    # --------------------------------------------------------
    # Affichage des classes non trouvées
    # --------------------------------------------------------

    unmatched_classes = [c["class_name"] for c in parsed_classes if c["class_name"] not in matched_classes]

    if unmatched_classes:
        print("\nClasses ne correspondant à aucun taxon :")

        for class_name in unmatched_classes:
            print(f"  - {class_name}")


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    compare_taxons_and_classes(
        TAXONS_FILE,
        CLASSES_FILE,
        OUTPUT_CLASSES_PRESENTES,
        OUTPUT_TAXONS_ABSENTS,
    )
