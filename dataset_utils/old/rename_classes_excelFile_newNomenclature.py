import pandas as pd


def extraire_famille_genre_espece_depuis_fichier1(nom):
    """
    Format attendu :
    Famille_Genre_Espece
    """
    if pd.isna(nom):
        return None, None, None

    morceaux = str(nom).strip().split("_")
    if len(morceaux) < 3:
        return None, None, None

    # On prend les 3 derniers éléments pour être un peu plus robuste
    famille = morceaux[-3]
    genre = morceaux[-2]
    espece = morceaux[-1]

    return famille, genre, espece


def extraire_famille_genre_espece_depuis_fichier2(nom):
    """
    Format attendu :
    Embranchement_Classe_Ordre_Famille_Genre_Espece
    """
    if pd.isna(nom):
        return None, None, None

    morceaux = str(nom).strip().split("_")
    if len(morceaux) < 6:
        return None, None, None

    # On prend les 3 derniers éléments
    famille = morceaux[-3]
    genre = morceaux[-2]
    espece = morceaux[-1]

    return famille, genre, espece


def generer_correspondance_excel(fichier1, fichier2, fichier_sortie):
    # Lecture des fichiers Excel
    df1 = pd.read_excel(fichier1)
    df2 = pd.read_excel(fichier2)

    # Vérification minimale
    if df1.shape[1] < 2:
        raise ValueError("Le premier fichier doit contenir au moins 2 colonnes : classe + nombre d'images.")
    if df2.shape[1] < 1:
        raise ValueError("Le deuxième fichier doit contenir au moins 1 colonne : nom de classe.")

    # Renommer les colonnes utiles
    df1 = df1.copy()
    df2 = df2.copy()

    df1 = df1.rename(columns={df1.columns[0]: "classe_fichier1", df1.columns[1]: "nb_images"})

    df2 = df2.rename(columns={df2.columns[0]: "classe_fichier2"})

    # Extraction Famille / Genre / Espece
    df1[["famille", "genre", "espece"]] = df1["classe_fichier1"].apply(lambda x: pd.Series(extraire_famille_genre_espece_depuis_fichier1(x)))

    df2[["famille", "genre", "espece"]] = df2["classe_fichier2"].apply(lambda x: pd.Series(extraire_famille_genre_espece_depuis_fichier2(x)))

    # Jointure sur famille, genre, espece
    fusion = df1.merge(df2, on=["famille", "genre", "espece"], how="left")

    # Détection des cas ambigus (plusieurs correspondances)
    doublons_df2 = df2.groupby(["famille", "genre", "espece"]).size().reset_index(name="nb_correspondances")

    fusion = fusion.merge(doublons_df2, on=["famille", "genre", "espece"], how="left")

    # Statut de correspondance
    def definir_statut(row):
        if pd.isna(row["classe_fichier2"]):
            return "non_trouve"
        elif row["nb_correspondances"] > 1:
            return "plusieurs_correspondances"
        else:
            return "ok"

    fusion["statut"] = fusion.apply(definir_statut, axis=1)

    # Fichier principal demandé :
    # colonne 1 = classe correspondante du fichier 2
    # colonne 2 = nombre d'images du fichier 1
    resultat = fusion[["classe_fichier2", "nb_images", "classe_fichier1", "statut"]].copy()
    resultat = resultat.rename(columns={"classe_fichier2": "classe_correspondante_fichier2", "nb_images": "nombre_images", "classe_fichier1": "classe_originale_fichier1"})

    # Feuilles complémentaires utiles
    non_trouves = resultat[resultat["statut"] == "non_trouve"].copy()
    ambigus = resultat[resultat["statut"] == "plusieurs_correspondances"].copy()
    ok = resultat[resultat["statut"] == "ok"].copy()

    # Sauvegarde dans un Excel avec plusieurs feuilles
    with pd.ExcelWriter(fichier_sortie, engine="openpyxl") as writer:
        resultat.to_excel(writer, sheet_name="toutes_correspondances", index=False)
        ok.to_excel(writer, sheet_name="correspondances_ok", index=False)
        non_trouves.to_excel(writer, sheet_name="non_trouves", index=False)
        ambigus.to_excel(writer, sheet_name="ambigus", index=False)

    print(f"Fichier généré : {fichier_sortie}")
    print(f"Total lignes fichier 1 : {len(df1)}")
    print(f"Correspondances OK : {len(ok)}")
    print(f"Non trouvées : {len(non_trouves)}")
    print(f"Ambiguës : {len(ambigus)}")


if __name__ == "__main__":
    fichier1 = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/FIN-Benthic1-2/FIN-Benthic2_cleaned2_Nb_images_per_class.xlsx"
    fichier2 = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/FIN-Benthic1-2/AQUA-IA_dataset_Nb_images_per_class.xlsx"
    fichier_sortie = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/FIN-Benthic1-2/FIN-Benthic2_cleaned2_Nb_images_per_class_2.xlsx"

    generer_correspondance_excel(fichier1, fichier2, fichier_sortie)
