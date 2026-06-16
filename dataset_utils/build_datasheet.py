from pathlib import Path
import pandas as pd


# =========================
# PARAMÈTRES
# =========================

Path_dataset_excel_files = "/home/sarah.laroui/Bureau/AQUA-IA/Docs/"
# Mets ici tous tes fichiers Excel
FICHIERS_DATASETS = [
    Path_dataset_excel_files + "FIN-Benthic1-2/FIN-Benthic_Sam3_Nb_images_per_class.xlsx",
    Path_dataset_excel_files + "FIN-Benthic1-2/FIN-Benthic2_cleaned2_Nb_images_per_class.xlsx",
    Path_dataset_excel_files + "PERLA/PERLA_Nb_images_per_class.xlsx",
    # "dataset4.xlsx",
]

# Nom du fichier fusionné
FICHIER_SORTIE_DATASHEET = Path_dataset_excel_files + "Datasheet_AQUA-IA.xlsx"

# Nom du fichier de statistiques
FICHIER_SORTIE_STATS = Path_dataset_excel_files + "Stats_Datasheet_AQUA-IA.xlsx"

# Seuil pour définir une classe rare
SEUIL_CLASSE_RARE = 10

# =========================
# FONCTIONS UTILITAIRES
# =========================

def normaliser_nom_colonne(col):
    return str(col).strip().lower()


def nettoyer_nom_classe(val):
    if pd.isna(val):
        return None
    return str(val).strip()


def charger_dataset(fichier):
    """
    Charge un Excel et renomme la colonne 'nombre d'images'
    avec le nom du fichier.
    """
    fichier = Path(fichier)
    nom_dataset = fichier.stem

    df = pd.read_excel(fichier)
    df.columns = [normaliser_nom_colonne(c) for c in df.columns]

    # Correction de fautes fréquentes
    if "nombre d'imagess" in df.columns:
        df = df.rename(columns={"nombre d'imagess": "nombre d'images"})
    if "nombre_images" in df.columns:
        df = df.rename(columns={"nombre_images": "nombre d'images"})
    if "nombre image" in df.columns:
        df = df.rename(columns={"nombre image": "nombre d'images"})
    if "nb_images" in df.columns:
        df = df.rename(columns={"nb_images": "nombre d'images"})

    if "classe" not in df.columns:
        raise ValueError(f"{fichier} : colonne 'classe' introuvable.")
    if "nombre d'images" not in df.columns:
        raise ValueError(f"{fichier} : colonne 'nombre d\\'images' introuvable.")

    df = df.copy()
    df["classe"] = df["classe"].apply(nettoyer_nom_classe)
    df = df.dropna(subset=["classe"])

    # Si une classe apparaît plusieurs fois dans un même fichier, on somme
    df["nombre d'images"] = pd.to_numeric(df["nombre d'images"], errors="coerce").fillna(0)
    df = df.groupby("classe", as_index=False)["nombre d'images"].sum()

    df = df.rename(columns={"nombre d'images": nom_dataset})
    return df, nom_dataset


def fusionner_datasets(fichiers):
    if not fichiers:
        raise ValueError("La liste des fichiers est vide.")

    dataframes = []
    noms_datasets = []

    for fichier in fichiers:
        df, nom_dataset = charger_dataset(fichier)
        dataframes.append(df)
        noms_datasets.append(nom_dataset)

    df_final = dataframes[0]
    for df in dataframes[1:]:
        df_final = df_final.merge(df, on="classe", how="outer")

    df_final = df_final.fillna(0)

    for nom in noms_datasets:
        df_final[nom] = pd.to_numeric(df_final[nom], errors="coerce").fillna(0).astype(int)

    df_final = df_final.sort_values("classe").reset_index(drop=True)
    return df_final, noms_datasets


def calculer_stats(df_final, noms_datasets, seuil_classe_rare=20):
    df_stats_classes = df_final.copy()

    # Total par classe
    df_stats_classes["total_images"] = df_stats_classes[noms_datasets].sum(axis=1)

    # Nombre de datasets dans lesquels la classe est présente
    df_stats_classes["nb_datasets_presents"] = (df_stats_classes[noms_datasets] > 0).sum(axis=1)

    # Présence partielle / complète
    df_stats_classes["presence"] = df_stats_classes["nb_datasets_presents"].apply(
        lambda x: "tous_les_datasets"
        if x == len(noms_datasets)
        else ("un_seul_dataset" if x == 1 else "plusieurs_datasets")
    )

    # Classe rare selon le total
    df_stats_classes["classe_rare"] = df_stats_classes["total_images"] < seuil_classe_rare

    # Nombre max / min par classe
    df_stats_classes["max_images_dataset"] = df_stats_classes[noms_datasets].max(axis=1)
    df_stats_classes["min_images_dataset"] = df_stats_classes[noms_datasets].min(axis=1)

    # Ratio de déséquilibre simple
    def ratio_desequilibre(row):
        positives = [row[n] for n in noms_datasets if row[n] > 0]
        if len(positives) <= 1:
            return 1.0
        mini = min(positives)
        maxi = max(positives)
        return round(maxi / mini, 2) if mini > 0 else None

    df_stats_classes["ratio_desequilibre"] = df_stats_classes.apply(ratio_desequilibre, axis=1)

    # Nombre de datasets absents
    df_stats_classes["nb_datasets_absents"] = len(noms_datasets) - df_stats_classes["nb_datasets_presents"]

    # Colonne texte listant les datasets absents
    def lister_datasets_absents(row):
        absents = [nom for nom in noms_datasets if row[nom] == 0]
        return ", ".join(absents) if absents else ""

    df_stats_classes["datasets_absents"] = df_stats_classes.apply(lister_datasets_absents, axis=1)

    # ===== Résumés globaux =====
    total_classes = len(df_stats_classes)
    total_images_global = int(df_stats_classes["total_images"].sum())
    classes_rares = int(df_stats_classes["classe_rare"].sum())
    classes_presentes_partout = int((df_stats_classes["nb_datasets_presents"] == len(noms_datasets)).sum())
    classes_presentes_un_seul = int((df_stats_classes["nb_datasets_presents"] == 1).sum())

    # Totaux par dataset
    resume_datasets = []
    for nom in noms_datasets:
        total_images_dataset = int(df_final[nom].sum())
        nb_classes_dataset = int((df_final[nom] > 0).sum())
        moyenne_par_classe_presente = round(
            df_final.loc[df_final[nom] > 0, nom].mean(), 2
        ) if nb_classes_dataset > 0 else 0

        resume_datasets.append({
            "dataset": nom,
            "total_images": total_images_dataset,
            "nb_classes_presentes": nb_classes_dataset,
            "moyenne_images_par_classe_presente": moyenne_par_classe_presente,
        })

    df_resume_datasets = pd.DataFrame(resume_datasets)

    # Résumé général
    df_resume_global = pd.DataFrame([
        {"metrique": "nb_datasets", "valeur": len(noms_datasets)},
        {"metrique": "nb_classes_total", "valeur": total_classes},
        {"metrique": "nb_images_total", "valeur": total_images_global},
        {"metrique": f"nb_classes_rares_total_<_{seuil_classe_rare}", "valeur": classes_rares},
        {"metrique": "nb_classes_presentes_dans_tous_les_datasets", "valeur": classes_presentes_partout},
        {"metrique": "nb_classes_presentes_dans_un_seul_dataset", "valeur": classes_presentes_un_seul},
    ])

    # Sous-ensembles utiles
    df_classes_rares = df_stats_classes[df_stats_classes["classe_rare"]].copy()
    df_classes_rares = df_classes_rares.sort_values(["total_images", "classe"], ascending=[True, True])

    df_classes_absentes_partiellement = df_stats_classes[df_stats_classes["nb_datasets_absents"] > 0].copy()
    df_classes_absentes_partiellement = df_classes_absentes_partiellement.sort_values(
        ["nb_datasets_absents", "classe"], ascending=[False, True]
    )

    df_classes_tres_desequilibrees = df_stats_classes[df_stats_classes["ratio_desequilibre"] >= 5].copy()
    df_classes_tres_desequilibrees = df_classes_tres_desequilibrees.sort_values(
        ["ratio_desequilibre", "classe"], ascending=[False, True]
    )

    return {
        "stats_classes": df_stats_classes,
        "resume_global": df_resume_global,
        "resume_datasets": df_resume_datasets,
        "classes_rares": df_classes_rares,
        "classes_absentes_partiellement": df_classes_absentes_partiellement,
        "classes_tres_desequilibrees": df_classes_tres_desequilibrees,
    }


def sauvegarder_resultats(df_final, stats_dict, fichier_datasheet, fichier_stats):
    df_final.to_excel(fichier_datasheet, index=False)

    with pd.ExcelWriter(fichier_stats, engine="openpyxl") as writer:
        stats_dict["stats_classes"].to_excel(writer, sheet_name="stats_par_classe", index=False)
        stats_dict["resume_global"].to_excel(writer, sheet_name="resume_global", index=False)
        stats_dict["resume_datasets"].to_excel(writer, sheet_name="resume_datasets", index=False)
        stats_dict["classes_rares"].to_excel(writer, sheet_name="classes_rares", index=False)
        stats_dict["classes_absentes_partiellement"].to_excel(writer, sheet_name="classes_absentes", index=False)
        stats_dict["classes_tres_desequilibrees"].to_excel(writer, sheet_name="classes_desequilibrees", index=False)


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():
    df_final, noms_datasets = fusionner_datasets(FICHIERS_DATASETS)

    # Ajouter la colonne total
    df_final["Total nb images"] = df_final[noms_datasets].sum(axis=1)

    stats_dict = calculer_stats(
        df_final=df_final,
        noms_datasets=noms_datasets,
        seuil_classe_rare=SEUIL_CLASSE_RARE
    )

    sauvegarder_resultats(
        df_final=df_final,
        stats_dict=stats_dict,
        fichier_datasheet=FICHIER_SORTIE_DATASHEET,
        fichier_stats=FICHIER_SORTIE_STATS
    )

    print("=== TERMINÉ ===")
    print(f"Datasheet créé : {FICHIER_SORTIE_DATASHEET}")
    print(f"Fichier de stats créé : {FICHIER_SORTIE_STATS}")
    print(f"Datasets fusionnés : {len(noms_datasets)}")
    print(f"Nombre total de classes : {len(df_final)}")
    print(f"Classes rares (< {SEUIL_CLASSE_RARE}) : {len(stats_dict['classes_rares'])}")


if __name__ == "__main__":
    main()