
# -*- coding: utf-8 -*-

from openpyxl import load_workbook
from openpyxl.styles import Font
from pathlib import Path
from traceback import print_exc

# --------------------------------------------------
# Fichiers
# --------------------------------------------------

chemin = "c:/Users/Pierre.FANCELLI/Documents/_Dev/Aqua-IA/__Excel__/Travail_02"

file = "Adour_00"

fichier_entree = Path(chemin) / f"{file}.xlsx"
fichier_sortie = Path(chemin) / f"{file}_res_.xlsx"

if fichier_entree.exists():

    print(f"Fichier à traiter : {fichier_entree.stem}")

else:

    print(f"Le fichier '{fichier_entree.stem}' est introuvable")

    exit()

print()


# --------------------------------------------------
# Lecture Excel
# --------------------------------------------------

try:

    print("Ouverture du classeur...")
    
    wb = load_workbook(
    fichier_entree,
    data_only=True
    )

    print("Classeur ouvert.")

except Exception:

    print("Erreur pendant l'ouverture :")

    print_exc()

    raise


ws = wb.active


# --------------------------------------------------
# Création feuille résultat
# --------------------------------------------------

if "Macro_invertebres" in wb.sheetnames:

    ws_suppr = wb["Macro_invertebres"]

    wb.remove(ws_suppr)


ws_out = wb.create_sheet("Macro_invertebres")


entetes = [

    "Classe",
    "Ordre",
    "Famille",
    "Genre",
    "Doublon",
    "Color",
    "Taxon"

]

ws_out.append(entetes)


# --------------------------------------------------
# Fonction de normalisation des noms
# --------------------------------------------------

def nom_standard(nom, nom_colonne):
    """
    Première lettre en majuscule,
    le reste en minuscule.

    Si le nom est vide, utilise le nom de la colonne.
    """

    if nom is None or str(nom).strip() == "":
        return nom_colonne

    return str(nom).strip().capitalize()


# --------------------------------------------------
# Gestion cellules fusionnées
# --------------------------------------------------

def valeur_cellule(ws, ligne, colonne):

    """
    Retourne la valeur réelle d'une cellule,
    y compris si elle appartient à une fusion.
    """

    cellule = ws.cell(ligne, colonne)

    if cellule.value is not None:

        return cellule.value

    # Recherche dans les cellules fusionnées

    for fusion in ws.merged_cells.ranges:

        if cellule.coordinate in fusion:

            return ws.cell(
                fusion.min_row,
                fusion.min_col
            ).value

    return None


def cellule_coloree(ws, ligne, colonne):

    """
    Retourne 1 si la cellule possède un remplissage,
    sinon 0.
    """

    cellule = ws.cell(ligne, colonne)

    remplissage = cellule.fill

    # Pas de remplissage

    if remplissage.fill_type is None:
        return 0

    # Vérification de la couleur

    frequency = remplissage.fgColor

    if frequency.type == "rgb":

        if frequency.rgb not in (None, "00000000"):
            return 1

    elif frequency.type in ("theme", "indexed"):
        return 1

    return 0


# --------------------------------------------------
# Blocs Genre / Espèce
# --------------------------------------------------

blocs = [
    (3, 4, "C-D"),
    (9, 10, "I-J"),
    (15, 16, "O-P")
]

doublons = {}


# --------------------------------------------------
# Gestion du début et de la fin de l'analyse
# --------------------------------------------------

premier_ligne = 10

ligne = premier_ligne


# --------------------------------------------------
# Extraction
# --------------------------------------------------

while True:

    # Arrêt à la première cellule vide de la colonne A

    if valeur_cellule(ws, ligne, 1) in (None, ""):

        break

    classe = valeur_cellule(ws, ligne, 1)

    ordre = valeur_cellule(ws, ligne, 2)


    for col_famille, col_genre, source in blocs:

        famille = ws.cell(ligne, col_famille).value

        genre = ws.cell(ligne, col_genre).value

        if famille is None or str(famille).strip() == "":

            continue


        # --------------------------------------------------
        # Normalisation des noms
        # --------------------------------------------------

        classe_std = nom_standard(classe, "Classe")
        ordre_std = nom_standard(ordre, "Ordre")
        famille_std = nom_standard(famille, "Famille")
        genre_std = nom_standard(genre, "Genus")


        # --------------------------------------------------
        # Clé utilisée pour détecter les doublons
        # --------------------------------------------------

        cle = (
            famille_std.lower()
            + "|"
            + genre_std.lower()
        )


        # --------------------------------------------------
        # Couleur
        # --------------------------------------------------

        frequency = cellule_coloree(ws, ligne, col_genre)


        # --------------------------------------------------
        # Construction du Taxon
        # --------------------------------------------------

        taxon = "_".join([
            classe_std[:3],
            ordre_std[:3],
            famille_std,
            genre_std
        ])


        # --------------------------------------------------
        # Écriture de la ligne
        # --------------------------------------------------

        sortie = [
            classe_std,
            ordre_std,
            famille_std,
            genre_std,
            "",
            frequency,
            taxon
        ]

        ws_out.append(sortie)


        # --------------------------------------------------
        # Gestion des doublons
        # --------------------------------------------------

        ligne_sortie = ws_out.max_row

        if cle in doublons:

            ws_out.cell(
                ligne_sortie,
                5
            ).value = "DOUBLON"

            ws_out.cell(
                doublons[cle],
                5
            ).value = "DOUBLON"

        else:

            doublons[cle] = ligne_sortie


    ligne += 1


# --------------------------------------------------
# Mise en forme
# --------------------------------------------------

for cellule in ws_out[1]:

    cellule.font = Font(bold=True)


for colonne in ws_out.columns:

    largeur = max(

        len(str(cell.value)) if cell.value else 0

        for cell in colonne

    )

    ws_out.column_dimensions[
        colonne[0].column_letter
    ].width = largeur + 3


# --------------------------------------------------
# Sauvegarde
# --------------------------------------------------

wb.save(fichier_sortie)

print(f"Terminé : {ws_out.max_row - 1} lignes extraites")
print(f"Fichier créé : {fichier_sortie}")
