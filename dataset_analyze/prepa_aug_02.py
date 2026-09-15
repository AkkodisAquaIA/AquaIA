from pathlib import Path

import numpy as np
import cv2
import albumentations as A

from statistics import mean, median
from collections import Counter
import matplotlib.pyplot as plt



from tools import utility as util
from tools import system as syst


# ============================================================
# PARAMÈTRES
# ============================================================

# Répertoire de travail

REPERTOIRE_TRAVAIL = Path(
    r"C:\Users\pierre.fancelli\Documents\_Dev\Aqua-IA\Data\PERLA"
)

REPERTOIRE_SORTIE = Path(
    r"C:\Users\pierre.fancelli\Documents\_Dev\_Back_Up"
)

# Seuil : les sous-répertoires contenant moins que cette
# valeur seront listés

SEUIL = 2

# Nombre de noms affichés par ligne

NOMS_PAR_LIGNE = 2

# Extensions des fichiers considérés comme des images

EXTENSIONS_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

# Nom du fichier de rapport

FICHIER_RAPPORT = REPERTOIRE_SORTIE / "statistiques_images.txt"


# ============================================================
# NOMBRE DE MODIFICATIONS PAR TYPE
# ============================================================

# Transformation unique

NOMBRE_FLIP_H = 1
NOMBRE_FLIP_V = 1

# Transformations multiples

NOMBRE_ROTATIONS = 1
NOMBRE_ZOOM = 1
NOMBRE_LUM_CONTRASTE = 1
NOMBRE_BRUIT = 1
NOMBRE_LUM_CONTRASTE_BRUIT = 1



#==============================================================================

def afficher_distribution_images(nombres_images):
    """
    Affiche la distribution du nombre d'images
    par sous-répertoire.
    """

    distribution = Counter(nombres_images)

    valeurs = sorted(distribution.keys())
    nombres = [distribution[valeur] for valeur in valeurs]

    plt.figure(figsize=(12, 6))

    plt.bar(valeurs, nombres)

    plt.xlabel("Nombre d'images par sous-répertoire")
    plt.ylabel("Nombre de sous-répertoires")
    plt.title("Distribution du nombre d'images par sous-répertoire")

    plt.xticks(valeurs)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

#==============================================================================



# ============================================================
# COULEUR DU FOND
# ============================================================

def couleur_fond(image):
    """
    Estime la couleur du fond à partir des quatre coins
    de l'image.
    """

    hauteur, largeur = image.shape[:2]

    taille = 20

    coins = np.concatenate([
        image[
            0:taille,
            0:taille
        ].reshape(-1, 3),

        image[
            0:taille,
            largeur - taille:largeur
        ].reshape(-1, 3),

        image[
            hauteur - taille:hauteur,
            0:taille
        ].reshape(-1, 3),

        image[
            hauteur - taille:hauteur,
            largeur - taille:largeur
        ].reshape(-1, 3)
    ])

    fond = np.median(coins, axis=0)

    return tuple(int(x) for x in fond)


# ============================================================
# TRANSFORMATIONS
# ============================================================

def augmentation_flip_horizontal():
    """
    Symétrie horizontale.
    """

    return A.Compose([
        A.HorizontalFlip(
            p=1.0
        )
    ])


def augmentation_flip_vertical():
    """
    Symétrie verticale.
    """

    return A.Compose([
        A.VerticalFlip(
            p=1.0
        )
    ])


def augmentation_rotation(fond):
    """
    Rotation aléatoire de l'image.

    Le fond détecté dans les coins est utilisé pour remplir
    les zones apparues lors de la rotation.
    """

    return A.Compose([
        A.SafeRotate(
            limit=180,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=fond,
            p=1.0
        )
    ])


def augmentation_luminosite_contraste():
    """
    Modification aléatoire de la luminosité et du contraste.
    """

    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=1.0
        )
    ])


def augmentation_bruit():
    """
    Ajout de bruit gaussien.
    """

    return A.Compose([
        A.GaussNoise(
            std_range=(0.01, 0.05),
            p=1.0
        )
    ])




def augmentation_luminosite_contraste_bruit():
    """
    Modification de la luminosité/contraste suivie
    d'un ajout de bruit.
    """

    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=1.0
        ),

        A.GaussNoise(
            std_range=(0.01, 0.05),
            p=1.0
        )
    ])



def augmentation_zoom(fond):
    """
    Zoom / dézoom aléatoire de l'image.

    Échelle comprise entre 0.90 et 1.10.
    Translation maximale de 5 %.

    Les zones apparues lors de la transformation
    sont remplies avec la couleur du fond détectée.
    """

    return A.Compose([
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.05, 0.05),
            rotate=0,
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=fond,
            p=1.0
        )
    ])








# ============================================================
# APPLICATION D'UNE TRANSFORMATION
# ============================================================

def appliquer_augmentation(
    image,
    transform,
    nom_augmentation,
    nombre,
    numero,
    dossier_sortie,
    nom_source,
    extension
):
    """
    Applique une transformation plusieurs fois
    et sauvegarde les images produites.

    Retourne le prochain numéro disponible.
    """

    for _ in range(nombre):

        resultat = transform(image=image)

        image_aug = resultat["image"]

        nom = (
            f"{nom_source}"
            f"_aug_{numero:02d}"
            f"_{nom_augmentation}"
            f"{extension}"
        )

        chemin_sortie = dossier_sortie / nom

        cv2.imwrite(
            str(chemin_sortie),
            image_aug
        )

        print(
            f"{numero:02d} : "
            f"{nom_augmentation:25s} → "
            f"{chemin_sortie.name}"
        )

        numero += 1

    return numero


# ============================================================
# AUGMENTATION D'UNE IMAGE
# ============================================================

def augmentation_image(chemin_image, choix):
    """
    Applique uniquement l'augmentation choisie par
    l'opérateur à l'image sélectionnée.
    """

    # --------------------------------------------------------
    # Lecture de l'image
    # --------------------------------------------------------

    image = cv2.imread(str(chemin_image))

    if image is None:

        print(
            f"Impossible de lire l'image : "
            f"{chemin_image}"
        )

        return

    # --------------------------------------------------------
    # Paramètres de sortie
    # --------------------------------------------------------

    dossier_sortie = chemin_image.parent
    nom_source = chemin_image.stem
    extension = chemin_image.suffix

    # --------------------------------------------------------
    # Couleur du fond
    # --------------------------------------------------------

    fond = couleur_fond(image)

    numero = 1

    # --------------------------------------------------------
    # Choix de l'augmentation
    # --------------------------------------------------------

    if choix == 1:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_flip_horizontal(),
            nom_augmentation="flip_h",
            nombre=NOMBRE_FLIP_H,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    elif choix == 2:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_flip_vertical(),
            nom_augmentation="flip_v",
            nombre=NOMBRE_FLIP_V,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    elif choix == 3:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_rotation(fond),
            nom_augmentation="rotation",
            nombre=NOMBRE_ROTATIONS,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    elif choix == 4:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_zoom(fond),
            nom_augmentation="zoom",
            nombre=NOMBRE_ZOOM,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    elif choix == 5:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_luminosite_contraste(),
            nom_augmentation="lum_contraste",
            nombre=NOMBRE_LUM_CONTRASTE,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    elif choix == 6:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_bruit(),
            nom_augmentation="bruit",
            nombre=NOMBRE_BRUIT,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    elif choix == 7:

        numero = appliquer_augmentation(
            image=image,
            transform=augmentation_luminosite_contraste_bruit(),
            nom_augmentation="lum_contraste_bruit",
            nombre=NOMBRE_LUM_CONTRASTE_BRUIT,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=nom_source,
            extension=extension
        )

    print()
    print(
        f"Augmentation terminée pour : "
        f"{chemin_image.name}"
    )


# ============================================================
# MENU D'AUGMENTATION
# ============================================================

def afficher_menu_augmentation(image, nom_repertoire, nom_image):
    """
    Affiche le menu des augmentations directement dans
    la fenêtre OpenCV.

    Retourne :
        1 à 7 : augmentation choisie
        None  : annulation
    """

    image_menu = image.copy()

    hauteur, largeur = image_menu.shape[:2]

    # --------------------------------------------------------
    # Fond semi-transparent du menu
    # --------------------------------------------------------

    overlay = image_menu.copy()

    hauteur_menu = min(430, hauteur - 20)

    cv2.rectangle(
        overlay,
        (20, 20),
        (min(largeur - 20, 760), hauteur_menu),
        (30, 30, 30),
        -1
    )

    image_menu = cv2.addWeighted(
        overlay,
        0.80,
        image_menu,
        0.20,
        0
    )

    # --------------------------------------------------------
    # Titre
    # --------------------------------------------------------

    cv2.putText(
        image_menu,
        "AUGMENTATION DE DONNEES",
        (45, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Nom de l'image
    # --------------------------------------------------------

    cv2.putText(
        image_menu,
        nom_image,
        (45, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Liste des augmentations
    # --------------------------------------------------------

    menu = [
        "1 - Flip horizontal",
        "2 - Flip vertical",
        "3 - Rotation",
        "4 - Zoom / dezoom",
        "5 - Luminosite / contraste",
        "6 - Bruit",
        "7 - Luminosite / contraste + bruit"
    ]

    y = 135

    for texte in menu:

        cv2.putText(
            image_menu,
            texte,
            (45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 38

    # --------------------------------------------------------
    # Annulation
    # --------------------------------------------------------

    cv2.putText(
        image_menu,
        "Echap - Annuler",
        (45, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 200, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(
        "Examen des images",
        image_menu
    )

    # --------------------------------------------------------
    # Attente du choix
    # --------------------------------------------------------

    while True:

        touche = cv2.waitKeyEx(0)

        if touche == 27:

            return None

        if touche in (
            ord("1"),
            ord("2"),
            ord("3"),
            ord("4"),
            ord("5"),
            ord("6"),
            ord("7")
        ):

            return int(chr(touche))


# ============================================================
# CONFIRMATION DE SUPPRESSION
# ============================================================

def confirmer_suppression(image, nom_image):
    """
    Demande confirmation de suppression directement
    dans la fenêtre OpenCV.

    O = Oui
    N = Non
    Échap = Annuler

    Retourne True si l'image doit être supprimée.
    """

    image_confirmation = image.copy()

    hauteur, largeur = image_confirmation.shape[:2]

    # --------------------------------------------------------
    # Fond semi-transparent
    # --------------------------------------------------------

    overlay = image_confirmation.copy()

    largeur_boite = min(760, largeur - 40)
    hauteur_boite = min(220, hauteur - 40)

    x1 = 20
    y1 = 20
    x2 = x1 + largeur_boite
    y2 = y1 + hauteur_boite

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (30, 30, 30),
        -1
    )

    image_confirmation = cv2.addWeighted(
        overlay,
        0.85,
        image_confirmation,
        0.15,
        0
    )

    # --------------------------------------------------------
    # Message
    # --------------------------------------------------------

    cv2.putText(
        image_confirmation,
        "SUPPRIMER CETTE IMAGE ?",
        (45, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image_confirmation,
        nom_image,
        (45, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        image_confirmation,
        "O - Oui",
        (45, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image_confirmation,
        "N - Non",
        (200, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 150, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image_confirmation,
        "Echap - Annuler",
        (370, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(
        "Examen des images",
        image_confirmation
    )

    # --------------------------------------------------------
    # Attente de la réponse
    # --------------------------------------------------------

    while True:

        touche = cv2.waitKeyEx(0)

        if touche in (
            ord("o"),
            ord("O")
        ):

            return True

        elif touche in (
            ord("n"),
            ord("N"),
            27
        ):

            return False


#--------------------------------------------------------------------------------------------------
# ============================================================
# ANALYSE DES SOUS-RÉPERTOIRES
# ============================================================

def compter_images(repertoire):
    """
    Compte les images directement présentes dans
    un répertoire.

    Les éventuels sous-répertoires ne sont pas parcourus.
    """

    nombre = 0

    for fichier in repertoire.iterdir():

        if (
            fichier.is_file()
            and fichier.suffix.lower()
            in EXTENSIONS_IMAGES
        ):

            nombre += 1

    return nombre


def construire_lignes(
    liste,
    noms_par_ligne,
    largeur_nom
):
    """
    Construit les lignes d'affichage.

    Exemple :

    Taxon_01        :    12 | Taxon_02 :     8
    """

    lignes = []

    for i in range(
        0,
        len(liste),
        noms_par_ligne
    ):

        groupe = liste[
            i:i + noms_par_ligne
        ]

        elements = []

        for nom, nombre in groupe:

            element = (
                f"{nom:<{largeur_nom}} : "
                f"{nombre:>5}"
            )

            elements.append(element)

        ligne = " | ".join(elements)

        lignes.append(ligne)

    return lignes


# ============================================================
# EXAMEN DES SOUS-RÉPERTOIRES
# ============================================================

def examiner_sous_seuil(
    sous_seuil,
    repertoire_travail
):
    """
    Parcourt les sous-répertoires contenus dans sous_seuil
    et affiche leurs images avec OpenCV.

    Touches :

        ←       : image précédente
        →       : image suivante
        P       : répertoire précédent
        N       : répertoire suivant
        A       : menu d'augmentation
        S       : supprimer l'image
        Échap   : quitter

    Dans le menu d'augmentation :

        1 : Flip horizontal
        2 : Flip vertical
        3 : Rotation
        4 : Zoom / dézoom
        5 : Luminosité / contraste
        6 : Bruit
        7 : Luminosité / contraste + bruit

    Dans la confirmation de suppression :

        O : Oui
        N : Non
    """

    if not sous_seuil:

        print("\nAucun répertoire à examiner.")

        return

    numero_rep = 0

    # ========================================================
    # PARCOURS DES RÉPERTOIRES
    # ========================================================

    while 0 <= numero_rep < len(sous_seuil):

        nom_repertoire, nombre_images = (
            sous_seuil[numero_rep]
        )

        chemin_repertoire = (
            repertoire_travail / nom_repertoire
        )

        # ----------------------------------------------------
        # Recherche des images
        # ----------------------------------------------------

        images = sorted(
            [
                fichier
                for fichier in chemin_repertoire.iterdir()
                if (
                    fichier.is_file()
                    and fichier.suffix.lower()
                    in EXTENSIONS_IMAGES
                )
            ],
            key=lambda p: p.name.lower()
        )

        # ----------------------------------------------------
        # Répertoire vide
        # ----------------------------------------------------

        if not images:

            print()

            print(
                f"[{numero_rep + 1}/{len(sous_seuil)}] "
                f"{nom_repertoire} : aucune image"
            )

            numero_rep += 1

            continue

        index_image = 0

        print()

        print("=" * 80)

        print(
            f"Répertoire {numero_rep + 1}/"
            f"{len(sous_seuil)} : "
            f"{nom_repertoire}"
        )

        print(
            f"Nombre d'images : {len(images)}"
        )

        print("=" * 80)

        # ====================================================
        # PARCOURS DES IMAGES
        # ====================================================

        while True:

            # ------------------------------------------------
            # Sécurité sur l'index
            # ------------------------------------------------

            if not images:

                numero_rep += 1

                break

            if index_image < 0:

                index_image = len(images) - 1

            if index_image >= len(images):

                index_image = 0

            chemin_image = images[index_image]

            # ------------------------------------------------
            # Lecture de l'image
            # ------------------------------------------------

            image = cv2.imread(
                str(chemin_image)
            )

            if image is None:

                print(
                    f"Impossible de lire : "
                    f"{chemin_image.name}"
                )

                images.pop(index_image)

                if not images:

                    numero_rep += 1
                    break

                continue

            # ------------------------------------------------
            # Préparation de l'affichage
            # ------------------------------------------------

            image_affichage = image.copy()

            texte = (f"{chemin_image.name}")

            cv2.putText(
                image_affichage,
                texte,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Affichage des commandes
            # ------------------------------------------------

            hauteur, largeur = (
                image_affichage.shape[:2]
            )

            texte_commandes = (
                "P/N : repertoires   "
                "<-/-> : images   "
                "A : augmentation   "
                "S : supprimer   "
                "Echap : quitter"
            )

            # On place les commandes uniquement si
            # l'image est suffisamment large.

            if largeur >= 900:

                cv2.putText(
                    image_affichage,
                    texte_commandes,
                    (20, hauteur - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow(
                "Examen des images",
                image_affichage
            )

            touche = cv2.waitKeyEx(0)

            # =================================================
            # ÉCHAP
            # =================================================

            if touche == 27:

                cv2.destroyAllWindows()

                return

            # =================================================
            # FLÈCHE DROITE
            # =================================================

            elif touche == 2555904:

                index_image += 1

                if index_image >= len(images):

                    index_image = 0

            # =================================================
            # FLÈCHE GAUCHE
            # =================================================

            elif touche == 2424832:

                index_image -= 1

                if index_image < 0:

                    index_image = len(images) - 1

            # =================================================
            # TOUCHE A : AUGMENTATION
            # =================================================

            elif touche in (
                ord("a"),
                ord("A")
            ):

                choix = afficher_menu_augmentation(
                    image=image,
                    nom_repertoire=nom_repertoire,
                    nom_image=chemin_image.name
                )

                # --------------------------------------------
                # Annulation du menu
                # --------------------------------------------

                if choix is None:

                    continue

                # --------------------------------------------
                # Application de l'augmentation choisie
                # --------------------------------------------

                augmentation_image(
                    chemin_image,
                    choix
                )

                # --------------------------------------------
                # Retour à l'image originale
                # --------------------------------------------

                cv2.imshow(
                    "Examen des images",
                    image_affichage
                )

                cv2.waitKey(300)

            # =================================================
            # TOUCHE S : SUPPRIMER L'IMAGE
            # =================================================

            elif touche in (
                ord("s"),
                ord("S")
            ):

                confirmation = confirmer_suppression(
                    image=image,
                    nom_image=chemin_image.name
                )

                if confirmation:

                    try:

                        chemin_image.unlink()

                        print()

                        print(
                            f"Image supprimée : "
                            f"{chemin_image.name}"
                        )

                        # ------------------------------------
                        # Retirer l'image de la liste
                        # ------------------------------------

                        images.pop(index_image)

                        # ------------------------------------
                        # Plus aucune image
                        # ------------------------------------

                        if not images:

                            print(
                                f"Le répertoire "
                                f"{nom_repertoire} "
                                f"ne contient plus "
                                f"d'image."
                            )

                            numero_rep += 1

                            break

                        # ------------------------------------
                        # Ajustement de l'index
                        # ------------------------------------

                        if index_image >= len(images):

                            index_image = (
                                len(images) - 1
                            )

                    except OSError as erreur:

                        print()

                        print(
                            f"Erreur lors de la "
                            f"suppression : "
                            f"{erreur}"
                        )

            # =================================================
            # TOUCHE N : RÉPERTOIRE SUIVANT
            # =================================================

            elif touche in (
                ord("n"),
                ord("N")
            ):

                numero_rep += 1

                break

            # =================================================
            # TOUCHE P : RÉPERTOIRE PRÉCÉDENT
            # =================================================

            elif touche in (
                ord("p"),
                ord("P")
            ):

                if numero_rep > 0:

                    numero_rep -= 1

                    break

                else:

                    # Aucun changement de répertoire.
                    # On reste sur l'image courante.

                    continue

        # ----------------------------------------------------
        # Fermeture éventuelle de la fenêtre
        # ----------------------------------------------------

        cv2.destroyAllWindows()

    # ========================================================
    # FIN
    # ========================================================

    cv2.destroyAllWindows()

    print()

    print("=" * 80)

    print(
        "Examen des répertoires terminé."
    )

    print("=" * 80)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 80)

    print(
        "STATISTIQUES DES IMAGES"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Vérification du répertoire
    # --------------------------------------------------------

    if not REPERTOIRE_TRAVAIL.exists():

        print()

        print("ERREUR : le répertoire n'existe pas :")
        print(REPERTOIRE_TRAVAIL)

        return

    if not REPERTOIRE_TRAVAIL.is_dir():

        print()

        print(
            "ERREUR : le chemin indiqué "
            "n'est pas un répertoire :")
        print(REPERTOIRE_TRAVAIL)

        return

    # --------------------------------------------------------
    # Recherche des sous-répertoires
    # --------------------------------------------------------

    sous_repertoires = sorted(
        [
            repertoire
            for repertoire in REPERTOIRE_TRAVAIL.iterdir()
            if repertoire.is_dir()
        ],
        key=lambda p: p.name.lower()
    )

    nombre_sous_repertoires = len(sous_repertoires)

    if nombre_sous_repertoires == 0:

        print()
        print("Aucun sous-répertoire trouvé.")

        return

    # --------------------------------------------------------
    # Comptage des images
    # --------------------------------------------------------

    statistiques = []

    for repertoire in sous_repertoires:

        nombre_images = compter_images(
            repertoire
        )

        statistiques.append(
            (
                repertoire.name,
                nombre_images
            )
        )

    # --------------------------------------------------------
    # Calcul des statistiques
    # --------------------------------------------------------

    nombres_images = [
        nombre
        for nom, nombre in statistiques
    ]


    afficher_distribution_images(nombres_images)


    moyenne = mean(
        nombres_images
    )

    mediane = median(
        nombres_images
    )

    # --------------------------------------------------------
    # Sous-répertoire contenant le plus d'images
    # --------------------------------------------------------

    repertoire_max, nombre_max = max(
        statistiques,
        key=lambda x: x[1]
    )

    # --------------------------------------------------------
    # Sous-répertoires sous le seuil
    # --------------------------------------------------------

    sous_seuil = [
        (nom, nombre)
        for nom, nombre in statistiques
        if nombre <= SEUIL
    ]

    # --------------------------------------------------------
    # Calcul de la largeur nécessaire
    # --------------------------------------------------------

    if sous_seuil:

        largeur_nom = max(
            len(nom)
            for nom, nombre in sous_seuil
        )

    else:

        largeur_nom = 1

    # --------------------------------------------------------
    # Construction de la liste formatée
    # --------------------------------------------------------

    lignes_sous_seuil = construire_lignes(
        sous_seuil,
        NOMS_PAR_LIGNE,
        largeur_nom
    )

    # ========================================================
    # AFFICHAGE À L'ÉCRAN
    # ========================================================

    print()

    print(
        "Répertoire de travail :"
    )

    print(
        f"  {REPERTOIRE_TRAVAIL}"
    )

    print()

    print(
        f"Nombre de sous-répertoires : "
        f"{nombre_sous_repertoires}"
    )

    print()

    print(
        "Sous-répertoire contenant "
        "le plus d'images :"
    )

    print(
        f"  {repertoire_max} : "
        f"{nombre_max} images"
    )

    print()

    print(
        f"Moyenne : {moyenne:.2f} images"
    )

    print(
        f"Médiane : {mediane:.2f} images"
    )

    print()

    print(
        f"Sous-répertoires contenant au maximun "
        f"{SEUIL} images : "
        f"{len(sous_seuil)}"
    )

    # ========================================================
    # CRÉATION DU FICHIER TXT
    # ========================================================

    # Création du répertoire de sortie si nécessaire

    REPERTOIRE_SORTIE.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FICHIER_RAPPORT,
        "w",
        encoding="utf-8"
    ) as fichier:

        fichier.write("=" * 80 + "\n")
        fichier.write("STATISTIQUES DES IMAGES\n")
        fichier.write("=" * 80 + "\n\n")
        fichier.write("Répertoire de travail :\n")
        fichier.write(f"  {REPERTOIRE_TRAVAIL}\n\n")
        fichier.write(
            f"Nombre de sous-répertoires : "
            f"{nombre_sous_repertoires}\n\n")
        fichier.write(
            "Sous-répertoire contenant "
            "le plus d'images :\n")
        fichier.write(f"  {repertoire_max}\n")
        fichier.write(
            f"  Nombre d'images : "
            f"{nombre_max}\n\n")
        fichier.write(
            f"Moyenne : "
            f"{moyenne:.2f} images\n")
        fichier.write(
            f"Médiane : "
            f"{mediane:.2f} images\n\n")

        fichier.write(
            f"Sous-répertoires contenant moins de "
            f"{SEUIL} images : "
            f"{len(sous_seuil)}\n")

        if sous_seuil:

            fichier.write("\n")

            fichier.write("Liste :\n")
            fichier.write("\n")

            for ligne in lignes_sous_seuil:

                fichier.write(ligne + "\n")

        else:

            fichier.write("\n")

            fichier.write(
                "Aucun sous-répertoire "
                "sous le seuil.\n"
            )

    # ========================================================
    # FIN DE LA PARTIE STATISTIQUES
    # ========================================================

    print()

    print("-" * 80)
    print("Rapport créé :")
    print(f"  {FICHIER_RAPPORT}")
    print("-" * 80)
    print()

    # --------------------------------------------------------
    # Pause avant l'ouverture de la visionneuse
    # --------------------------------------------------------

    util.waiting_any_key(
        "Appuyez sur 'Enter' pour continuer ..."
    )

    syst.clear_screen()

    print()

    print("Utilisation des touches :")
    print("  - N             : Répertoire suivant")
    print("  - P             : Répertoire précédent")
    print("  - Flèche Droite : Image suivante")
    print("  - Flèche Gauche : Image précédente")
    print("  - A             : Menu d'augmentation")
    print("  - S             : Supprimer l'image")
    print("  - Échap         : Quitter")
    print()

    # --------------------------------------------------------
    # Examen des images
    # --------------------------------------------------------

    examiner_sous_seuil(
        sous_seuil,
        REPERTOIRE_TRAVAIL
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    main()