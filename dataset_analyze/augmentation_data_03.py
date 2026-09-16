"""
  !!!!!!!!
Dans le programme, les parties de code en commentaire, ne sont pas à suprimer.
Elles sont mises en veille pour ne tester que'une partie du code 
  
   !!!!!!!!
"""

from tools import system as syst
from pathlib import Path

from dataclasses import dataclass
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt

import cv2


from tools import utility as util
from tools import augmentation as aug
from config import constants as cst
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors


display = dc.DisplayColor()


#**************************************************************************************************

# ============================================================
# PARAMÈTRES
# ============================================================

@dataclass
class LightParameters:

    brightness: float
    contrast: float

    gamma_min: int
    gamma_max: int

@dataclass
class InfoOccupationBestiole:

    occupation : float
    occupation_pct : float

    surface_bestiole : float
    surface_image : float

    # x_min: int
    # y_min: int
    # x_max: int
    # y_max: int


REPERTOIRE_TRAVAIL = Path(
    "C:/Users/pierre.fancelli/Documents/_Dev/Aqua-IA/"
    "Data/Data_Perla_Thibaud"
)

MODE_TEST = False


# ------------------------------------------------------------
# Nombre de modifications par type
# ------------------------------------------------------------

# Transformation unique
NB_UNITAIRE =1

# Transformations multiples
NB_MULTI = 1
#


SEUIL_DETECTION =  35  # seuil_detection(image)


#==================================================================================================
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
# Analyse de la luminosité et du contraste
# ============================================================
def analyser_luminosite(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    moyenne = np.mean(gray)

    ecart_type = np.std(gray)

    return moyenne, ecart_type

def parametres_lum_contraste(image):

    _, sigma = analyser_luminosite(image)

    if sigma < 25:
        return 0.10, 0.10

    elif sigma < 50:
        return 0.15, 0.15

    else:
        return 0.25, 0.25

def parametres_gamma(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sigma = np.std(gray)

    if sigma < 25:
        return (85, 115)

    elif sigma < 50:
        return (90, 110)

    else:
        return (95, 105)

def parametres_bruit(image):

    _, sigma = analyser_luminosite(image)

    if sigma < 25:
        return (0.005, 0.02)

    elif sigma < 50:
        return (0.01, 0.03)

    else:
        return (0.01, 0.04)

def seuil_detection(image):

    _, sigma = analyser_luminosite(image)

    if sigma < 20:
        return 28

    elif sigma < 35:
        return 35

    elif sigma < 50:
        return 40

    else:
        return 45

# ============================================================
# Analyse de l'image pour détecter la bestiole pour le zoom
# ============================================================

def ajouter_marge_bbox(
    image,
    bbox,
    marge_pct=0.03,
    marge_min_px=5
):
    """
    Agrandit une bounding box en ajoutant une marge.

    La marge est calculée à partir des dimensions de l'image,
    avec une valeur minimale en pixels.

    Parameters
    ----------
    image : np.ndarray
        Image OpenCV.

    bbox : tuple | None
        (x_min, y_min, x_max, y_max).

    marge_pct : float
        Marge proportionnelle aux dimensions de l'image.
        Par exemple, 0.03 correspond à 3 %.

    marge_min_px : int
        Marge minimale en pixels.

    Returns
    -------
    tuple | None
        Bounding box agrandie et limitée aux dimensions
        de l'image.
    """

    if bbox is None:
        return None

    hauteur, largeur = image.shape[:2]

    x_min, y_min, x_max, y_max = bbox

    marge_x = max(
        marge_min_px,
        int(largeur * marge_pct)
    )

    marge_y = max(
        marge_min_px,
        int(hauteur * marge_pct)
    )

    x_min = max(
        0,
        x_min - marge_x
    )

    y_min = max(
        0,
        y_min - marge_y
    )

    x_max = min(
        largeur - 1,
        x_max + marge_x
    )

    y_max = min(
        hauteur - 1,
        y_max + marge_y
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


def selectionner_composante_principale(
    masque,
    surface_min_pct=0.001
):
    """
    Conserve la plus grande composante connexe du masque.

    Contrairement à la version précédente, une composante
    touchant un bord n'est pas automatiquement rejetée.
    """

    hauteur, largeur = masque.shape[:2]

    surface_image = hauteur * largeur

    surface_min = int(
        surface_image * surface_min_pct
    )

    nombre_labels, labels, statistiques, _ = (
        cv2.connectedComponentsWithStats(
            masque,
            connectivity=8
        )
    )

    meilleur_label = None
    meilleure_surface = 0

    for label in range(1, nombre_labels):

        surface = statistiques[
            label,
            cv2.CC_STAT_AREA
        ]

        if surface < surface_min:
            continue

        if surface > meilleure_surface:

            meilleur_label = label
            meilleure_surface = surface

    masque_final = np.zeros_like(
        masque,
        dtype=np.uint8
    )

    if meilleur_label is not None:

        masque_final[
            labels == meilleur_label
        ] = 255

    return (
        masque_final,
        meilleur_label,
        meilleure_surface
    )

def creer_masque_par_saturation(
    image,
    seuil_saturation=25
):
    """
    Détecte une bestiole colorée sur un fond gris
    en utilisant la saturation HSV.
    """

    # --------------------------------------------------------
    # Lissage léger
    # --------------------------------------------------------

    image_lissee = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Conversion BGR vers HSV
    # --------------------------------------------------------

    image_hsv = cv2.cvtColor(
        image_lissee,
        cv2.COLOR_BGR2HSV
    )

    saturation = image_hsv[:, :, 1]

    # --------------------------------------------------------
    # Les pixels colorés sont plus saturés que le fond gris
    # --------------------------------------------------------

    masque = (
        saturation > seuil_saturation
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    noyau_ouverture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    masque = cv2.morphologyEx(
        masque,
        cv2.MORPH_OPEN,
        noyau_ouverture
    )

    noyau_fermeture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    masque = cv2.morphologyEx(
        masque,
        cv2.MORPH_CLOSE,
        noyau_fermeture
    )

    # --------------------------------------------------------
    # Conservation de la composante principale
    # --------------------------------------------------------

    masque_bestiole, _, _ = (
        selectionner_composante_principale(
            masque=masque,
            surface_min_pct=0.001
        )
    )

    return masque_bestiole


def creer_masque_bestiole(
    image,
    seuil= SEUIL_DETECTION,
    taille_bord=20,
    surface_min_pct=0.001
):
    """
    Crée un masque nettoyé contenant uniquement la bestiole.

    Étapes :
      1. lissage léger de l'image ;
      2. estimation de la couleur du fond ;
      3. création du masque initial ;
      4. nettoyage morphologique ;
      5. suppression des composantes touchant les bords ;
      6. conservation de la plus grande composante valide.

    Returns
    -------
    masque_bestiole : np.ndarray
        Masque uint8 contenant 0 pour le fond et 255
        pour la bestiole.

    fond : tuple
        Couleur BGR estimée du fond.

    difference : np.ndarray
        Carte de distance à la couleur du fond.
    """

    hauteur, largeur = image.shape[:2]

    # --------------------------------------------------------
    # Réduction du bruit et des artefacts JPEG
    # --------------------------------------------------------

    image_lissee = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Estimation du fond
    # --------------------------------------------------------

    fond = couleur_fond(
        image_lissee
    )

    fond_array = np.array(
        fond,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Distance de chaque pixel à la couleur du fond
    # --------------------------------------------------------

    difference = np.sqrt(
        np.sum(
            (
                image_lissee.astype(np.float32)
                - fond_array
            ) ** 2,
            axis=2
        )
    )

    # --------------------------------------------------------
    # Masque initial
    # --------------------------------------------------------

    masque_initial = (
        difference > seuil
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Suppression des petits pixels isolés
    # --------------------------------------------------------

    noyau_ouverture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    masque_nettoye = cv2.morphologyEx(
        masque_initial,
        cv2.MORPH_OPEN,
        noyau_ouverture
    )

    # --------------------------------------------------------
    # Fermeture de petits trous dans la bestiole
    # --------------------------------------------------------

    noyau_fermeture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    masque_nettoye = cv2.morphologyEx(
        masque_nettoye,
        cv2.MORPH_CLOSE,
        noyau_fermeture
    )

    # --------------------------------------------------------
    # Recherche des composantes connexes
    # --------------------------------------------------------

    nombre_labels, labels, statistiques, _ = (
        cv2.connectedComponentsWithStats(
            masque_nettoye,
            connectivity=8
        )
    )

    surface_image = hauteur * largeur

    surface_min = int(
        surface_image * surface_min_pct
    )

    meilleur_label = None
    meilleure_surface = 0

    # --------------------------------------------------------
    # Sélection de la plus grande composante ne touchant
    # aucun bord de l'image
    # --------------------------------------------------------

    for label in range(1, nombre_labels):

        x = statistiques[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = statistiques[
            label,
            cv2.CC_STAT_TOP
        ]

        largeur_composante = statistiques[
            label,
            cv2.CC_STAT_WIDTH
        ]

        hauteur_composante = statistiques[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        surface = statistiques[
            label,
            cv2.CC_STAT_AREA
        ]

        x_max = x + largeur_composante - 1
        y_max = y + hauteur_composante - 1

        # ----------------------------------------------------
        # Ignore les petites composantes
        # ----------------------------------------------------

        if surface < surface_min:
            continue

        # ----------------------------------------------------
        # Ignore toutes les composantes touchant un bord
        # ----------------------------------------------------

        touche_bord = (
            x <= 0
            or y <= 0
            or x_max >= largeur - 1
            or y_max >= hauteur - 1
        )

        if touche_bord:
            continue

        # ----------------------------------------------------
        # Conservation de la plus grande composante valide
        # ----------------------------------------------------

        if surface > meilleure_surface:

            meilleure_surface = surface
            meilleur_label = label

    # --------------------------------------------------------
    # Création du masque final par distance au fond
    # --------------------------------------------------------

    masque_bestiole = np.zeros(
        (hauteur, largeur),
        dtype=np.uint8
    )

    if meilleur_label is not None:

        masque_bestiole[
            labels == meilleur_label
        ] = 255

        methode_detection = "distance au fond"

    else:

        # ----------------------------------------------------
        # Méthode de secours pour les spécimens colorés
        # sur un fond gris
        # ----------------------------------------------------

        masque_bestiole = creer_masque_par_saturation(
            image=image,
            seuil_saturation=25
        )

        methode_detection = "saturation HSV"
    if MODE_TEST :
        print(
            "Méthode de détection :",
            methode_detection
        )

    return (
        masque_bestiole,
        fond,
        difference
    )



def diagnostiquer_detection_bestiole(image, seuil=SEUIL_DETECTION):
    """
    Affiche le masque nettoyé et la bounding box
    réellement utilisés par le programme.
    """

    # --------------------------------------------------------
    # Création du masque nettoyé
    # --------------------------------------------------------

    masque_bestiole, fond, difference = (
        creer_masque_bestiole(
            image=image,
            seuil=seuil
        )
    )

    if MODE_TEST :
    # --------------------------------------------------------
    # Affichage du masque final
    # --------------------------------------------------------


        plt.figure(
            figsize=(12, 8)
        )

        plt.imshow(
            masque_bestiole,
            cmap="gray"
        )

        plt.title(
            "Masque nettoyé de la bestiole"
        )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # --------------------------------------------------------
        # Affichage facultatif de la carte des distances
        # Cette partie peut rester en commentaire après les tests.
        # --------------------------------------------------------
        # TODO : à commenter après les tests
        plt.figure(figsize=(12, 8))
        
        plt.imshow(
            difference,
            cmap="hot"
        )
        
        plt.colorbar()
        
        plt.title(
            "Distance à la couleur du fond"
        )
        
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # Recherche de la bounding box
    # --------------------------------------------------------

    positions = np.where(
        masque_bestiole > 0
    )

    if len(positions[0]) == 0:

        print()
        print("Aucune bestiole détectée.")
        print(f"Couleur du fond estimée : {fond}")
        print(f"Seuil utilisé : {seuil}")

        return

    y_min = int(
        positions[0].min()
    )

    y_max = int(
        positions[0].max()
    )

    x_min = int(
        positions[1].min()
    )

    x_max = int(
        positions[1].max()
    )

    # --------------------------------------------------------
    # Dimensions de l'image
    # --------------------------------------------------------

    hauteur, largeur = image.shape[:2]

    # --------------------------------------------------------
    # Calcul des marges
    # --------------------------------------------------------

    marge_gauche = x_min
    marge_droite = largeur - 1 - x_max

    marge_haut = y_min
    marge_bas = hauteur - 1 - y_max

    marge_min = min(
        marge_gauche,
        marge_droite,
        marge_haut,
        marge_bas
    )

    # --------------------------------------------------------
    # Calcul des marges en pourcentage
    # --------------------------------------------------------

    marge_gauche_pct = (
        marge_gauche / largeur * 100
    )

    marge_droite_pct = (
        marge_droite / largeur * 100
    )

    marge_haut_pct = (
        marge_haut / hauteur * 100
    )

    marge_bas_pct = (
        marge_bas / hauteur * 100
    )



    if MODE_TEST : 
        # --------------------------------------------------------
        # Affichage des informations
        # --------------------------------------------------------

        print()
        print("-" * 60)
        print("DIAGNOSTIC DE LA DÉTECTION")
        print("-" * 60)

        print(
            f"Couleur du fond estimée : {fond}"
        )

        print(
            f"Seuil de détection      : {seuil}"
        )

        print()
        print("Zone détectée :")
        print(f"  Gauche : {x_min} px")
        print(f"  Haut   : {y_min} px")
        print(f"  Droite : {x_max} px")
        print(f"  Bas    : {y_max} px")

        print()
        print("Marges :")

        print(
            f"  Gauche : {marge_gauche:4d} px "
            f"({marge_gauche_pct:.2f} %)"
        )

        print(
            f"  Droite : {marge_droite:4d} px "
            f"({marge_droite_pct:.2f} %)"
        )

        print(
            f"  Haut   : {marge_haut:4d} px "
            f"({marge_haut_pct:.2f} %)"
        )

        print(
            f"  Bas    : {marge_bas:4d} px "
            f"({marge_bas_pct:.2f} %)"
        )

        print()
        print(
            f"Marge minimale : {marge_min} px"
        )


def detecter_zone_bestiole(image, seuil=SEUIL_DETECTION):
    """
    Détecte la bounding box de la bestiole à partir
    du masque nettoyé.

    Returns
    -------
    tuple | None
        (x_min, y_min, x_max, y_max), ou None
        si aucune bestiole n'est détectée.
    """

    masque_bestiole, _, _ = creer_masque_bestiole(
        image=image,
        seuil=seuil
    )

    positions = np.where(
        masque_bestiole > 0
    )

    if len(positions[0]) == 0:
        return None

    y_min = int(
        positions[0].min()
    )

    y_max = int(
        positions[0].max()
    )

    x_min = int(
        positions[1].min()
    )

    x_max = int(
        positions[1].max()
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


def calculer_zoom_max(image, bbox):
    """
    Calculate the maximum zoom factor that keeps
    the detected specimen inside the image.

    The zoom is assumed to be centered on the image.
    """

    if bbox is None:
        return 1.0

    x_min, y_min, x_max, y_max = bbox

    hauteur, largeur = image.shape[:2]

    centre_x = largeur / 2
    centre_y = hauteur / 2

    limites = []

    # Horizontal limits
    if centre_x - x_min > 0:
        limites.append(
            centre_x / (centre_x - x_min)
        )

    if x_max - centre_x > 0:
        limites.append(
            (largeur - centre_x) / (x_max - centre_x)
        )

    # Vertical limits
    if centre_y - y_min > 0:
        limites.append(
            centre_y / (centre_y - y_min)
        )

    if y_max - centre_y > 0:
        limites.append(
            (hauteur - centre_y) / (y_max - centre_y)
        )

    if not limites:
        return 1.0

    return min(limites)


def calculer_occupation(image, seuil=SEUIL_DETECTION):
    """
    Calcule l'occupation réelle à partir du masque nettoyé
    de la bestiole.
    """

    masque_bestiole, _, _ = creer_masque_bestiole(
        image=image,
        seuil=seuil
    )

    surface_bestiole = np.count_nonzero(
        masque_bestiole
    )

    surface_image = masque_bestiole.size

    occupation = (
        surface_bestiole
        / surface_image
    )

    occupation_pct = (
        occupation * 100
    )

    return InfoOccupationBestiole(
        occupation=occupation,
        occupation_pct=occupation_pct,
        surface_bestiole=float(surface_bestiole),
        surface_image=float(surface_image)
    )

def calculer_occupation_bbox(image, bbox):

    if bbox is None:
        return 0.0

    hauteur, largeur = image.shape[:2]

    surface_image = hauteur * largeur

    x_min, y_min, x_max, y_max = bbox

    surface_bbox = (
        (x_max - x_min + 1)
        * (y_max - y_min + 1)
    )

    return (
        surface_bbox
        / surface_image
    )


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

        success, buffer = cv2.imencode(chemin_sortie.suffix, image_aug)
        if success:
            buffer.tofile(str(chemin_sortie))


        if MODE_TEST :
            print(
                f"{numero:02d} : "
                f"{nom_augmentation:25s} → "
                f"{chemin_sortie.name}"
            )

        numero += 1

    return numero

################################################################################################################
def transf_image(img, repertoire, defaut, liste_defauts):

    try:

        # ------------------------------------------------------------
        # Lecture de l'image
        # ------------------------------------------------------------

        image = cv2.imdecode(
            np.fromfile(str(img), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            print(f"[ERREUR] Impossible de lire : {img.name}")

            liste_defauts.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "repertoire": str(repertoire),
                "image": img.name,
                "erreur": "Impossible de lire l'image"
            })

            defaut += 1
            return 0, defaut
        


        dossier_sortie = img.parent
            

        # ------------------------------------------------------------
        # Détection de la couleur du fond
        # ------------------------------------------------------------
        fond = couleur_fond(image)
        if MODE_TEST :
            print(f"Couleur de fond détectée : {fond}")


        # ------------------------------------------------------------
        # Détermination du zoom maxi
        # ------------------------------------------------------------

        diagnostiquer_detection_bestiole(image, seuil = SEUIL_DETECTION)
        bestiole = detecter_zone_bestiole(image, seuil = SEUIL_DETECTION)

        bestiole = ajouter_marge_bbox(
            image=image,
            bbox=bestiole,
            marge_pct=0.03,
            marge_min_px=5
            )

        info_occ = calculer_occupation(image)
        info_image_bboxe = calculer_occupation_bbox(image, bestiole)

        if MODE_TEST :
            print()
            print()
            print(f"Surface image     : {info_occ.surface_image:,.0f} px")
            print(f"Surface bestiole  : {info_occ.surface_bestiole:,.0f} px")
            print(f"Occupation réelle : {info_occ.occupation:.3f} soit {info_occ.occupation_pct:.1f} %")
            print(f"Occupation Bboxe  : {info_image_bboxe:.3f}")
            print()

        occupation_pct = info_occ.occupation_pct

        if bestiole is None:

            zoom_max_possible = 1.0
            zoom_limite = 1.0

        else:

            zoom_max_possible = calculer_zoom_max(
                image,
                bestiole
            )

            if occupation_pct < 3:
                zoom_limite = 1.0

            elif occupation_pct < 10:
                zoom_limite = 1.35

            elif occupation_pct < 20:
                zoom_limite = 1.25

            elif occupation_pct < 30:
                zoom_limite = 1.15

            else:
                zoom_limite = 1.05

        zoom_max = min(
            zoom_max_possible,
            zoom_limite
        )

        if MODE_TEST :
            print()
            print('-' * 80)
            print(f" - Zoom maxi possible/retenu : {zoom_max_possible:.2f} / {zoom_max:.2f}")
            print()

        # ------------------------------------------------------------
        # Détermination du contraste et de la luminosité
        # ------------------------------------------------------------

        moyenne, ecart_type = analyser_luminosite(image)
        brightness, contrast = parametres_lum_contraste(image)
        brig_cont = brightness, contrast

        gamma = parametres_gamma(image)

        noise_range = parametres_bruit(image)


        if MODE_TEST :
            print(f" - Luminosité : {moyenne:.3f}")
            print(f" - Écart-type : {ecart_type:.3f}")
            print()

            print(f" - Paramètres luminosité/contraste : {brightness:.3f} / {contrast:.3f}")
            print()

            print(f' - plage réglage gamma : {gamma}')
            print(  )


            print(f' - plage réglage bruit : {noise_range}')
            print(  )

            print('-' * 80)
            print(  )


        light_data = LightParameters(
            brightness=brightness,
            contrast=contrast,
            gamma_min=gamma[0],
            gamma_max=gamma[1]
        )


        # ------------------------------------------------------------
        # Initialisation du compteur
        # ------------------------------------------------------------
        numero = 1


        # ------ Augmentations simples ---------------------------------------------------------- 
        # 1 : FLIP HORIZONTAL
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_flip_horizontal(),
            nom_augmentation="flip_h",
            nombre= NB_UNITAIRE,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 2 : FLIP VERTICAL
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_flip_vertical(),
            nom_augmentation="flip_v",
            nombre= NB_UNITAIRE,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 3 : ROTATIONS
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_rotation(fond),
            nom_augmentation="Rot",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 4 : LUMINOSITÉ / CONTRASTE
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_luminosite_contraste(light_data),
            nom_augmentation="Lum_Contraste",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 5 : BRUIT
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_bruit(noise_range),
            nom_augmentation="Bruit",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 6 : Zoom
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_zoom(zoom_max, fond),
            nom_augmentation="Zoom",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # ------ Augmentations combinées -------------------------------------------------------- 
        # 11 : Rotation + Symétrie Horizontale 
        numero = appliquer_augmentation(
            image=image,
            transform=aug.aug_rot_flip_h(fond),
            nom_augmentation="R_+_FH",
            nombre= NB_MULTI,
            numero= numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 12 : Rotation + Symétrie Horizontale + Contrast
        numero = appliquer_augmentation(
            image=image,
            transform=aug.aug_rot_flip_h_cont(fond, light_data),
            nom_augmentation="R_+_FH_+_Cont",
            nombre= NB_MULTI,
            numero= numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 13 : Rotation + Symétrie Verticale 
        numero = appliquer_augmentation(
            image=image,
            transform= aug.aug_rot_flip_v(fond),
            nom_augmentation="R_+_FV",
            nombre= NB_MULTI,
            numero= numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 14 : Rotation + Symétrie Verticale + Contrast
        numero = appliquer_augmentation(
            image=image,
            transform= aug.aug_rot_flip_v_cont(fond, light_data),
            nom_augmentation="R_+_FH_+_Cont",
            nombre= NB_MULTI,
            numero= numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 15 : Zoom + contrast
        numero = appliquer_augmentation(
            image=image,
            transform=aug.aug_zoom_cont(zoom_max, fond, light_data),
            nom_augmentation="Z_+_ Cont",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 16 : Zoom + rotation
        numero = appliquer_augmentation(
            image=image,
            transform=aug.aug_zoom_rot(zoom_max, fond),
            nom_augmentation="Z_+_ R",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 17 : Zoom + rotation + Contrast
        numero = appliquer_augmentation(
            image=image,
            transform=aug.aug_zoom_rot_cont(zoom_max, fond, light_data),
            nom_augmentation="Z_+_ R_+_ Cont",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 18 :  LUMINOSITÉ / CONTRASTE + BRUIT
        numero = appliquer_augmentation(
            image=image,
            transform=aug.aug_lum_cont_bruit(light_data, noise_range),
            nom_augmentation="Lum_Cont_Bruit",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        return numero - 1 , defaut

    except Exception as e:

        print(f"[ERREUR] {img.name} : {e}")

        liste_defauts.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "repertoire": str(repertoire),
            "image": img.name,
            "erreur": str(e)
        })

        defaut += 1

        return 0, defaut


################################################################################################################


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

# Efface l'écran avant de commencer
syst.clear_screen()


# --------------------------------------------------------
# Vérification du répertoire
# --------------------------------------------------------

if not REPERTOIRE_TRAVAIL.exists():

    print()
    display.print(f"le répertoire '{REPERTOIRE_TRAVAIL}' n'existe pas", colors['error'])
    exit()
    

if not REPERTOIRE_TRAVAIL.is_dir():
    print()
    display.print(f"le chemin indiqué '{REPERTOIRE_TRAVAIL}' n'est pas un répertoire", colors['error'])
    # print("\nERREUR : le chemin indiqué n'est pas un répertoire :")
    # print(REPERTOIRE_TRAVAIL)
    exit()
    

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

    print("\nAucun sous-répertoire trouvé.")
    exit()
else:
    print(f"\n  Il y a {nombre_sous_repertoires} sous répertoires à traiter")    



# Initialisation des variables de Travail
nombre_total_images_creees = 0
df = 0
liste_defauts = []


# --------------------------------------------------------
# Parcours des sous-répertoires
# --------------------------------------------------------

for indice_rep, repertoire in enumerate(
    sous_repertoires,
    start=1
):

    print()
    print("-" * 80)
    print(
        f"Répertoire "
        f"{indice_rep}/{nombre_sous_repertoires} : "
        f"{repertoire.name}"
    )
    print("-" * 80)

    # Recherche des images
    images = sorted(
        [
            fichier
            for fichier in repertoire.iterdir()
            if fichier.is_file()
            and fichier.suffix.lower() in cst.IMAGE_EXT
        ],
        key=lambda p: p.name.lower()
    )

    nombre_images = len(images)

    if nombre_images == 0:
        print("  Aucune image trouvée.")
        continue

    print(f"  {nombre_images} image(s) trouvée(s).")
    print()

    # Traitement des images
    for indice_img, image in enumerate(
        images,
        start=1
    ):

        print(
            f"    [{indice_img:4d}/{nombre_images:4d}] "
            f"{image.name}"
        )

        # ------------------------------------------------
        # Appel de votre fonction d'augmentation
        # ------------------------------------------------
 
        nombre_images_creees, df = transf_image(
            img=image,
            repertoire=repertoire,
            defaut=df,
            liste_defauts=liste_defauts
        )

        nombre_total_images_creees += (
                nombre_images_creees
        )


# ============================================================
# RÉSULTAT
# ============================================================

print()
print("-" * 80)
print(
    f"Nombre total d'images créées : "
    f"{nombre_total_images_creees}"
)
print("-" * 80)
print()


display.print(f"nombre de défaut : {df}", colors['error'])

with open(
    REPERTOIRE_TRAVAIL / "rapport_defauts.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f, delimiter=";")

    writer.writerow([
        "Date",
        "Répertoire",
        "Image",
        "Erreur"
    ])

    for defaut in liste_defauts:

        writer.writerow([
            defaut["date"],
            defaut["repertoire"],
            defaut["image"],
            defaut["erreur"]
        ])

print()
print(
f"Rapport des défauts enregistré dans :\n"
f"{REPERTOIRE_TRAVAIL}"

)


print("\nFin du traitement !!!")

