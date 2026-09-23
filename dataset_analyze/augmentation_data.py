"""
Génération automatique d'images augmentées

Ce programme permet de générer automatiquement des images supplémentaires à partir d'un jeu d'images
 de référence afin d'enrichir un jeu de données destiné à l'entraînement de modèles d'intelligence artificielle.

Le traitement s'effectue dans chaque sous-répertoire direct sur l'ensemble des sous-répertoires contenus dans le répertoire de travail
spécifié. Les éventuels sous-répertoires de niveau inférieur ne sont pas parcourus.Les nouvelles images sont générées à l'aide de la 
bibliothèque Albumentations, spécialisée dans l'augmentation de données pour le traitement d'images.

Les fichiers dont le nom contient le marqueur _aug_ sont exclus du traitement afin d’éviter d’appliquer de nouvelles augmentations 
aux images déjà générées.


Transformations unitaires

Les transformations suivantes sont appliquées individuellement :

Symétrie horizontale (Horizontal Flip)
Symétrie verticale (Vertical Flip)
Rotation aléatoire comprise entre -20° et +20°
Modification aléatoire de la luminosité, du contraste ou du gamma
Ajout de bruit gaussien
Zoom ou dézoom aléatoire
Transformations combinées

Des combinaisons de transformations sont également générées afin d'augmenter la diversité du jeu de données :

Rotation + symétrie horizontale
Rotation + symétrie horizontale + modification du contraste
Rotation + symétrie verticale
Rotation + symétrie verticale + modification du contraste
Zoom + modification du contraste
Zoom + rotation
Zoom + rotation + modification du contraste
Modification du contraste + ajout de bruit
Nombre d'images générées

À l'exception des symétries horizontales et verticales, pour lesquelles une seule image est produite,
le nombre d'images générées pour chaque type d'augmentation est paramétrable.

Cette approche permet d'adapter facilement le volume d'images produites en fonction de la taille du jeu 
de données initial et des besoins du projet.


Convention de nommage

Les images générées sont enregistrées dans le même répertoire que l'image source.

Le nom d'origine du fichier est conservé et complété par un suffixe indiquant les transformations appliquées :

Suffixe	 Transformation
_H	     Symétrie horizontale
_V	     Symétrie verticale
_R	     Rotation
_C	     Modification de la luminosité, du contraste ou du gamma
_Z	     Zoom ou dézoom
_B	 Ajout de bruit

Exemples
photo_01_aug_01_H.jpg
photo_01_aug_02_V.jpg
photo_01_aug_03_R.jpg
photo_01_aug_11_Z_C.jpg
photo_01_aug_13_Z_R_C.jpg

photo_01_aug_13_Z_R_C.jpg
│        │   │  └──── Zoom + rotation + contraste/luminosité/gamma
│        │   └─────── Numéro de l'image augmentée
│        └─────────── Marqueur d'une image générée
└──────────────────── Nom de l'image source



"""
from tools import system as syst
from pathlib import Path

from dataclasses import dataclass
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm
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

@dataclass
class DetectionBestiole:
    masque: np.ndarray
    bbox: tuple[int, int, int, int] | None
    fond: tuple[int, int, int]
    difference: np.ndarray
    surface_bestiole: int
    surface_image: int
    occupation: float
    occupation_pct: float


# ------------------------------------------------------------
# Nombre de modifications par type
# ------------------------------------------------------------

NB_MULTI = 1

# seuil_detection(image)
SEUIL_DETECTION =  35  


#==================================================================================================
# Création d'un fichier d'erreurs pour les images
def create_file_fault(repertoire, defauts):

    chemin_rapport = repertoire / "rapport_defauts.csv"

    with chemin_rapport.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as fichier:

        writer = csv.writer(fichier, delimiter=";")

        writer.writerow([
            "Date",
            "Répertoire",
            "Image",
            "Erreur"
        ])

        for defaut in defauts:
            writer.writerow([
                defaut["date"],
                defaut["repertoire"],
                defaut["image"],
                defaut["erreur"]
            ])

    return chemin_rapport



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


    else:

        # ----------------------------------------------------
        # Méthode de secours pour les spécimens colorés
        # sur un fond gris
        # ----------------------------------------------------

        masque_bestiole = creer_masque_par_saturation(
            image=image,
            seuil_saturation=25
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

    
    # --------------------------------------------------------
    # Recherche de la bounding box
    # --------------------------------------------------------

    positions = np.where(masque_bestiole > 0)

    if len(positions[0]) == 0:

        print()
        print("Aucune bestiole détectée.")
        print(f"Couleur du fond estimée : {fond}")
        print(f"Seuil utilisé : {seuil}")

        return

    y_min = int(positions[0].min())

    y_max = int(positions[0].max())

    x_min = int(positions[1].min())

    x_max = int(positions[1].max())

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

    marge_gauche_pct = (marge_gauche / largeur * 100)
    marge_droite_pct = (marge_droite / largeur * 100)
    marge_haut_pct = (marge_haut / hauteur * 100)
    marge_bas_pct = (marge_bas / hauteur * 100)

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

    y_min = int(positions[0].min())
    y_max = int(positions[0].max())
    x_min = int(positions[1].min())
    x_max = int(positions[1].max())

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
    Applique une transformation plusieurs fois et sauvegarde
    les images produites.

    Retourne le prochain numéro disponible.

    Raises
    ------
    RuntimeError
        Si Albumentations ne retourne pas d'image ou si
        l'encodage OpenCV échoue.

    OSError
        Si le fichier produit est absent ou vide.
    """

    for _ in range(nombre):

        resultat = transform(image=image)

        if "image" not in resultat:
            raise RuntimeError(
                f"La transformation {nom_augmentation} "
                "n'a retourné aucune image."
            )

        image_aug = resultat["image"]

        if image_aug is None:
            raise RuntimeError(
                f"La transformation {nom_augmentation} "
                "a retourné une image vide."
            )

        nom = (
            f"{nom_source}"
            f"_aug_{numero:02d}"
            f"_{nom_augmentation}"
            f"{extension.lower()}"
        )

        chemin_sortie = dossier_sortie / nom

        success, buffer = cv2.imencode(
            chemin_sortie.suffix.lower(),
            image_aug
        )

        if not success:
            raise RuntimeError(
                f"Impossible d'encoder l'image : {chemin_sortie}"
            )

        buffer.tofile(str(chemin_sortie))

        if (
            not chemin_sortie.exists()
            or chemin_sortie.stat().st_size == 0
        ):
            raise OSError(
                f"Fichier absent ou vide : {chemin_sortie}"
            )

        numero += 1

    return numero

################################################################################################################
def transf_image(img, repertoire,  liste_defauts):

    try:

        # ------------------------------------------------------------
        # Lecture de l'image
        # ------------------------------------------------------------
        image = cv2.imdecode(
            np.fromfile(str(img), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        # Ce n'est pas un format d'image valide
        if image is None:
            liste_defauts.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "repertoire": str(repertoire),
                "image": img.name,
                "erreur": "Lecture Impossible"
            })

            
            return 0
        
        dossier_sortie = img.parent
            
        # ------------------------------------------------------------
        # Détection de la couleur du fond
        # ------------------------------------------------------------
        fond = couleur_fond(image)
        

        # ------------------------------------------------------------
        # Détermination du zoom maxi
        # ------------------------------------------------------------

        # diagnostiquer_detection_bestiole(image, seuil = SEUIL_DETECTION)
        bestiole = detecter_zone_bestiole(image, seuil = SEUIL_DETECTION)

        bestiole = ajouter_marge_bbox(
            image=image,
            bbox=bestiole,
            marge_pct=0.03,
            marge_min_px=5
            )

        info_occ = calculer_occupation(
            image=image,
            seuil=SEUIL_DETECTION
            )


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


        # ------------------------------------------------------------
        # Détermination du contraste et de la luminosité
        # ------------------------------------------------------------

        brightness, contrast = parametres_lum_contraste(image)
     
        gamma = parametres_gamma(image)

        noise_range = parametres_bruit(image)

        light_data = LightParameters(
            brightness=brightness,
            contrast=contrast,
            gamma_min=gamma[0],
            gamma_max=gamma[1]
        )


        # Début des blocs d'augmentations
        # ------------------------------------------------------------
        # Initialisation du compteur
        # ------------------------------------------------------------
        numero = 1


        # ------ Augmentations simples ---------------------------------------------------------- 
        # 1 : FLIP HORIZONTAL
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_flip_horizontal(),
            nom_augmentation="H",
            nombre= 1,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 2 : FLIP VERTICAL
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_flip_vertical(),
            nom_augmentation="V",
            nombre= 1,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        # 3 : ROTATIONS
        numero = appliquer_augmentation(
            image=image,
            transform=aug.augmentation_rotation(fond),
            nom_augmentation="R",
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
            nom_augmentation="C",
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
            nom_augmentation="B",
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
            nom_augmentation="Z",
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
            nom_augmentation="R_H",
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
            nom_augmentation="R_H_C",
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
            nom_augmentation="R_V",
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
            nom_augmentation="R_V_C",
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
            nom_augmentation="Z_C",
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
            nom_augmentation="Z_R",
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
            nom_augmentation="Z_R_C",
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
            nom_augmentation="C_B",
            nombre=NB_MULTI,
            numero=numero,
            dossier_sortie=dossier_sortie,
            nom_source=img.stem,
            extension=img.suffix
        )

        return numero - 1 

    except Exception as e:

        liste_defauts.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "repertoire": str(repertoire),
            "image": img.name,
            "erreur": str(e)
        })

        return 0

################################################################################################################


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

# Efface l'écran avant de commencer
syst.clear_screen()
print()
display.titre(
        "Lancement de l'Augmentation de Données",
        colors['aqua']
    )
print()

# --------------------------------------------------------
# Récupération du Répertoire à utiliser
# --------------------------------------------------------
repertoire_de_travail = util.get_directory_color("Répertoire à utiliser")


# --------------------------------------------------------
# Recherche des sous-répertoires
# --------------------------------------------------------

sous_repertoires = sorted(
    [
        repertoire
        for repertoire in repertoire_de_travail.iterdir()
        if repertoire.is_dir()
    ],
    key=lambda p: p.name.lower()
)

nombre_sous_repertoires = len(sous_repertoires)

if nombre_sous_repertoires == 0:

    print("\nAucun sous-répertoire trouvé.")
    exit()
else:
    nb_spret =  util.format_nombre(nombre_sous_repertoires)
    pls = "s" if nombre_sous_repertoires != 1 else ""
    print(f"\n  Il y a {nb_spret} sous répertoire{pls} à traiter\n\n")    



# Initialisation des variables de Travail
nombre_total_images_creees = 0
liste_defauts = []
liste_rep_vide = []
total_image = 0

# --------------------------------------------------------
# Parcours des sous-répertoires
# --------------------------------------------------------

# Boucle principale sur Répétoire
for indice_rep, repertoire in enumerate(
    tqdm(
        sous_repertoires,
        desc="Répertoires",
        unit="rep",
        ncols=120,
        position=0
    ),
    start=1
    ):


    # Recherche des images
    images = sorted(
        [
            fichier
            for fichier in repertoire.iterdir()
            if fichier.is_file()
            and fichier.suffix.lower() in cst.IMAGE_EXT
            and "_aug_" not in fichier.stem.lower()
        ],
        key=lambda p: p.name.lower()
    )

    nombre_images = len(images)
    total_image += nombre_images

    if nombre_images == 0:
        liste_rep_vide.append(repertoire.name)
        continue


    # Boucle secondaire pour les Images
    for indice_img, image in enumerate(
        tqdm(
            images,
            desc="Images",
            unit="img",
            ncols=120,
            position=1,
            leave=False
            ),
        start=1
        ):
    
        # ------------------------------------------------
        # Appel la fonction d'augmentation
        # ------------------------------------------------
 
        nombre_images_creees = transf_image(
            img=image,
            repertoire=repertoire,
            liste_defauts=liste_defauts
        )

        nombre_total_images_creees += (
                nombre_images_creees
        )

nombre_defauts = len(liste_defauts)


# ============================================================
# RÉSULTAT
# ============================================================
print()
display.titre(
        "Résumé d’exécution",
        colors['aqua']
    )

# Détermination du status d’exécution
if nombre_defauts != 0 :
    status = 'error'
elif len(liste_rep_vide) != 0:
    status = 'warning'
else: 
    status = 'ok'

# Nombres de répertoire & d'images
nombre_repertoires_traites = nombre_sous_repertoires - len(liste_rep_vide) 
nb_i = util.format_nombre(total_image)
display.print(
    f"- {nombre_repertoires_traites}/{nombre_sous_repertoires} Répertoires traités comprenant "
    f"{nb_i} images ",
    colors[status]
    )  

# Nombre de répertoire vide & liste de ceux-ci
if len(liste_rep_vide) != 0:

    pls = "s" if len(liste_rep_vide) > 1 else ""
    display.print(
        f"- Il y a {len(liste_rep_vide)} répertoire{pls} vide{pls} "
        f"\n  Voici la liste :",
        colors['warning']
        )  
    util.afficher_liste_alignee(liste_rep_vide)
    print()

# Nombre d'images crées
nb_t = util.format_nombre(nombre_total_images_creees)
display.print(
    f"- Nombre total d'images créées : "
    f"{nb_t} ",
    colors['info']
)

# Nombres images en défaut
if nombre_defauts != 0:

    pluriel = "s" if nombre_defauts > 1 else ""

    display.print(
        f"- Nombre de défaut{pluriel} : {nombre_defauts}",
        colors["error"]
    )

    chemin_rapport = create_file_fault(
        repertoire=repertoire_de_travail,
        defauts=liste_defauts
    )

    print(
        "\n- Rapport des défauts enregistré dans :"
        f"\n  {chemin_rapport}"
    )

print("\nFin du traitement !!!")
