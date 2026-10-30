
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

import csv
from datetime import datetime

import cv2
import numpy as np

from config import constants as cst

from augmentation import augmentation as aug



# ============================================================
# NOMS DES AUGMENTATIONS
# ============================================================

FLIP_H = "Flip_H"
FLIP_V = "Flip_V"
ROTATION = "Rotation"
CONTRASTE = "Contraste"
BRUIT = "Bruit"
ZOOM = "Zoom"

ROT_FLIT_H = "Rotation + Flip_H"
ROT_FLIT_H_C = "Rotation + Flip_H + Contraste"
ROT_FLIT_V = "Rotation + Flip_V"
ROT_FLIT_V_C = "Rotation + Flip_V + Contraste"
ZOOM_C = "Zoom + Contraste"
ZOOM_ROTATION = "Zoom + Rotation"
ZOOM_ROTATION_C = "Zoom + Rotation + Contraste"
CONTRASTE_BRUIT = "Contraste + Bruit"


# ============================================================
# PARAMÈTRES PAR DÉFAUT
# ============================================================

SEUIL_DETECTION = 35
NB_MULTI_MAX = 10



# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ConfigurationAugmentation:

    repertoire: Path
    traitements: set[str] = field(default_factory=set)

    nombre_symetries: int = 1
    nombre_rotations: int = 1
    nombre_zooms: int = 1
    nombre_autres: int = 1

    seuil_detection: int = SEUIL_DETECTION

    # --------------------------------------------------------
    # Vérifie si une augmentation a été sélectionnée
    # --------------------------------------------------------
    def traitement_actif(
        self,
        nom_traitement: str
    ) -> bool:

        return nom_traitement in self.traitements

    # --------------------------------------------------------
    # Nombre d'augmentations à générer selon le type
    # --------------------------------------------------------
    def nombre_pour(
        self,
        nom_traitement: str
    ) -> int:

        # Les symétries sont toujours générées une seule fois
        if nom_traitement in {
            FLIP_H,
            FLIP_V
        }:
            return self.nombre_symetries

        # Transformations basées sur la rotation
        if nom_traitement in {
            ROTATION,
            ROT_FLIT_H,
            ROT_FLIT_H_C,
            ROT_FLIT_V,
            ROT_FLIT_V_C,
        }:
            return self.nombre_rotations

        # Transformations basées sur le zoom
        if nom_traitement in {
            ZOOM,
            ZOOM_C,
            ZOOM_ROTATION,
            ZOOM_ROTATION_C,
        }:
            return self.nombre_zooms

        # Contraste, bruit, contraste + bruit...
        return self.nombre_autres

@dataclass
class ResumeTraitement:
    termine: bool = False
    annule: bool = False

    nombre_repertoires: int = 0
    nombre_repertoires_traites: int = 0
    nombre_images_sources: int = 0
    nombre_images_creees: int = 0

    repertoires_vides: list[str] = field(default_factory=list)
    defauts: list[dict] = field(default_factory=list)

    rapport_defauts: Path | None = None


@dataclass
class EvenementProgression:
    type: str
    valeur: int = 0
    maximum: int = 0
    message: str = ""
    donnees: dict = field(default_factory=dict)

#--------------------------------------------------------------

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    marge_x = max(marge_min_px, int(largeur * marge_pct))
    marge_y = max(marge_min_px, int(hauteur * marge_pct))

    x_min = max(0, x_min - marge_x)
    y_min = max(0, y_min - marge_y)

    x_max = min(largeur - 1, x_max + marge_x)
    y_max = min(hauteur - 1, y_max + marge_y)

    return (x_min, y_min, x_max, y_max)

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


def traitement_actif(nom_traitement):
    return nom_traitement in traitements_a_effectuer


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
def transf_image(img,
                repertoire,
                liste_defauts,
                configuration
                ):

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

        bestiole = detecter_zone_bestiole(image, seuil = configuration.seuil_detection)

        bestiole = ajouter_marge_bbox(
            image=image,
            bbox=bestiole,
            marge_pct=0.03,
            marge_min_px=5
            )

        info_occ = calculer_occupation(
            image=image,
            seuil=configuration.seuil_detection
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


        # -----Début des blocs d'augmentations ----------------------------------------------------
        # ------------------------------------------------------------
        # Initialisation du compteur
        # ------------------------------------------------------------
        numero = 1


        # ------ Augmentations simples ---------------------------------------------------------- 
        # 1 : FLIP HORIZONTAL
        if configuration.traitement_actif(FLIP_H) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.augmentation_flip_horizontal(),
                nom_augmentation="H",
                nombre= configuration.nombre_pour(FLIP_H),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 2 : FLIP VERTICAL
        if configuration.traitement_actif(FLIP_V) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.augmentation_flip_vertical(),
                nom_augmentation="V",
                nombre= configuration.nombre_pour(FLIP_V),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 3 : ROTATIONS
        if configuration.traitement_actif(ROTATION) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.augmentation_rotation(fond),
                nom_augmentation="R",
                nombre= configuration.nombre_pour(ROTATION),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 4 : LUMINOSITÉ / CONTRASTE
        if configuration.traitement_actif(CONTRASTE) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.augmentation_luminosite_contraste(light_data),
                nom_augmentation="C",
                nombre= configuration.nombre_pour(CONTRASTE),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 5 : BRUIT
        if configuration.traitement_actif(BRUIT) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.augmentation_bruit(noise_range),
                nom_augmentation="B",
                nombre= configuration.nombre_pour(BRUIT),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 6 : Zoom
        if configuration.traitement_actif(ZOOM) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.augmentation_zoom(zoom_max, fond),
                nom_augmentation="Z",
                nombre= configuration.nombre_pour(ZOOM),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # # ------ Augmentations combinées -------------------------------------------------------- 
        # 7 : Rotation + Symétrie Horizontale 
        if configuration.traitement_actif(ROT_FLIT_H) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.aug_rot_flip_h(fond),
                nom_augmentation="R_H",
                nombre= configuration.nombre_pour(ROTATION),
                numero= numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 8 : Rotation + Symétrie Horizontale + Contrast
        if configuration.traitement_actif(ROT_FLIT_H_C) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.aug_rot_flip_h_cont(fond, light_data),
                nom_augmentation="R_H_C",
                nombre= configuration.nombre_pour(ROTATION),
                numero= numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 9 : Rotation + Symétrie Verticale 
        if configuration.traitement_actif(ROT_FLIT_V) :
            numero = appliquer_augmentation(
                image=image,
                transform= aug.aug_rot_flip_v(fond),
                nom_augmentation="R_V",
                nombre= configuration.nombre_pour(ROTATION),
                numero= numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 10 : Rotation + Symétrie Verticale + Contrast
        if configuration.traitement_actif(ROT_FLIT_V_C) :
            numero = appliquer_augmentation(
                image=image,
                transform= aug.aug_rot_flip_v_cont(fond, light_data),
                nom_augmentation="R_V_C",
                nombre= configuration.nombre_pour(ROTATION),
                numero= numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 11 : Zoom + contrast
        if configuration.traitement_actif(ZOOM_C) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.aug_zoom_cont(zoom_max, fond, light_data),
                nom_augmentation="Z_C",
                nombre= configuration.nombre_pour(ZOOM),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 12 : Zoom + rotation
        if configuration.traitement_actif(ZOOM_ROTATION) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.aug_zoom_rot(zoom_max, fond),
                nom_augmentation="Z_R",
                nombre= configuration.nombre_pour(ZOOM),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 13 : Zoom + rotation + Contrast
        if configuration.traitement_actif(ZOOM_ROTATION_C) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.aug_zoom_rot_cont(zoom_max, fond, light_data),
                nom_augmentation="Z_R_C",
                nombre= configuration.nombre_pour(ZOOM),
                numero=numero,
                dossier_sortie=dossier_sortie,
                nom_source=img.stem,
                extension=img.suffix
            )

        # 14 :  LUMINOSITÉ / CONTRASTE + BRUIT
        if configuration.traitement_actif(CONTRASTE_BRUIT) :
            numero = appliquer_augmentation(
                image=image,
                transform=aug.aug_lum_cont_bruit(light_data, noise_range),
                nom_augmentation="C_B",
                nombre= configuration.nombre_pour(CONTRASTE_BRUIT),
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



def executer_augmentation(
    configuration: ConfigurationAugmentation,
    callback: Callable[[EvenementProgression], None] | None = None,
    demande_arret: Event | None = None
) -> ResumeTraitement:
    """
    Exécute le traitement complet.

    Cette fonction ne contient aucune instruction Tkinter.
    Elle peut donc être utilisée depuis une interface graphique,
    la console ou des tests automatisés.
    """

    resume = ResumeTraitement()

    repertoire_travail = configuration.repertoire

    if demande_arret is None:
        demande_arret = Event()

    def notifier(
        type_evenement: str,
        valeur: int = 0,
        maximum: int = 0,
        message: str = "",
        **donnees
    ):
        if callback is not None:
            callback(
                EvenementProgression(
                    type=type_evenement,
                    valeur=valeur,
                    maximum=maximum,
                    message=message,
                    donnees=donnees
                )
            )

    try:
        sous_repertoires = sorted(
            [
                repertoire
                for repertoire in repertoire_travail.iterdir()
                if repertoire.is_dir()
            ],
            key=lambda chemin: chemin.name.lower()
        )

        resume.nombre_repertoires = len(sous_repertoires)

        notifier(
            type_evenement="initialisation",
            maximum=len(sous_repertoires),
            message=(
                f"{len(sous_repertoires)} sous-répertoire(s) trouvé(s)."
            )
        )

        if not sous_repertoires:

            notifier(
                type_evenement="journal",
                message="Aucun sous-répertoire trouvé."
            )

            resume.termine = True

            notifier(
                type_evenement="termine",
                message="Traitement terminé : aucun sous-répertoire à traiter.",
                resume=resume
            )

            return resume

        for indice_repertoire, repertoire in enumerate(
            sous_repertoires,
            start=1
        ):
            if demande_arret.is_set():
                resume.annule = True

                notifier(
                    type_evenement="annulation",
                    message="Arrêt demandé par l'utilisateur."
                )

                break

            images = sorted(
                [
                    fichier
                    for fichier in repertoire.iterdir()
                    if fichier.is_file()
                    and fichier.suffix.lower() in cst.IMAGE_EXT
                    and "_aug_" not in fichier.stem.lower()
                ],
                key=lambda chemin: chemin.name.lower()
            )

            notifier(
                type_evenement="repertoire",
                valeur=indice_repertoire,
                maximum=len(sous_repertoires),
                message=repertoire.name,
                nombre_images=len(images)
            )

            if not images:
                resume.repertoires_vides.append(repertoire.name)

                notifier(
                    type_evenement="journal",
                    message=(
                        f"Répertoire vide : {repertoire.name}"
                    )
                )

                continue

            resume.nombre_repertoires_traites += 1
            resume.nombre_images_sources += len(images)

            for indice_image, image in enumerate(images, start=1):
                if demande_arret.is_set():
                    resume.annule = True
                    break

                notifier(
                    type_evenement="image",
                    valeur=indice_image,
                    maximum=len(images),
                    message=image.name
                )

                nombre_images_creees = transf_image(
                    img=image,
                    repertoire=repertoire,
                    liste_defauts=resume.defauts,
                    configuration=configuration
                )

                resume.nombre_images_creees += nombre_images_creees

                notifier(
                    type_evenement="statistiques",
                    message=image.name,
                    images_creees=resume.nombre_images_creees,
                    defauts=len(resume.defauts)
                )

                if nombre_images_creees == 0:
                    notifier(
                        type_evenement="journal",
                        message=(
                            f"Aucune image créée pour : {image.name}"
                        )
                    )

            if resume.annule:
                break

        if resume.defauts:
            resume.rapport_defauts = create_file_fault(
                repertoire=repertoire_travail,
                defauts=resume.defauts
            )

        resume.termine = not resume.annule

        notifier(
            type_evenement="termine",
            message=(
                "Traitement annulé."
                if resume.annule
                else "Traitement terminé."
            ),
            resume=resume
        )

        return resume

    except Exception as erreur:
        notifier(
            type_evenement="erreur",
            message=f"{type(erreur).__name__} : {erreur}"
        )

        raise
