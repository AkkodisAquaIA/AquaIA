"""
  !!!!!!!!
Dans le programme, les parties de code en commentaire, ne sont pas à suprimer.
Elles sont mises en veille pour ne tester que'une partie du code 
  
   !!!!!!!!
"""

from tools import system as syst
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from dataclasses import dataclass


from tools import utility as util


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



IMAGE_SOURCE = Path(
    "C:/Users/pierre.fancelli/Documents/_Dev/Aqua-IA/"
    "Data/Augmentation/Data_Org/photo_01.jpg"
)


DOSSIER_SORTIE = Path(
    "C:/Users/pierre.fancelli/Documents/_Dev/Aqua-IA/"
    "Data/Augmentation/Data_Aug"
)


# ------------------------------------------------------------
# Nombre de modifications par type
# ------------------------------------------------------------

# Transformation unique
NOMBRE_FLIP_H = 1
NOMBRE_FLIP_V = 1

# Transformations multiples
NOMBRE_ROTATIONS = 1
NOMBRE_LUM_CONTRASTE = 1
NOMBRE_BRUIT = 1
NOMBRE_LUM_CONTRASTE_BRUIT = 1
NB_ZOOM = 1


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



# ============================================================
# Analyse de l'image pour détecter la bestiole pour le zoom
# ============================================================

def diagnostiquer_detection_bestiole(image, seuil=25):
    """
    Detect the approximate specimen area from the background
    and display the detected bounding box.

    The detection is based on the color difference
    between the image pixels and the estimated background color.
    """

    # --------------------------------------------------------
    # Detection of the background color
    # --------------------------------------------------------

    fond = couleur_fond(image)

    fond = np.array(fond, dtype=np.float32)

    # --------------------------------------------------------
    # Calculate the color distance from the background
    # --------------------------------------------------------

    difference = np.sqrt(
        np.sum(
            (
                image.astype(np.float32)
                - fond
            ) ** 2,
            axis=2
        )
    )

    # --------------------------------------------------------
    # Create the foreground mask
    # --------------------------------------------------------

    masque = (difference > seuil).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Slightly close small gaps
    # --------------------------------------------------------

    noyau = np.ones((5, 5), np.uint8)

    masque = cv2.morphologyEx(
        masque,
        cv2.MORPH_CLOSE,
        noyau
    )

    # --------------------------------------------------------
    # Find foreground pixels
    # --------------------------------------------------------

    positions = np.where(masque > 0)

    if len(positions[0]) == 0:

        print()
        print("Aucune bestiole détectée.")
        print(f"Couleur du fond estimée : {tuple(int(x) for x in fond)}")
        print(f"Seuil utilisé : {seuil}")

        return

    y_min = int(positions[0].min())
    y_max = int(positions[0].max())

    x_min = int(positions[1].min())
    x_max = int(positions[1].max())

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    hauteur, largeur = image.shape[:2]

    # --------------------------------------------------------
    # Calculate margins
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
    # Calculate margins in percent
    # --------------------------------------------------------

    marge_gauche_pct = marge_gauche / largeur * 100
    marge_droite_pct = marge_droite / largeur * 100

    marge_haut_pct = marge_haut / hauteur * 100
    marge_bas_pct = marge_bas / hauteur * 100

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("DIAGNOSTIC DE LA DETECTION")
    print("-" * 60)

    print(f"Couleur du fond estimée : {tuple(int(x) for x in fond)}")
    print(f"Seuil de détection      : {seuil}")

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
        f"({marge_gauche_pct:5.1f} %)"
    )

    print(
        f"  Droite : {marge_droite:4d} px "
        f"({marge_droite_pct:5.1f} %)"
    )

    print(
        f"  Haut   : {marge_haut:4d} px "
        f"({marge_haut_pct:5.1f} %)"
    )

    print(
        f"  Bas    : {marge_bas:4d} px "
        f"({marge_bas_pct:5.1f} %)"
    )

    print()
    print(f"Marge minimale : {marge_min} px")

    # --------------------------------------------------------
    # Create image for display
    # --------------------------------------------------------

    image_affichage = image.copy()

    cv2.rectangle(
        image_affichage,
        (x_min, y_min),
        (x_max, y_max),
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_affichage,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(image_rgb)

    plt.title(
        "Détection de la zone occupée par la bestiole"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

def detecter_zone_bestiole(image, seuil=25):
    """
    Detect the approximate bounding box of the specimen
    by comparing pixels with the background color.

    Returns:
        (x_min, y_min, x_max, y_max)
        or None if no specimen is detected.
    """

    fond = couleur_fond(image)

    # Calculate the color distance from the background
    difference = np.sqrt(
        np.sum(
            (image.astype(np.float32) - np.array(fond)) ** 2,
            axis=2
        )
    )

    # Pixels sufficiently different from the background
    masque = difference > seuil

    # Find foreground pixels
    positions = np.where(masque)

    if len(positions[0]) == 0:
        return None

    y_min = positions[0].min()
    y_max = positions[0].max()
    x_min = positions[1].min()
    x_max = positions[1].max()

    return x_min, y_min, x_max, y_max

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

def calculer_occupation(image, seuil=25):

    fond = couleur_fond(image)

    difference = np.sqrt(
        np.sum(
            (
                image.astype(np.float32)
                - np.array(fond, dtype=np.float32)
            ) ** 2,
            axis=2
        )
    )

    masque = (
        difference > seuil
    ).astype(np.uint8)

    surface_bestiole = np.count_nonzero(
        masque
    )

    surface_image = masque.size

    occupation = (
        surface_bestiole
        / surface_image
    )

    occupation_pct = occupation * 100

    return InfoOccupationBestiole(
        occupation = occupation,
        occupation_pct = occupation_pct,
        surface_bestiole = surface_bestiole,
        surface_image = surface_image
    )

def calculer_occupation_bbox(image, bbox):

    if bbox is None:
        return 0

    h, w = image.shape[:2]

    surface_image = h * w

    x_min, y_min, x_max, y_max = bbox

    surface_bbox = (
        (x_max - x_min)
        * (y_max - y_min)
    )

    return surface_bbox / surface_image

# ============================================================
# TRANSFORMATIONS : 6 unitaires & 8 combinées
# ============================================================

# ----- Fonctions génériques ------------------------------------------------------------
def transformation_eclairage(light_data):

    return A.OneOf([

        A.RandomBrightnessContrast(
            brightness_limit=light_data.brightness,
            contrast_limit=light_data.contrast,
            p=0.4
        ),

        A.RandomGamma(
            gamma_limit=(
                light_data.gamma_min,
                light_data.gamma_max),
            p=0.6
        )

    ], p=1.0)

def transformation_rotation(fond):
    """
    Rotation aléatoire de l'image.

    Le fond détecté dans les coins est utilisé pour remplir
    les zones apparues lors de la rotation.
    """

    return A.Compose([
        A.SafeRotate(
            limit=(-20, 20 ),
            interpolation= cv2.INTER_LINEAR,
            border_mode= cv2.BORDER_CONSTANT,  #  cv2.BORDER_REFLECT_101,    border_mode= cv2.BORDER_CONSTANT,
            fill=fond,
            p=1.0
        )
    ])

def transformation_zoom(zoom, fond):
    return A.Compose([
        A.Affine(
            scale=(0.90, zoom),
            translate_percent=(-0.02, 0.02),
            rotate=0,
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=fond,
            fit_output=True,
            p=1.0
        )
    ])

def transformation_bruit(bruit):
    return A.Compose([
        A.GaussNoise(
            std_range= noise_range,
            p=1.0
        )
    ])


# ----- Transformations unitaires (6) ---------------------------------------------------

# 01 : Symétrie horizontale
def augmentation_flip_horizontal():
    """
    Symétrie horizontale.
    """

    return A.Compose([
        A.HorizontalFlip(
            p=1.0
        )
    ])

# 02 : Symétrie verticale
def augmentation_flip_vertical():
    """
    Symétrie verticale.
    """

    return A.Compose([
        A.VerticalFlip(
            p=1.0
        )
    ])

# 03 : Rotation aléatoire
def augmentation_rotation(fond):
    """
    Rotation aléatoire de l'image.

    Le fond détecté dans les coins est utilisé pour remplir
    les zones apparues lors de la rotation.
    """

    return A.Compose([
         transformation_rotation(fond)
    ])

# 04 : Modification de la luminosité et du contraste
def augmentation_luminosite_contraste(light_data):
    """
    Modification aléatoire de la luminosité,
    du contraste ou du gamma.

    Une seule transformation est appliquée
    à chaque appel.
    """

    return A.Compose([

        transformation_eclairage(light_data)

    ])

# 05 : Ajout de bruit gaussien
def augmentation_bruit(noise_range):
    """
    Ajout de bruit gaussien.
    """

    return A.Compose([
        transformation_bruit(noise_range)
    ])

# 06 : Zoom / dézoom aléatoire
def augmentation_zoom(zoom, fond):
    """
    Zoom / dézoom + translation + rotation aléatoire.

    Échelle comprise entre 0.90 et 1.0.
    Translation maximale de 2 %.

    Le canevas de sortie est adapté afin de conserver
    l'intégralité de l'image après transformation.

    Les zones apparues lors de la transformation
    sont remplies avec la couleur du fond détectée.
    """

    return A.Compose([
        transformation_zoom(zoom, fond)
        ])



# ----- Transformations combinées (8) ---------------------------------------------------

# 11 : Rotation + Flip horizontal 
def aug_rot_flip_h(fond):

    return A.Compose([
        transformation_rotation(fond),

        A.HorizontalFlip(
            p=1.0
        )
        ])

# 12 : Rotation + Flip horizontal + Contrast
def aug_rot_flip_h_cont(fond, light_data):

    return A.Compose([
        transformation_rotation(fond),

        A.HorizontalFlip(p=1.0),

        transformation_eclairage(light_data)

        ])

# 13 : Rotation + Flip vertical
def aug_rot_flip_v(fond):

    return A.Compose([
        transformation_rotation(fond),

        A.VerticalFlip(p=1.0)
        
        ])

# 14 : Rotation + Flip vertical + Contrast
def aug_rot_flip_v_cont(fond, light_data):

    return A.Compose([
        transformation_rotation(fond),

        A.VerticalFlip(p=1.0),

        transformation_eclairage(light_data)
        
        ])

# 15 : Zoom + Contrast
def aug_zoom_cont(zoom, fond, light_data):
    """
    Zoom / dézoom aléatoire de l'image.

    Échelle comprise entre 0.90 et 1.10.
    Translation maximale de 5 %.

    Les zones apparues lors de la transformation
    sont remplies avec la couleur du fond détectée.
    """

    return A.Compose([
        transformation_zoom(zoom, fond),

        transformation_eclairage(light_data)

    ])

# 16 : Zoom + rotation
def aug_zoom_rot(zoom, fond):
    """
    Zoom / dézoom aléatoire de l'image.

    Échelle comprise entre 0.90 et 1.10.
    Translation maximale de 5 %.

    Les zones apparues lors de la transformation
    sont remplies avec la couleur du fond détectée.
    """

    return A.Compose([
        transformation_zoom(zoom, fond),

        transformation_rotation(fond)

    ])

# 17 : Zoom + rotation + Contrast
def aug_zoom_rot_cont(zoom, fond, light_data):
    """
    Zoom / dézoom aléatoire de l'image.

    Échelle comprise entre 0.90 et 1.10.
    Translation maximale de 5 %.

    Les zones apparues lors de la transformation
    sont remplies avec la couleur du fond détectée.
    """

    return A.Compose([
        transformation_zoom(zoom, fond),

        transformation_rotation(fond),

        transformation_eclairage(light_data)

    ])

# 18 : luminosité/contraste + bruit
def aug_lum_cont_bruit(light_data):
    """
    Modification de la luminosité/contraste suivie
    d'un ajout de bruit.
    """

    return A.Compose([

        transformation_eclairage(light_data),

        transformation_bruit(noise_range)
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
# PROGRAMME PRINCIPAL
# ============================================================

# Efface l'écran avant de commencer
syst.clear_screen()


# ------------------------------------------------------------
# Lecture de l'image
# ------------------------------------------------------------

image = cv2.imread(str(IMAGE_SOURCE))

if image is None:
    raise FileNotFoundError(
        f"Impossible de lire : {IMAGE_SOURCE}"
    )
    

# ------------------------------------------------------------
# Détection de la couleur du fond
# ------------------------------------------------------------
fond = couleur_fond(image)
print(f"Couleur de fond détectée : {fond}")



# ------------------------------------------------------------
# Détermination du zoom maxi
# ------------------------------------------------------------

diagnostiquer_detection_bestiole(image, seuil=25)
bestiole = detecter_zone_bestiole(image, seuil=25)



info_occ = calculer_occupation(image)
info_image_bboxe = calculer_occupation_bbox(image, bestiole)
print()
print()
print(f"Surface image     : {info_occ.surface_image:,.0f} px")
print(f"Surface bestiole  : {info_occ.surface_bestiole:,.0f} px")
print(f"Occupation réelle : {info_occ.occupation:.3f} soit {info_occ.occupation_pct:.1f} %")
print(f"Occupation Bboxe  : {info_image_bboxe:.3f}")
print()


zoom_max_possible = calculer_zoom_max(image, bestiole)
zoom_max = min(
    zoom_max_possible,
    1.25
)

print()
print('-' * 80)
print(f" - Zoom maxi possible/retenu : {zoom_max_possible:.2f} / {zoom_max:.2f}")
print()

# ------------------------------------------------------------
# Détermination du contraste et de la luminosité
# ------------------------------------------------------------

moyenne, ecart_type = analyser_luminosite(image)

print(f" - Luminosité : {moyenne:.3f}")
print(f" - Écart-type : {ecart_type:.3f}")
print()

brightness, contrast = parametres_lum_contraste(image)
print(f" - Paramètres luminosité/contraste : {brightness:.3f} / {contrast:.3f}")
print()

brig_cont = brightness, contrast

gamma = parametres_gamma(image)
print(f' - plage réglage gamma : {gamma}')
print(  )

noise_range = parametres_bruit(image)
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


# ----- Début de la réalisation des augmentations -------------------------------------------------
# ------------------------------------------------------------
# Création du dossier de sortie
# ------------------------------------------------------------

DOSSIER_SORTIE.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Initialisation du compteur
# ------------------------------------------------------------
numero = 1


# ------ Augmentations simples ---------------------------------------------------------- 
# 1 : FLIP HORIZONTAL
numero = appliquer_augmentation(
    image=image,
    transform=augmentation_flip_horizontal(),
    nom_augmentation="flip_h",
    nombre=NOMBRE_FLIP_H,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 2 : FLIP VERTICAL
numero = appliquer_augmentation(
    image=image,
    transform=augmentation_flip_vertical(),
    nom_augmentation="flip_v",
    nombre=NOMBRE_FLIP_V,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 3 : ROTATIONS
numero = appliquer_augmentation(
    image=image,
    transform=augmentation_rotation(fond),
    nom_augmentation="Rot",
    nombre=NOMBRE_ROTATIONS,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 4 : LUMINOSITÉ / CONTRASTE
numero = appliquer_augmentation(
    image=image,
    transform=augmentation_luminosite_contraste(light_data),
    nom_augmentation="Lum_Contraste",
    nombre=NOMBRE_LUM_CONTRASTE,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 5 : BRUIT
numero = appliquer_augmentation(
    image=image,
    transform=augmentation_bruit(noise_range),
    nom_augmentation="Bruit",
    nombre=NOMBRE_BRUIT,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 6 : Zoom
numero = appliquer_augmentation(
    image=image,
    transform=augmentation_zoom(zoom_max, fond),
    nom_augmentation="Zoom",
    nombre=NB_ZOOM,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# ------ Augmentations combinées -------------------------------------------------------- 
# 11 : Rotation + Symétrie Horizontale 
numero = appliquer_augmentation(
    image=image,
    transform= aug_rot_flip_h(fond),
    nom_augmentation="R_+_FH",
    nombre= NOMBRE_ROTATIONS,
    numero= numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 12 : Rotation + Symétrie Horizontale + Contrast
numero = appliquer_augmentation(
    image=image,
    transform= aug_rot_flip_h_cont(fond, light_data),
    nom_augmentation="R_+_FH_+_Cont",
    nombre= NOMBRE_ROTATIONS,
    numero= numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 13 : Rotation + Symétrie Verticale 
numero = appliquer_augmentation(
    image=image,
    transform= aug_rot_flip_v(fond),
    nom_augmentation="R_+_FV",
    nombre= NOMBRE_ROTATIONS,
    numero= numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 14 : Rotation + Symétrie Verticale + Contrast
numero = appliquer_augmentation(
    image=image,
    transform= aug_rot_flip_v_cont(fond, light_data),
    nom_augmentation="R_+_FH_+_Cont",
    nombre= NOMBRE_ROTATIONS,
    numero= numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 15 : Zoom + contrast
numero = appliquer_augmentation(
    image=image,
    transform=aug_zoom_cont(zoom_max, fond, light_data),
    nom_augmentation="Z_+_ Cont",
    nombre=NB_ZOOM,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 16 : Zoom + rotation
numero = appliquer_augmentation(
    image=image,
    transform=aug_zoom_rot(zoom_max, fond),
    nom_augmentation="Z_+_ R",
    nombre=NB_ZOOM,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 17 : Zoom + rotation + Contrast
numero = appliquer_augmentation(
    image=image,
    transform=aug_zoom_rot_cont(zoom_max, fond, light_data),
    nom_augmentation="Z_+_ R_+_ Cont",
    nombre=NB_ZOOM,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)

# 18 :  LUMINOSITÉ / CONTRASTE + BRUIT
numero = appliquer_augmentation(
    image=image,
    transform=aug_lum_cont_bruit(light_data),
    nom_augmentation="Lum_Cont_Bruit",
    nombre=NOMBRE_LUM_CONTRASTE_BRUIT,
    numero=numero,
    dossier_sortie=DOSSIER_SORTIE,
    nom_source=IMAGE_SOURCE.stem,
    extension=IMAGE_SOURCE.suffix
)


# ============================================================
# RÉSULTAT
# ============================================================

print()
print(
    f"{numero - 1} images créées."
)
