

import cv2
import albumentations as A

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
            std_range= bruit,
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
def aug_lum_cont_bruit(light_data, noise_range):
    """
    Modification de la luminosité/contraste suivie
    d'un ajout de bruit.
    """

    return A.Compose([

        transformation_eclairage(light_data),

        transformation_bruit(noise_range)
    ])


