
from tools import system as syst
import cv2
import albumentations as A


import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from tools import utility as util
from tools import menu_color as menu_c
from config import constants as cst




# ------------------------------------------------------------
# Ensemble des augmentations disponibles 
# ------------------------------------------------------------
# ---- augmentations unitaires -------------------------------
FLIP_H      = "Flip_H"     # 1
FLIP_V      = "Flip_V"     # 2
ROTATION    = "Rotation"   # 3  
CONTRASTE   = "Contraste"  # 4
BRUIT       = "Bruit"      # 5
ZOOM        = "Zoom"       # 6

# ---- augmentations combinées -------------------------------
ROT_FLIT_H       = "Roration + Flip_H"              # 7       
ROT_FLIT_H_C     = "Roration + Flip_H + Contraste"  # 8
ROT_FLIT_V       = "Roration + Flip_V"              # 9
ROT_FLIT_V_C     = "Roration + Flip_V + Contraste"  # 10
ZOOM_C           = "Zoom + Contraste"               # 11
ZOOM_ROTATION    = "Zoom + Rotation"                # 12
ZOOM_ROTATION_C  = "Zoom + Rotation + Contraste"    # 13
CONTRASTE_BRUIT = "Contraste + Bruit"               # 14




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
            border_mode= cv2.BORDER_REFLECT_101,  #  cv2.BORDER_REFLECT_101,  / cv2.BORDER_REFLECT / cv2.BORDER_CONSTANT,  
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

# 07 : Rotation + Flip horizontal 
def aug_rot_flip_h(fond):

    return A.Compose([
        transformation_rotation(fond),

        A.HorizontalFlip(
            p=1.0
        )
        ])

# 08 : Rotation + Flip horizontal + Contrast
def aug_rot_flip_h_cont(fond, light_data):

    return A.Compose([
        transformation_rotation(fond),

        A.HorizontalFlip(p=1.0),

        transformation_eclairage(light_data)

        ])

# 09 : Rotation + Flip vertical
def aug_rot_flip_v(fond):

    return A.Compose([
        transformation_rotation(fond),

        A.VerticalFlip(p=1.0)
        
        ])

# 10 : Rotation + Flip vertical + Contrast
def aug_rot_flip_v_cont(fond, light_data):

    return A.Compose([
        transformation_rotation(fond),

        A.VerticalFlip(p=1.0),

        transformation_eclairage(light_data)
        
        ])

# 11 : Zoom + Contrast
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

# 12 : Zoom + rotation
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

# 13 : Zoom + rotation + Contrast
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

# 14 : luminosité/contraste + bruit
def aug_lum_cont_bruit(light_data, noise_range):
    """
    Modification de la luminosité/contraste suivie
    d'un ajout de bruit.
    """

    return A.Compose([

        transformation_eclairage(light_data),

        transformation_bruit(noise_range)
    ])



# ============================================================
# CHOIX DES TRANSFORMATIONS A REALISER 
# ============================================================

def ajouter_traitements(menu, mapping, traitement_a_realiser):
    menu.display_menu()

    for val in menu.multiple_selection():
        traitement_a_realiser.extend(mapping.get(val, []))


def choix_augmentation():   

    display = dc.DisplayColor()

    traitement_a_realiser = []

    
    display.titre(
        "Choix des augmentations à appliquer",
        colors['aqua']
    )


    # ---- Symètrie Horizontale & Verticale -----------------------------------
    flip_map = {
    1: [],
    2: [FLIP_H],
    3: [FLIP_V],
    4: [FLIP_H, FLIP_V]
    }

    menu_flip = menu_c.Menu('SYMETRIE',           # Nom du menu
                        style= "double",          # Style du menu 
                        theme = menu_c.AQUA_IA    # Theme du menu
                        )
    menu_flip.display_menu()

    for val in menu_flip.multiple_selection():
        traitement_a_realiser.extend(flip_map.get(val, []))


    # # ---- Rotation -----------------------------------------------------------
    rotation_map = {
        1: [],
        2: [ROTATION],
        3: [ROT_FLIT_H],
        4: [ROT_FLIT_H_C],
        5: [ROT_FLIT_V],
        6: [ROT_FLIT_V_C],
        7: [
            ROTATION,
            ROT_FLIT_H,
            ROT_FLIT_H_C,
            ROT_FLIT_V,
            ROT_FLIT_V_C
        ]
    }

    menu_rotation = menu_c.Menu('ROTATION',     # Nom du menu
                        style= "double",        # Style du menu 
                        theme = menu_c.AQUA_IA  # Theme du menu
                    )
    menu_rotation.display_menu()

    for val in menu_rotation.multiple_selection():
        traitement_a_realiser.extend(rotation_map.get(val, []))





    # # ---- Zoom ---------------------------------------------------------------
    zoom_map = {
        1: [],
        2: [ZOOM],
        3: [ZOOM_C],
        4: [ZOOM_ROTATION],
        5: [ZOOM_ROTATION_C],
        6: [
            ZOOM,
            ZOOM_C,
            ZOOM_ROTATION,
            ZOOM_ROTATION_C
        ]
    }
    menu_zoom = menu_c.Menu('ZOOM',           # Nom du menu
                        style= "double",          # Style du menu 
                        theme = menu_c.AQUA_IA    # Theme du menu
                        )

    menu_zoom.display_menu()

    for val in menu_zoom.multiple_selection():
        traitement_a_realiser.extend(zoom_map.get(val, []))




    # # ---- Divers -------------------------------------------------------------
    divers_map = {
        1: [],
        2: [CONTRASTE],
        3: [BRUIT],
        4: [CONTRASTE_BRUIT],
        5: [
            CONTRASTE,
            BRUIT,
            CONTRASTE_BRUIT
        ]
    }
    menu_divers = menu_c.Menu('DIVERS',           # Nom du menu
                        style= "double",          # Style du menu 
                        theme = menu_c.AQUA_IA    # Theme du menu
                        )

    menu_divers.display_menu()

    for val in menu_divers.multiple_selection():
        traitement_a_realiser.extend(divers_map.get(val, []))




    # # ---- Symètrie Horizontale & Verticale -----------------------------------
    # menu_flip = menu_c.Menu('SYMETRIE',               # Nom du menu
    #                     style= "double",          # Style du menu 
    #                     theme = menu_c.AQUA_IA    # Theme du menu
    #                     )

    # menu_flip.display_menu()
    # choice = menu_flip.multiple_selection()

    # for val in choice:

    #     if choice == 1 :
    #         continue

    #     if choice == 4 :
    #         traitement_a_realiser.append(FLIP_H)
    #         traitement_a_realiser.append(FLIP_V)
    #     else:

    #         if choice == 2 :
    #             traitement_a_realiser.append(FLIP_H) 
    #         elif choice == 3 :    
    #             traitement_a_realiser.append(FLIP_V)
        


    # # ---- Rotation -----------------------------------------------------------
    # menu_rotation = menu_c.Menu('ROTATION',     # Nom du menu
    #                     style= "double",        # Style du menu 
    #                     theme = menu_c.AQUA_IA  # Theme du menu
    #                     )

    # menu_rotation.display_menu()
    # choice =  menu_rotation.multiple_selection()

    # for val in choice: 

    #     if val == 7 :
    #         traitement_a_realiser.append(ROTATION)
    #         traitement_a_realiser.append(ROT_FLIT_H)
    #         traitement_a_realiser.append(ROT_FLIT_H_C)
    #         traitement_a_realiser.append(ROT_FLIT_V)
    #         traitement_a_realiser.append(ROT_FLIT_V_C)
    #     else:     
    #         if val == 2 :
    #             traitement_a_realiser.append(ROTATION)
    #         if val == 3 :
    #             traitement_a_realiser.append(ROT_FLIT_H)
    #         if val == 4 :
    #             traitement_a_realiser.append(ROT_FLIT_H_C)
    #         if val == 5 :
    #             traitement_a_realiser.append(ROT_FLIT_V)
    #         if val == 6 :
    #             traitement_a_realiser.append(ROT_FLIT_V_C)


    # # ---- Zoom ---------------------------------------------------------------
    # menu_zoom = menu_c.Menu('ZOOM',           # Nom du menu
    #                     style= "double",          # Style du menu 
    #                     theme = menu_c.AQUA_IA    # Theme du menu
    #                     )

    # menu_zoom.display_menu()
    # choice =  menu_zoom.multiple_selection()

    # for val in choice: 

    #     if val == 6 :
    #         traitement_a_realiser.append(ZOOM)
    #         traitement_a_realiser.append(ZOOM_C)
    #         traitement_a_realiser.append(ZOOM_ROTATION)
    #         traitement_a_realiser.append(ZOOM_ROTATION_C)

    #     else:
    #         if val == 2 :
    #             traitement_a_realiser.append(ZOOM)
                
    #         if val == 3 :
    #             traitement_a_realiser.append(ZOOM_C)
                
    #         if val == 4 :
    #             traitement_a_realiser.append(ZOOM_ROTATION)
                
    #         if val == 5 :
    #             traitement_a_realiser.append(ZOOM_ROTATION_C)



    # # ---- Divers -------------------------------------------------------------
    # menu_divers = menu_c.Menu('DIVERS',           # Nom du menu
    #                     style= "double",          # Style du menu 
    #                     theme = menu_c.AQUA_IA    # Theme du menu
    #                     )

    # menu_divers.display_menu()
    # choice = menu_divers.multiple_selection()

    # for val in choice: 

    #     if val == 5 :
    #         traitement_a_realiser.append(CONTRASTE)
    #         traitement_a_realiser.append(BRUIT)
    #         traitement_a_realiser.append(CONTRASTE_BRUIT)

    #     else: 
    #         if val == 2 :
    #             traitement_a_realiser.append(CONTRASTE)

    #         if val == 3 :
    #             traitement_a_realiser.append(BRUIT)

    #         if val == 4 :
    #             traitement_a_realiser.append(CONTRASTE_BRUIT)








    return traitement_a_realiser

