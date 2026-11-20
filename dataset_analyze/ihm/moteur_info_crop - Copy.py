import sys
import numpy as np
from pathlib import Path


import cv2


from config import constants as cst
from tools import utility as util

# =================================================================================================
# PARAMÈTRES
# ==================================================================================================

SEUIL_DETECTION = None
pas_seuil = 5
TAILLE_BORD = 20
SURFACE_MIN_PCT = 0.001

MARGE_BBOX_PCT = 0.03
MARGE_BBOX_MIN_PX = 5


class ErreurLectureImage(Exception):
    """Erreur lors de la lecture ou du décodage d'une image."""
    pass


# ==================================================================================================
# LECTURE D'IMAGE
# ==================================================================================================
def lire_image(chemin_image):

    chemin_image = Path(
        chemin_image
    )

    try:

        donnees = np.fromfile(
            str(chemin_image),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            donnees,
            cv2.IMREAD_COLOR
        )

    except Exception as erreur:

        raise ErreurLectureImage(
            f"Impossible de lire l'image : {chemin_image}"
        ) from erreur

    if image is None:

        raise ErreurLectureImage(
            f"Impossible de décoder l'image : {chemin_image}"
        )

    return image


# ==================================================================================================
# ESTIMATION DE LA COULEUR DU FOND
# ==================================================================================================

def couleur_fond(
    image,
    taille=20
):
    """
    Estime la couleur BGR du fond à partir des quatre coins.
    """

    if image is None:
        raise ValueError(
            "L'image transmise à couleur_fond() est None."
        )

    hauteur, largeur = image.shape[:2]

    taille_maximale = max(
        1,
        min(
            hauteur // 2,
            largeur // 2
        )
    )

    taille = int(
        np.clip(
            taille,
            1,
            taille_maximale
        )
    )

    coin_haut_gauche = image[
        0:taille,
        0:taille
    ].reshape(-1, 3)

    coin_haut_droit = image[
        0:taille,
        largeur - taille:largeur
    ].reshape(-1, 3)

    coin_bas_gauche = image[
        hauteur - taille:hauteur,
        0:taille
    ].reshape(-1, 3)

    coin_bas_droit = image[
        hauteur - taille:hauteur,
        largeur - taille:largeur
    ].reshape(-1, 3)

    pixels_coins = np.concatenate(
        [
            coin_haut_gauche,
            coin_haut_droit,
            coin_bas_gauche,
            coin_bas_droit
        ],
        axis=0
    )

    fond = np.median(
        pixels_coins,
        axis=0
    )

    return tuple(
        int(valeur)
        for valeur in fond
    )


# ==================================================================================================
# TRAITEMENT DES COMPOSANTES CONNEXES
# ==================================================================================================

def selectionner_composante_principale(
    masque,
    surface_min_pct=0.001
):
    """
    Conserve la plus grande composante connexe du masque.
    """

    if masque is None:
        raise ValueError(
            "Le masque transmis est None."
        )

    if masque.ndim != 2:
        raise ValueError(
            "Le masque doit comporter un seul canal."
        )

    masque = masque.astype(
        np.uint8,
        copy=False
    )

    hauteur, largeur = masque.shape[:2]

    surface_image = (
        hauteur * largeur
    )

    surface_min = max(
        1,
        int(
            surface_image
            * surface_min_pct
        )
    )

    (
        nombre_labels,
        labels,
        statistiques,
        _
    ) = cv2.connectedComponentsWithStats(
        masque,
        connectivity=8
    )

    meilleur_label = None
    meilleure_surface = 0

    for label in range(
        1,
        nombre_labels
    ):

        surface = int(
            statistiques[
                label,
                cv2.CC_STAT_AREA
            ]
        )

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


# ==================================================================================================
# CALCUL AUTOMATIQUE DU SEUIL
# ==================================================================================================

def calculer_seuil_depuis_difference(
    afficher_informations,
    difference,
    taille_bord=20,
    percentile_fond=95.0,
    facteur_mad=4.0,
    marge_seuil=2.0,
    seuil_min=5.0,
    seuil_max=None
):
    """
    Calcule automatiquement le seuil de détection à partir
    des distances observées dans une bande périphérique.
    """

    if difference is None:
        raise ValueError(
            "La carte de différence est None."
        )

    if difference.ndim != 2:
        raise ValueError(
            "La carte de différence doit comporter "
            "deux dimensions."
        )

    hauteur, largeur = (
        difference.shape
    )

    if hauteur == 0 or largeur == 0:
        raise ValueError(
            "La carte de différence est vide."
        )

    taille_maximale = max(
        1,
        min(
            hauteur // 2,
            largeur // 2
        )
    )

    taille_bord = int(
        np.clip(
            taille_bord,
            1,
            taille_maximale
        )
    )

    # ----------------------------------------------------------
    # Bande périphérique
    # ----------------------------------------------------------

    masque_bord = np.zeros(
        (hauteur, largeur),
        dtype=bool
    )

    masque_bord[
        :taille_bord,
        :
    ] = True

    masque_bord[
        -taille_bord:,
        :
    ] = True

    masque_bord[
        :,
        :taille_bord
    ] = True

    masque_bord[
        :,
        -taille_bord:
    ] = True

    differences_fond = (
        difference[
            masque_bord
        ]
    )

    differences_fond = differences_fond[
        np.isfinite(
            differences_fond
        )
    ]

    if differences_fond.size == 0:
        raise ValueError(
            "Aucune distance valide n'a été "
            "trouvée sur les bords."
        )

    # ----------------------------------------------------------
    # MAD
    # ----------------------------------------------------------

    mediane = float(
        np.median(
            differences_fond
        )
    )

    mad = float(
        np.median(
            np.abs(
                differences_fond
                - mediane
            )
        )
    )

    sigma_robuste = (
        1.4826 * mad
    )

    seuil_mad = float(
        mediane
        + facteur_mad
        * sigma_robuste
    )

    # ----------------------------------------------------------
    # Percentile
    # ----------------------------------------------------------

    seuil_percentile = float(
        np.percentile(
            differences_fond,
            percentile_fond
        )
    )

    # ----------------------------------------------------------
    # Comparaison
    # ----------------------------------------------------------

    epsilon = 1e-6

    if seuil_mad > epsilon:

        rapport_percentile_mad = (
            seuil_percentile
            / seuil_mad
        )

    else:

        rapport_percentile_mad = (
            float("inf")
            if seuil_percentile > epsilon
            else 1.0
        )

    percentile_suspect = (
        rapport_percentile_mad > 2.5
    )

    # ----------------------------------------------------------
    # Sélection du seuil
    # ----------------------------------------------------------

    if seuil_mad <= epsilon:

        seuil = seuil_percentile

        methode_seuil = (
            "percentile périphérique, "
            "seuil MAD nul"
        )

    elif rapport_percentile_mad < 0.8:

        seuil = seuil_percentile

        methode_seuil = (
            "percentile périphérique"
        )

    elif percentile_suspect:

        seuil = (
            mediane
            + 6.0
            * sigma_robuste
        )

        methode_seuil = (
            "MAD renforcé, "
            "percentile périphérique suspect"
        )

    else:

        seuil = max(
            seuil_mad,
            seuil_percentile
        )

        methode_seuil = (
            "maximum MAD / "
            "percentile périphérique"
        )

    # ----------------------------------------------------------
    # Marge et limites
    # ----------------------------------------------------------

    seuil += float(
        marge_seuil
    )

    seuil = max(
        float(seuil_min),
        float(seuil)
    )

    if seuil_max is not None:

        seuil = min(
            float(seuil_max),
            seuil
        )

    # ----------------------------------------------------------
    # Diagnostic console
    # ----------------------------------------------------------

    if afficher_informations:

        print()
        print("Calcul automatique du seuil")
        print(
            f"  Médiane          : {mediane:.2f}"
        )
        print(
            f"  MAD              : {mad:.2f}"
        )
        print(
            f"  Sigma robuste    : {sigma_robuste:.2f}"
        )
        print(
            f"  Seuil MAD        : {seuil_mad:.2f}"
        )
        print(
            f"  Seuil percentile : {seuil_percentile:.2f}"
        )
        print(
            f"  Méthode          : {methode_seuil}"
        )
        print(
            f"  Seuil final      : {seuil:.2f}"
        )

    informations = {
        "mediane_fond": mediane,
        "mad_fond": mad,
        "sigma_robuste_fond": sigma_robuste,
        "seuil_mad": seuil_mad,
        "seuil_percentile": seuil_percentile,
        "rapport_percentile_mad": rapport_percentile_mad,
        "percentile_suspect": bool(
            percentile_suspect
        ),
        "methode_seuil": methode_seuil,
        "percentile_fond": float(
            percentile_fond
        ),
        "taille_bord": int(
            taille_bord
        ),
        "nombre_pixels_fond": int(
            differences_fond.size
        ),
        "seuil_final": float(
            seuil
        )
    }

    return (
        float(seuil),
        informations
    )


# ==================================================================================================
# CRÉATION DU MASQUE
# ==================================================================================================

def creer_masque_bestiole(
    voir_image,
    rep,
    image,
    seuil=None,
    taille_bord=TAILLE_BORD,
    surface_min_pct=SURFACE_MIN_PCT,
    afficher_informations=False
):
    """
    Crée un masque contenant la principale composante détectée.

    Si seuil est None, le seuil est calculé automatiquement
    à partir des distances observées en périphérie.
    """

    if image is None:
        raise ValueError(
            "L'image transmise à "
            "creer_masque_bestiole() est None."
        )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "L'image doit être une image BGR "
            "à trois canaux."
        )

    hauteur, largeur = (
        image.shape[:2]
    )

    # ----------------------------------------------------------
    # Lissage
    # ----------------------------------------------------------

    image_lissee = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # ----------------------------------------------------------
    # Couleur du fond
    # ----------------------------------------------------------

    fond = couleur_fond(
        image=image_lissee,
        taille=taille_bord
    )

    fond_array = np.asarray(
        fond,
        dtype=np.float32
    )

    # ----------------------------------------------------------
    # Distance de chaque pixel au fond
    # ----------------------------------------------------------

    difference = np.linalg.norm(
        image_lissee.astype(
            np.float32
        )
        - fond_array,
        axis=2
    )

    # ----------------------------------------------------------
    # Calcul du seuil
    # ----------------------------------------------------------

    seuil_automatique = (
        seuil is None
    )

    informations_seuil = None

    if seuil_automatique:

        (
            seuil,
            informations_seuil
        ) = calculer_seuil_depuis_difference(
            afficher_informations=afficher_informations,
            difference=difference,
            taille_bord=taille_bord,
            percentile_fond=95.0,
            facteur_mad=4.0,
            marge_seuil=2.0,
            seuil_min=5.0,
            seuil_max=None
        )

    else:

        seuil = float(
            seuil
        )

    # ----------------------------------------------------------
    # Seuillage
    # ----------------------------------------------------------

    masque_initial = (
        difference > seuil
    ).astype(
        np.uint8
    ) * 255

    # ----------------------------------------------------------
    # Ouverture
    # ----------------------------------------------------------

    noyau_ouverture = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    )

    masque_nettoye = cv2.morphologyEx(
        masque_initial,
        cv2.MORPH_OPEN,
        noyau_ouverture
    )

    # ----------------------------------------------------------
    # Fermeture
    # ----------------------------------------------------------

    noyau_fermeture = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )
    )

    masque_nettoye = cv2.morphologyEx(
        masque_nettoye,
        cv2.MORPH_CLOSE,
        noyau_fermeture
    )

    # ----------------------------------------------------------
    # Composante principale
    # ----------------------------------------------------------

    (
        masque_bestiole,
        meilleur_label,
        meilleure_surface
    ) = selectionner_composante_principale(
        masque=masque_nettoye,
        surface_min_pct=surface_min_pct
    )

    # Pas de composante détectée
    if meilleur_label is None:

        masque_bestiole = np.zeros(
            (hauteur, largeur),
            dtype=np.uint8
        )

    # ----------------------------------------------------------
    # Diagnostic
    # ----------------------------------------------------------

    if afficher_informations:

        print()
        print("Création du masque")
        print(
            f"  Couleur fond : {fond}"
        )
        print(
            f"  Seuil        : {seuil:.3f}"
        )
        print(
            f"  Surface      : {meilleure_surface} px"
        )

    return (
        masque_bestiole,
        fond,
        difference,
        float(seuil)
    )


# ==================================================================================================
# CALCUL DE LA BOUNDING BOX
# ==================================================================================================

def detecter_zone_depuis_masque(
    masque
):
    """
    Calcule la bounding box à partir d'un masque déjà créé.

    Returns
    -------
    tuple | None
        (x_min, y_min, x_max, y_max)
    """

    positions_y, positions_x = np.where(
        masque > 0
    )

    if positions_y.size == 0:
        return None

    x_min = int(
        positions_x.min()
    )

    x_max = int(
        positions_x.max()
    )

    y_min = int(
        positions_y.min()
    )

    y_max = int(
        positions_y.max()
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )

def ajouter_marge_bbox(
    image,
    bbox,
    marge_pct=MARGE_BBOX_PCT,
    marge_min_px=MARGE_BBOX_MIN_PX
):
    """
    Ajoute une marge autour de la bounding box tout en
    restant dans les dimensions de l'image.
    """

    if bbox is None:
        return None

    hauteur, largeur = image.shape[:2]

    x_min, y_min, x_max, y_max = bbox

    marge_x = max(
        int(marge_min_px),
        int(round(largeur * marge_pct))
    )

    marge_y = max(
        int(marge_min_px),
        int(round(hauteur * marge_pct))
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

# ==================================================================================================
# CALCUL DE L'OCCUPATION
# ==================================================================================================

def calculer_occupation_depuis_masque(
    masque_bestiole
):
    """
    Calcule la surface de l'organisme et son occupation
    par rapport à la surface totale de l'image.

    Returns
    -------
    tuple
        surface_image,
        surface_bestiole,
        occupation_pct
    """

    surface_bestiole = int(
        np.count_nonzero(
            masque_bestiole
        )
    )

    surface_image = int(
        masque_bestiole.size
    )

    if surface_image == 0:

        occupation_pct = 0.0

    else:

        occupation_pct = (
            surface_bestiole
            / surface_image
            * 100.0
        )

    return (
        surface_image,
        surface_bestiole,
        occupation_pct
    )

def calculer_occupation_bbox(
    image,
    bbox
):
    """
    Calcule le pourcentage de l'image occupé par
    la bounding box.
    """

    if bbox is None:
        return 0.0

    hauteur, largeur = image.shape[:2]

    surface_image = (
        hauteur * largeur
    )

    x_min, y_min, x_max, y_max = bbox

    largeur_bbox = (
        x_max - x_min + 1
    )

    hauteur_bbox = (
        y_max - y_min + 1
    )

    if (
        largeur_bbox <= 0
        or hauteur_bbox <= 0
    ):
        return 0.0

    surface_bbox = (
        largeur_bbox
        * hauteur_bbox
    )

    return (
        surface_bbox
        / surface_image
        * 100.0
    )

def sauvegarder_bbox(
    image,
    bbox,
    chemin_image,
    suffixe
):
    """
    Sauvegarde le contenu d'une BBox dans le même
    répertoire que l'image d'origine.
    """

    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox

    image_crop = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    chemin_image = Path(
        chemin_image
    )

    nom_fichier = (
        f"{chemin_image.stem}_{suffixe}.jpg"
    )

    fichier_sortie = (
        chemin_image.parent
        / nom_fichier
    )

    success, buffer = cv2.imencode(
        ".jpg",
        image_crop
    )

    if not success:
        return None

    buffer.tofile(
        str(fichier_sortie)
    )

    return fichier_sortie

def sauvegarder_liste_erreurs(
    repertoire,
    nom_fichier,
    liste
):
    """
    Sauvegarde une liste de noms d'images dans
    un fichier texte.

    Les noms sont répartis sur plusieurs colonnes,
    alignés et séparés par ' | '.
    """

    repertoire = Path(repertoire)
    fichier_sortie = (repertoire / nom_fichier)

    # ----------------------------------------------------------
    # Aucune anomalie
    # ----------------------------------------------------------

    if not liste:

        # Suppression éventuelle d'un ancien rapport
        if fichier_sortie.exists():

            fichier_sortie.unlink()

        return None

    # ----------------------------------------------------------
    # Largeur de la colonne
    # ----------------------------------------------------------
    largeur_colonne = max(len(nom) for nom in liste)

    # ----------------------------------------------------------
    # Nombre de colonnes
    # ----------------------------------------------------------
    nombre_colonnes = cst.NB_COLONNES

    # ----------------------------------------------------------
    # Écriture
    # ----------------------------------------------------------

    with fichier_sortie.open(
        "w",
        encoding="utf-8"
    ) as fichier:

        for i in range(
            0,
            len(liste),
            nombre_colonnes
        ):

            ligne = liste[i:i + nombre_colonnes]

            ligne_formatee = " | ".join(
                f"{nom:<{largeur_colonne}}"
                for nom in ligne
            )

            fichier.write(
                ligne_formatee.rstrip()
                + "\n"
            )

    return fichier_sortie
