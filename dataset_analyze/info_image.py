from tools import system as syst
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tools import utility as util


# ==================================================================================================
# PARAMÈTRES
# ==================================================================================================

SEUIL_DETECTION = None
pas_seuil = 5
TAILLE_BORD = 20
SURFACE_MIN_PCT = 0.001

MARGE_BBOX_PCT = 0.03
MARGE_BBOX_MIN_PX = 5


# ==================================================================================================
# STRUCTURES DE DONNÉES
# ==================================================================================================

@dataclass
class InfoOccupationBestiole:
    occupation: float
    occupation_pct: float
    surface_bestiole: float
    surface_image: float


def afficher_histogramme_difference(
    difference,
    seuil
):
    """
    Affiche l'histogramme des distances à la couleur du fond.
    """

    plt.figure(
        figsize=(12, 6)
    )

    plt.hist(
        difference.ravel(),
        bins=100,
        color="steelblue",
        alpha=0.7
    )

    plt.axvline(
        seuil,
        color="red",
        linewidth=2,
        label=f"Seuil = {seuil:.1f}"
    )

    plt.xlabel(
        "Distance à la couleur du fond"
    )

    plt.ylabel(
        "Nombre de pixels"
    )

    plt.title(
        "Histogramme des distances au fond"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    p90 = np.percentile(difference, 90)
    p95 = np.percentile(difference, 95)
    p99 = np.percentile(difference, 99)

    plt.axvline(p90, color="green", linestyle="--", label=f"P90={p90:.1f}")
    plt.axvline(p95, color="orange", linestyle="--", label=f"P95={p95:.1f}")
    plt.axvline(p99, color="purple", linestyle="--", label=f"P99={p99:.1f}")





    plt.tight_layout()

    plt.show()





# ==================================================================================================
# LECTURE D'IMAGE
# ==================================================================================================

def lire_image(chemin_image):
    """
    Lit une image en prenant en charge les chemins comportant
    des caractères accentués ou Unicode.

    Parameters
    ----------
    chemin_image : str | Path
        Chemin de l'image.

    Returns
    -------
    np.ndarray
        Image OpenCV au format BGR.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas ou ne peut pas être lu.
    """

    chemin_image = Path(chemin_image)

    if not chemin_image.is_file():
        raise FileNotFoundError(
            f"Le fichier n'existe pas : {chemin_image}"
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
        raise FileNotFoundError(
            f"Impossible de lire l'image : {chemin_image}"
        ) from erreur

    if image is None:
        raise FileNotFoundError(
            f"Impossible de décoder l'image : {chemin_image}"
        )

    return image


# ==================================================================================================
# ANALYSE DE LA LUMINOSITÉ ET DU CONTRASTE
# ==================================================================================================

def analyser_luminosite(image):
    """
    Calcule la luminosité moyenne et l'écart-type
    de l'image en niveaux de gris.
    """

    image_grise = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    moyenne = float(
        np.mean(image_grise)
    )

    ecart_type = float(
        np.std(image_grise)
    )

    return moyenne, ecart_type


def parametres_lum_contraste(image):
    """
    Détermine les paramètres de luminosité et de contraste
    en fonction de l'écart-type de l'image.
    """

    _, sigma = analyser_luminosite(image)

    if sigma < 25:
        return 0.10, 0.10

    if sigma < 50:
        return 0.15, 0.15

    return 0.25, 0.25


def parametres_gamma(image):
    """
    Détermine une plage de gamma en fonction
    de l'écart-type de l'image.
    """

    _, sigma = analyser_luminosite(image)

    if sigma < 25:
        return 85, 115

    if sigma < 50:
        return 90, 110

    return 95, 105


def parametres_bruit(image):
    """
    Détermine la plage de bruit à appliquer en fonction
    de l'écart-type de l'image.
    """

    _, sigma = analyser_luminosite(image)

    if sigma < 25:
        return 0.005, 0.02

    if sigma < 50:
        return 0.01, 0.03

    return 0.01, 0.04


def seuil_detection(image):
    """
    Détermine automatiquement un seuil de détection
    selon l'écart-type de l'image.
    """

    _, sigma = analyser_luminosite(image)

    if sigma < 20:
        return 28

    if sigma < 35:
        return 35

    if sigma < 50:
        return 40

    return 45


# ==================================================================================================
# ESTIMATION DE LA COULEUR DU FOND
# ==================================================================================================

def couleur_fond(
    image,
    taille=20
):
    """
    Estime la couleur BGR du fond à partir des quatre coins.

    Parameters
    ----------
    image : np.ndarray
        Image OpenCV BGR.

    taille : int
        Taille en pixels des zones analysées dans les coins.

    Returns
    -------
    tuple
        Couleur médiane BGR du fond.
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

    Une composante touchant un bord n'est pas rejetée.

    Parameters
    ----------
    masque : np.ndarray
        Masque binaire uint8.

    surface_min_pct : float
        Surface minimale d'une composante par rapport
        à la surface totale de l'image.

    Returns
    -------
    masque_final : np.ndarray
        Masque contenant uniquement la composante retenue.

    meilleur_label : int | None
        Identifiant de la composante retenue.

    meilleure_surface : int
        Surface en pixels de la composante retenue.
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

    surface_image = hauteur * largeur

    surface_min = max(
        1,
        int(surface_image * surface_min_pct)
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




def calculer_seuil_depuis_difference(
    difference,
    taille_bord=20,
    percentile_fond=99.0,
    facteur_mad=4.0,
    marge_seuil=2.0,
    seuil_min=5.0,
    seuil_max=None
):
    """
    Calcule automatiquement le seuil de détection à partir
    des distances observées dans une bande périphérique.

    Parameters
    ----------
    difference : np.ndarray
        Carte des distances à la couleur estimée du fond.

    taille_bord : int
        Épaisseur de la bande périphérique analysée.

    percentile_fond : float
        Percentile des distances du fond utilisé pour
        éliminer la majorité des variations du fond.

    facteur_mad : float
        Nombre d'écarts-types robustes ajoutés à la médiane.

    marge_seuil : float
        Petite marge supplémentaire ajoutée au seuil.

    seuil_min : float
        Valeur minimale autorisée pour le seuil.

    seuil_max : float | None
        Valeur maximale autorisée. Si None, aucune limite
        maximale n'est appliquée.

    Returns
    -------
    seuil : float
        Seuil automatique calculé.

    informations : dict
        Informations permettant de diagnostiquer le calcul.
    """

    if difference is None:
        raise ValueError(
            "La carte de différence est None."
        )

    if difference.ndim != 2:
        raise ValueError(
            "La carte de différence doit comporter deux dimensions."
        )

    hauteur, largeur = difference.shape

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

    # Création d'un masque correspondant à toute la périphérie
    masque_bord = np.zeros(
        (hauteur, largeur),
        dtype=bool
    )

    masque_bord[:taille_bord, :] = True
    masque_bord[-taille_bord:, :] = True
    masque_bord[:, :taille_bord] = True
    masque_bord[:, -taille_bord:] = True

    differences_fond = difference[masque_bord]

    differences_fond = differences_fond[
        np.isfinite(differences_fond)
    ]

    if differences_fond.size == 0:
        raise ValueError(
            "Aucune distance valide n'a été trouvée sur les bords."
        )

    mediane = float(
        np.median(differences_fond)
    )

    mad = float(
        np.median(
            np.abs(
                differences_fond - mediane
            )
        )
    )

    # Conversion du MAD en estimation robuste de l'écart-type
    sigma_robuste = 1.4826 * mad

    seuil_mad = (
        mediane
        + facteur_mad * sigma_robuste
    )

    seuil_percentile = float(
        np.percentile(
            differences_fond,
            percentile_fond
        )
    )

    # On retient la méthode la plus prudente
    seuil = max(
        seuil_mad,
        seuil_percentile
    )

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

    informations = {
        "mediane_fond": mediane,
        "mad_fond": mad,
        "sigma_robuste_fond": sigma_robuste,
        "seuil_mad": float(seuil_mad),
        "seuil_percentile": seuil_percentile,
        "percentile_fond": float(percentile_fond),
        "nombre_pixels_fond": int(differences_fond.size)
    }

    return float(seuil), informations


# ==================================================================================================
# CRÉATION DU MASQUE
# ==================================================================================================

def creer_masque_bestiole(
    image,
    seuil=None,
    taille_bord=TAILLE_BORD,
    surface_min_pct=SURFACE_MIN_PCT,
    afficher_informations=False
):
    """
    Crée un masque contenant la principale composante détectée.

    Si seuil est None, le seuil est calculé automatiquement à
    partir des distances observées dans la périphérie de l'image.
    """

    if image is None:
        raise ValueError(
            "L'image transmise à creer_masque_bestiole() "
            "est None."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "L'image doit être une image BGR à trois canaux."
        )

    hauteur, largeur = image.shape[:2]

    # ------------------------------------------------------------------
    # Lissage léger
    # ------------------------------------------------------------------

    image_lissee = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # ------------------------------------------------------------------
    # Estimation de la couleur du fond
    # ------------------------------------------------------------------

    fond = couleur_fond(
        image=image_lissee,
        taille=taille_bord
    )

    fond_array = np.asarray(
        fond,
        dtype=np.float32
    )

    # ------------------------------------------------------------------
    # Distance euclidienne BGR de chaque pixel au fond
    # ------------------------------------------------------------------

    difference = np.linalg.norm(
        image_lissee.astype(np.float32) - fond_array,
        axis=2
    )

    # ------------------------------------------------------------------
    # Calcul automatique du seuil APRÈS le calcul de difference
    # ------------------------------------------------------------------

    seuil_automatique = seuil is None
    informations_seuil = None

    if seuil_automatique:

        seuil, informations_seuil = (
            calculer_seuil_depuis_difference(
                difference=difference,
                taille_bord=taille_bord,
                percentile_fond=99.0,
                facteur_mad=4.0,
                marge_seuil=2.0,
                seuil_min=5.0,
                seuil_max=None
            )
        )

    else:

        seuil = float(seuil)

    # ------------------------------------------------------------------
    # Seuillage
    # ------------------------------------------------------------------

    masque_initial = (
        difference > seuil
    ).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Ouverture
    # ------------------------------------------------------------------

    noyau_ouverture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    masque_nettoye = cv2.morphologyEx(
        masque_initial,
        cv2.MORPH_OPEN,
        noyau_ouverture
    )

    # ------------------------------------------------------------------
    # Fermeture
    # ------------------------------------------------------------------

    noyau_fermeture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    masque_nettoye = cv2.morphologyEx(
        masque_nettoye,
        cv2.MORPH_CLOSE,
        noyau_fermeture
    )

    # ------------------------------------------------------------------
    # Conservation de la composante principale
    # ------------------------------------------------------------------

    (
        masque_bestiole,
        meilleur_label,
        meilleure_surface
    ) = selectionner_composante_principale(
        masque=masque_nettoye,
        surface_min_pct=surface_min_pct
    )

    if meilleur_label is None:

        methode_detection = "aucune détection"

        masque_bestiole = np.zeros(
            (hauteur, largeur),
            dtype=np.uint8
        )

    else:

        methode_detection = (
            "distance à la couleur du fond"
        )

    # ------------------------------------------------------------------
    # Informations
    # ------------------------------------------------------------------

    if afficher_informations:

        pixels_masque_initial = int(
            np.count_nonzero(masque_initial)
        )

        pixels_masque_final = int(
            np.count_nonzero(masque_bestiole)
        )

        print()
        print("Informations sur la création du masque :")
        print(f"  Méthode               : {methode_detection}")
        print(f"  Couleur du fond BGR   : {fond}")

        if seuil_automatique:
            print("  Type de seuil         : automatique")
        else:
            print("  Type de seuil         : manuel")


        p_m_i = util.format_nombre(pixels_masque_initial)
        p_m_f = util.format_nombre(pixels_masque_final)
        m_s   = util.format_nombre(meilleure_surface)

        print(f"  Seuil utilisé         : {seuil:.3f}")
        print(f"  Pixels masque initial : {p_m_i}")
        print(f"  Pixels masque final   : {p_m_f}")
        print(f"  Composante principale : {m_s} px")

        if informations_seuil is not None:

            print()
            print("  Calcul automatique du seuil :")

            print(
                f"    Médiane du fond      : "
                f"{informations_seuil['mediane_fond']:.3f}"
            )

            print(
                f"    MAD du fond          : "
                f"{informations_seuil['mad_fond']:.3f}"
            )

            print(
                f"    Sigma robuste        : "
                f"{informations_seuil['sigma_robuste_fond']:.3f}"
            )

            print(
                f"    Seuil médiane + MAD  : "
                f"{informations_seuil['seuil_mad']:.3f}"
            )

            print(
                f"    Seuil percentile     : "
                f"{informations_seuil['seuil_percentile']:.3f}"
            )

            print(
                f"    Pixels analysés      : "
                f"{informations_seuil['nombre_pixels_fond']:,}"
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

def detecter_zone_depuis_masque(masque):
    """
    Calcule la bounding box à partir d'un masque déjà créé.

    Returns
    -------
    tuple | None
        (x_min, y_min, x_max, y_max).
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


def detecter_zone_bestiole(
    image,
    seuil=None
):
    """
    Crée le masque puis calcule la bounding box.

    Cette fonction reste disponible pour une utilisation
    indépendante. Dans le programme principal, le masque
    n'est calculé qu'une seule fois.
    """

    masque_bestiole, _, _, _ = creer_masque_bestiole(
        image=image,
        seuil=seuil
    )

    return detecter_zone_depuis_masque(
        masque_bestiole
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

def calculer_occupation_depuis_masque(masque_bestiole):
    """
    Calcule l'occupation réelle à partir d'un masque existant.
    """

    surface_bestiole = int(
        np.count_nonzero(masque_bestiole)
    )

    surface_image = int(
        masque_bestiole.size
    )

    if surface_image == 0:
        occupation = 0.0
    else:
        occupation = (
            surface_bestiole
            / surface_image
        )

    occupation_pct = (
        occupation * 100.0
    )

    return InfoOccupationBestiole(
        occupation=float(occupation),
        occupation_pct=float(occupation_pct),
        surface_bestiole=float(surface_bestiole),
        surface_image=float(surface_image)
    )


def calculer_occupation(
    image,
    seuil=None
):
    """
    Crée un masque et calcule l'occupation réelle.

    Pour éviter les recalculs, utiliser de préférence
    calculer_occupation_depuis_masque() lorsqu'un masque
    est déjà disponible.
    """

    masque_bestiole, _, _, _ = creer_masque_bestiole(
        image=image,
        seuil=seuil
    )

    return calculer_occupation_depuis_masque(
        masque_bestiole
    )


def calculer_occupation_bbox(
    image,
    bbox
):
    """
    Calcule le rapport entre la surface de la bounding box
    et la surface totale de l'image.
    """

    if bbox is None:
        return 0.0

    hauteur, largeur = image.shape[:2]

    surface_image = hauteur * largeur

    x_min, y_min, x_max, y_max = bbox

    largeur_bbox = (
        x_max - x_min + 1
    )

    hauteur_bbox = (
        y_max - y_min + 1
    )

    if largeur_bbox <= 0 or hauteur_bbox <= 0:
        return 0.0

    surface_bbox = (
        largeur_bbox * hauteur_bbox
    )

    return float(
        surface_bbox / surface_image
    )


# ==================================================================================================
# CALCUL DU ZOOM MAXIMAL
# ==================================================================================================

def calculer_zoom_max(
    image,
    bbox
):
    """
    Calcule le zoom maximal centré sur l'image permettant
    de conserver la bounding box dans l'image.
    """

    if bbox is None:
        return 1.0

    x_min, y_min, x_max, y_max = bbox

    hauteur, largeur = image.shape[:2]

    centre_x = largeur / 2.0
    centre_y = hauteur / 2.0

    limites = []

    distance_gauche = centre_x - x_min
    distance_droite = x_max - centre_x
    distance_haut = centre_y - y_min
    distance_bas = y_max - centre_y

    if distance_gauche > 0:
        limites.append(
            centre_x / distance_gauche
        )

    if distance_droite > 0:
        limites.append(
            (largeur - 1 - centre_x)
            / distance_droite
        )

    if distance_haut > 0:
        limites.append(
            centre_y / distance_haut
        )

    if distance_bas > 0:
        limites.append(
            (hauteur - 1 - centre_y)
            / distance_bas
        )

    limites_valides = [
        limite
        for limite in limites
        if np.isfinite(limite) and limite > 0
    ]

    if not limites_valides:
        return 1.0

    zoom_max = min(
        limites_valides
    )

    return max(
        1.0,
        float(zoom_max)
    )


# ==================================================================================================
# DIAGNOSTIC
# ==================================================================================================

def afficher_carte_difference(
    difference,
    seuil
):
    """
    Affiche la carte des distances avec le seuil utilisé.
    """

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        difference,
        cmap="hot"
    )

    plt.colorbar(
        label="Distance BGR au fond"
    )

    plt.title(
        f"Distance à la couleur du fond, seuil = {seuil}"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def afficher_comparaison_seuils(
    difference,
    seuil
):
    """
    Affiche les pixels supprimés lorsque le seuil passe
    de seuil à seuil + pas.
    """

    seuil_suivant = seuil + pas_seuil

    pixels_critiques = (
        (difference > seuil)
        & (difference <= seuil_suivant)
    ).astype(np.uint8) * 255

    nombre_pixels_critiques = int(
        np.count_nonzero(pixels_critiques)
    )

    nb = util.format_nombre(nombre_pixels_critiques)
    print(
        f"Pixels conservés à {seuil:.3f} mais supprimés à "
        f"{seuil_suivant:.3f} : {nb}"
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        pixels_critiques,
        cmap="gray"
    )

    plt.title(
        f"Pixels perdus entre les seuils "
        f"{seuil:.3f} et {seuil_suivant:.3f}"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def diagnostiquer_detection_bestiole(
    image,
    masque_bestiole,
    fond,
    difference,
    seuil,
    bbox_detectee=None,
    bbox_securisee=None
):
    """
    Affiche le masque, la carte des distances, les bounding
    boxes et les informations numériques de diagnostic.
    """

    hauteur, largeur = image.shape[:2]

    # ------------------------------------------------------------------
    # Affichage du masque
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        masque_bestiole,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.title(
        "Masque nettoyé de la bestiole"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Affichage de la carte des distances
    # ------------------------------------------------------------------

    afficher_carte_difference(
        difference=difference,
        seuil=seuil
    )

    # ------------------------------------------------------------------
    # Histogramme des distances
    # ------------------------------------------------------------------

    afficher_histogramme_difference(
    difference=difference,
    seuil=seuil
    )



    # ------------------------------------------------------------------
    # Visualisation des pixels critiques entre seuil et seuil + pas
    # ------------------------------------------------------------------

    afficher_comparaison_seuils(
        difference=difference,
        seuil=seuil
    )

    # ------------------------------------------------------------------
    # Vérification de la détection
    # ------------------------------------------------------------------

    if bbox_detectee is None:

        print()
        print("-" * 60)
        print("DIAGNOSTIC DE LA DÉTECTION")
        print("-" * 60)
        print("Aucune bestiole détectée.")
        print(f"Couleur du fond estimée : {fond}")
        print(f"Seuil utilisé            : {seuil}")

        return

    x_min, y_min, x_max, y_max = bbox_detectee

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

    marge_gauche_pct = (
        marge_gauche / largeur * 100.0
    )

    marge_droite_pct = (
        marge_droite / largeur * 100.0
    )

    marge_haut_pct = (
        marge_haut / hauteur * 100.0
    )

    marge_bas_pct = (
        marge_bas / hauteur * 100.0
    )

    # ------------------------------------------------------------------
    # Affichage des informations
    # ------------------------------------------------------------------

    print()
    print("-" * 60)
    print("DIAGNOSTIC DE LA DÉTECTION")
    print("-" * 60)

    print(f"Couleur du fond estimée : {fond}")
    print(f"Seuil de détection      : {seuil:.3f}")

    fond_pct = np.sum(difference < seuil)
    objet_pct = np.sum(difference >= seuil)

    fond_pct = fond_pct / difference.size * 100
    objet_pct = objet_pct / difference.size * 100

    print()
    print(f"Fond   : {fond_pct:.1f}%")
    print(f"Objet  : {objet_pct:.1f}%")


    print()
    print("Zone détectée :")
    print(f"  Gauche : {x_min} px")
    print(f"  Haut   : {y_min} px")
    print(f"  Droite : {x_max} px")
    print(f"  Bas    : {y_max} px")

    print()
    print(f"P90 : {np.percentile(difference,90):.1f}")
    print(f"P95 : {np.percentile(difference,95):.1f}")
    print(f"P98 : {np.percentile(difference,98):.1f}")
    print(f"P99 : {np.percentile(difference,99):.1f}")

    print(f"Médiane : {np.median(difference):.1f}")
    print(f"Maximum : {difference.max():.1f}")

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
    print(f"Marge minimale : {marge_min} px")

    # ------------------------------------------------------------------
    # Dessin des bounding boxes
    # ------------------------------------------------------------------

    image_affichage = image.copy()

    x_min, y_min, x_max, y_max = bbox_detectee

    cv2.rectangle(
        image_affichage,
        (x_min, y_min),
        (x_max, y_max),
        (0, 0, 255),
        2
    )

    if bbox_securisee is not None:

        (
            x_min_marge,
            y_min_marge,
            x_max_marge,
            y_max_marge
        ) = bbox_securisee

        cv2.rectangle(
            image_affichage,
            (x_min_marge, y_min_marge),
            (x_max_marge, y_max_marge),
            (0, 255, 0),
            2
        )

    image_rgb = cv2.cvtColor(
        image_affichage,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        image_rgb
    )

    plt.title(
        "BBox détectée en rouge, bbox sécurisée en vert"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ==================================================================================================
# PROGRAMME PRINCIPAL
# ==================================================================================================

def traiter_image(
    chemin_image,
    seuil=None
):
    """
    Analyse une image et affiche les informations nécessaires
    pour déterminer le zoom et les augmentations.
    """

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    image = lire_image(
        chemin_image
    )

    # ------------------------------------------------------------------
    # Création unique du masque
    # ------------------------------------------------------------------

    (
        masque_bestiole,
        fond,
        difference,
        seuil_utilise
    ) = creer_masque_bestiole(
        image=image,
        seuil=seuil,
        taille_bord=TAILLE_BORD,
        surface_min_pct=SURFACE_MIN_PCT,
        afficher_informations=True
    )

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    bbox_detectee = detecter_zone_depuis_masque(
        masque_bestiole
    )

    bbox_securisee = ajouter_marge_bbox(
        image=image,
        bbox=bbox_detectee,
        marge_pct=MARGE_BBOX_PCT,
        marge_min_px=MARGE_BBOX_MIN_PX
    )

    # ------------------------------------------------------------------
    # Occupation
    # ------------------------------------------------------------------

    info_occ = calculer_occupation_depuis_masque(
        masque_bestiole
    )

    info_image_bboxe = calculer_occupation_bbox(
        image=image,
        bbox=bbox_securisee
    )

    # ------------------------------------------------------------------
    # Diagnostic graphique
    # ------------------------------------------------------------------

    diagnostiquer_detection_bestiole(
        image=image,
        masque_bestiole=masque_bestiole,
        fond=fond,
        difference=difference,
        seuil=seuil_utilise,
        bbox_detectee=bbox_detectee,
        bbox_securisee=bbox_securisee
    )

    # ------------------------------------------------------------------
    # Affichage des surfaces
    # ------------------------------------------------------------------

    print()
    print(f"Surface image     : {info_occ.surface_image:,.0f} px")
    print(f"Surface bestiole  : {info_occ.surface_bestiole:,.0f} px")

    print(
        f"Occupation réelle : {info_occ.occupation:.3f}, "
        f"soit {info_occ.occupation_pct:.1f} %"
    )

    print(
        f"Occupation BBox   : {info_image_bboxe:.3f}, "
        f"soit {info_image_bboxe * 100:.1f} %"
    )

    occupation_pct = info_occ.occupation_pct

    # ------------------------------------------------------------------
    # Calcul du zoom maximal
    # ------------------------------------------------------------------

    if bbox_securisee is None:

        zoom_max_possible = 1.0
        zoom_limite = 1.0

    else:

        zoom_max_possible = calculer_zoom_max(
            image=image,
            bbox=bbox_securisee
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

    # ------------------------------------------------------------------
    # Paramètres d'augmentation
    # ------------------------------------------------------------------

    moyenne, ecart_type = analyser_luminosite(
        image
    )

    brightness, contrast = parametres_lum_contraste(
        image
    )

    gamma = parametres_gamma(
        image
    )

    noise_range = parametres_bruit(
        image
    )

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    print()
    print("-" * 80)

    print(
        f" - Zoom maxi possible/retenu         : "
        f"{zoom_max_possible:.2f} / {zoom_max:.2f}"
    )

    print(f" - Luminosité                        : {moyenne:.3f}")
    print(f" - Écart-type                        : {ecart_type:.3f}")

    print(
        f" - Paramètres luminosité/contraste   : "
        f"{brightness:.3f} / {contrast:.3f}"
    )

    print(f" - Plage de réglage gamma            : {gamma}")
    print(f" - Plage de réglage du bruit         : {noise_range}")

    print("-" * 80)
    print()

    return {
        "image": image,
        "masque": masque_bestiole,
        "fond": fond,
        "difference": difference,
        "bbox_detectee": bbox_detectee,
        "bbox_securisee": bbox_securisee,
        "occupation": info_occ,
        "zoom_max_possible": zoom_max_possible,
        "zoom_max": zoom_max
    }


def main():
    """
    Boucle principale du programme.
    """

    continuer = True

    while continuer:

        syst.clear_screen()

        chemin_image = util.get_path_color(
            "Sélectionner l'image"
        )

        try:

            traiter_image(
                chemin_image=chemin_image,
                seuil=SEUIL_DETECTION
            )

        except Exception as erreur:

            print()
            print("-" * 80)
            print("ERREUR PENDANT LE TRAITEMENT")
            print("-" * 80)
            print(f"Type    : {type(erreur).__name__}")
            print(f"Message : {erreur}")
            print("-" * 80)
            print()

        continuer = util.answer_yes_or_no(
            "Voulez-vous continuer",
            True
        )

    print("Fin du traitement !!!")


if __name__ == "__main__":
    main()