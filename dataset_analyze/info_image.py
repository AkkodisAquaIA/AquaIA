from tools import system as syst
from pathlib import Path
from dataclasses import dataclass

import traceback
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

from tools import utility as util
from config import constants as cst
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from tools import menu_color as menu_c


display = dc.DisplayColor()


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

class ErreurLectureImage(Exception):
    """Erreur lors de la lecture ou du décodage d'une image."""
    pass


def afficher_erreur(erreur):

    print()
    display.print("-" * 80, colors["error"])
    display.print("ERREUR PENDANT LE TRAITEMENT", colors["error"])
    display.print("-" * 80, colors["error"])
    display.print(f"Type    : {type(erreur).__name__}", colors["error"])
    display.print(f"Message : {erreur}", colors["error"])

    if cst.DEBUG_MOODE:
        print()
        traceback.print_exc()
        print()

    display.print("-" * 80, colors["error"])
    print()


def sauvegarder_liste_erreurs(
    repertoire, # Répertoire de sauvegarde
    fichier,    # Nom du fichier à créer
    liste,      # LIste des erreurs

):
    """
    Sauvegarde une liste de noms d'images dans un fichier texte.

    Les éléments sont répartis sur plusieurs colonnes,
    alignés et séparés par ' | '.
    """

    if not liste:
        return


    fichier_sortie = (repertoire  / fichier)

    # Largeur nécessaire pour la colonne la plus longue
    largeur_colonne = max(
        len(nom)
        for nom in liste
    )

    with fichier_sortie.open( "w", encoding="utf-8") as fichier:

        for i in range(
            0,
            len(liste),
            cst.NB_COLONNES
        ):

            ligne = liste[i:i + cst.NB_COLONNES]

            ligne_formatee = " | ".join(
                f"{nom:<{largeur_colonne}}"
                for nom in ligne
            )

            fichier.write( ligne_formatee.rstrip() + "\n")

    # Message 
    display.print("Liste sauvegardée dans le fichier : ", colors["info"])
    print(f"    -  {fichier_sortie}")


def afficher_histogramme_difference(
    difference,
    seuil
):
    """
    Affiche l'histogramme des distances à la couleur du fond.
    """

    plt.figure(figsize=(12, 8)
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

    plt.xlabel("Distance à la couleur du fond")
    plt.ylabel("Nombre de pixels")
    plt.title("Histogramme des distances au fond")

    plt.legend()
    plt.grid(alpha=0.3)

    p90 = np.percentile(difference, 90)
    p95 = np.percentile(difference, 95)
    p99 = np.percentile(difference, 99)

    plt.axvline(p90, color="green", linestyle="--", label=f"P90={p90:.1f}")
    plt.axvline(p95, color="orange", linestyle="--", label=f"P95={p95:.1f}")
    plt.axvline(p99, color="purple", linestyle="--", label=f"P99={p99:.1f}")

    plt.tight_layout()

    plt.show()


def sauvegarder_bbox(
    image,
    bbox,
    chemin_image,
    suffixe
):
    """
    Sauvegarde le contenu d'une bbox dans le même
    répertoire que l'image d'origine.
    """

    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox

    image_crop = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    chemin_image = Path(chemin_image)

    nom_fichier = f"{chemin_image.stem}_{suffixe}.jpg" 

    fichier_sortie = (
        chemin_image.parent
        / nom_fichier
    )

    success, buffer = cv2.imencode(
        ".jpg",
        image_crop
    )

    if success:
        buffer.tofile(
            str(fichier_sortie)
        )

    return  nom_fichier  #fichier_sortie



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

    # ----- Controle état des coins ---------------------------------
    ecart_type = np.std(
    pixels_coins,
    axis=0
    )

    if False :
        print(
            f"Ecart-type fond : "
            f"{ecart_type.mean():.1f}"
        )

        if ecart_type.mean() < 10.0 :
            print("Les coins représente bien le fond")
        elif ecart_type.mean() > 40.0 :
            print("Une partioe de la bestiole touche un coin !!")


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
    afficher_informations,
    difference,
    taille_bord=20,
    percentile_fond=95.0,  # 99.0
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

    # Création du masque de la bande périphérique
    masque_bord = np.zeros(
        (hauteur, largeur),
        dtype=bool
    )

    masque_bord[:taille_bord, :] = True
    masque_bord[-taille_bord:, :] = True
    masque_bord[:, :taille_bord] = True
    masque_bord[:, -taille_bord:] = True

    # Récupération des différences situées sur la périphérie
    differences_fond = difference[masque_bord]

    
    if afficher_informations :
        print(f"Min fond : {differences_fond.min():.2f}")
        print(f"Max fond : {differences_fond.max():.2f}")
        print(f"Moyenne fond : {differences_fond.mean():.2f}")
        print(f"Pixels non nuls : {np.count_nonzero(differences_fond)}")
        print(f"Nombre pixels : {differences_fond.size}")


    # Suppression des éventuelles valeurs NaN ou infinies
    differences_fond = differences_fond[
        np.isfinite(differences_fond)
    ]

    if differences_fond.size == 0:
        raise ValueError("Aucune distance valide n'a été trouvée sur les bords.")

    # --------------------------------------------------------
    # Calcul du seuil robuste fondé sur le MAD
    # --------------------------------------------------------

    mediane = float(np.median(differences_fond))

    mad = float(
        np.median(
            np.abs(
                differences_fond - mediane
            )
        )
    )

    # Conversion du MAD en estimation robuste de l'écart-type
    sigma_robuste = 1.4826 * mad

    seuil_mad = float(
        mediane
        + facteur_mad * sigma_robuste
    )

    # --------------------------------------------------------
    # Calcul du seuil fondé sur le percentile
    # --------------------------------------------------------

    seuil_percentile = float(
        np.percentile(
            differences_fond,
            percentile_fond
        )
    )

    # --------------------------------------------------------
    # Comparaison des deux méthodes
    # --------------------------------------------------------

    epsilon = 1e-6

    if seuil_mad > epsilon:

        rapport_percentile_mad = (
            seuil_percentile / seuil_mad
        )

        percentile_suspect = (
            rapport_percentile_mad > 2.5
        )

    else:

        # Le MAD ne fournit aucun seuil exploitable.
        rapport_percentile_mad = (
            float("inf")
            if seuil_percentile > epsilon
            else 1.0
        )

        percentile_suspect = False

    # --------------------------------------------------------
    # Sélection du seuil
    # --------------------------------------------------------

    if seuil_mad <= epsilon:

        # Fond parfaitement uniforme :
        # le MAD est nul, donc utilisation du percentile.
        seuil = seuil_percentile

        methode_seuil = (
            "percentile périphérique, seuil MAD nul"
        )

    elif percentile_suspect:

        # Le percentile est très supérieur au seuil MAD.
        # Une partie de l'organisme ou un artefact se trouve
        # probablement dans la bande périphérique.
        seuil = (
            mediane
            + 6.0 * sigma_robuste
        )

        methode_seuil = (
            "MAD renforcé, percentile périphérique suspect"
        )

    else:

        # Cas normal : utilisation de la méthode la plus prudente.
        seuil = max(
            seuil_mad,
            seuil_percentile
        )

        methode_seuil = (
            "maximum MAD / percentile périphérique"
        )

    # --------------------------------------------------------
    # Application de la marge et des limites
    # --------------------------------------------------------

    seuil += float(marge_seuil)

    seuil = max(float(seuil_min), float(seuil))

    if seuil_max is not None:
        seuil = min(float(seuil_max), seuil)

    # seuil à utiliser 

    # seuil = seuil * 0.8            # seuil_percentile        
    if rapport_percentile_mad < 0.8:
        seuil = seuil_percentile

    elif rapport_percentile_mad > 2.5:
        seuil = mediane + 6 * sigma_robuste

    else:
        seuil = max(seuil_mad,  seuil_percentile)

 

    # --------------------------------------------------------
    # Informations de diagnostic
    # --------------------------------------------------------

    if afficher_informations : 

        print(f"Médiane du fond         : {mediane:.2f}")
        print(f"MAD du fond             : {mad:.2f}")
        print(f"Sigma robuste du fond   : {sigma_robuste:.2f}")
        print(f"Seuil MAD               : {seuil_mad:.2f}")
        print(f"Seuil percentile        : {seuil_percentile:.2f}")

        if np.isfinite(rapport_percentile_mad):

            print(
                f"Rapport percentile/MAD: "
                f"{rapport_percentile_mad:.2f}"
            )

        else:

            print(
                "Rapport percentile/MAD : infini "
                "(seuil MAD nul)"
            )

        print(f"Méthode retenue         : {methode_seuil}")
        print(f"Seuil automatique final : {seuil:.2f}")

    informations = {
        "mediane_fond"           : mediane,
        "mad_fond"               : mad,
        "sigma_robuste_fond"     : sigma_robuste,
        "seuil_mad"              : seuil_mad,
        "seuil_percentile"       : seuil_percentile,
        "rapport_percentile_mad" : rapport_percentile_mad,
        "percentile_suspect"     : bool(percentile_suspect),
        "methode_seuil"          : methode_seuil,
        "percentile_fond"        : float(percentile_fond),
        "taille_bord"            : int(taille_bord),
        "nombre_pixels_fond"     : int(differences_fond.size),
        "seuil_final"            : float(seuil)
    }

    return float(seuil), informations



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
    afficher_informations= False
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


    # Normalisation entre 0 et 255
    visu = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


    if voir_image :
        plt.figure(figsize=(12, 8))
        plt.imshow(visu, cmap="gray")  # retirer cmap si image couleur
        plt.title("Difference Visualisation")
        plt.axis("off")
        plt.show()


    # ------------------------------------------------------------------
    # Calcul automatique du seuil APRÈS le calcul de difference
    # ------------------------------------------------------------------

    seuil_automatique = seuil is None
    informations_seuil = None

    if seuil_automatique:

        seuil, informations_seuil = (
                calculer_seuil_depuis_difference(
                afficher_informations=afficher_informations,
                difference=difference,
                taille_bord=taille_bord,
                percentile_fond= 95.0,   #  99.0,
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
    
    if afficher_informations :

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
            print("  Type de seuil      : automatique")
        else:
            print("  Type de seuil      : manuel")


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

    positions_y, positions_x = np.where(masque > 0)

    if positions_y.size == 0:
        return None

    x_min = int(positions_x.min())
    x_max = int(positions_x.max())
    y_min = int(positions_y.min())
    y_max = int(positions_y.max())

    return (
        x_min, y_min,
        x_max, y_max
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
        voir_image=voir_image,
        rep=repertoire, 
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

    x_min = max(0, x_min - marge_x)
    y_min = max(0, y_min - marge_y)
    x_max = min( largeur - 1, x_max + marge_x)
    y_max = min( hauteur - 1, y_max + marge_y)

    return (
        x_min, y_min,
        x_max, y_max
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
        voir_image=voir_image,
        rep=repertoire,
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
# DIAGNOSTIC
# ==================================================================================================

def afficher_carte_difference(
    difference,
    seuil
):
    """
    Affiche la carte des distances avec le seuil utilisé.
    """

    plt.figure(figsize=(12, 8))
    plt.imshow(difference, cmap="hot")
    plt.colorbar(label="Distance BGR au fond")
    plt.title(f"Distance à la couleur du fond, seuil = {seuil}")

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

    plt.figure(figsize=(12, 8))
    plt.imshow(pixels_critiques, cmap="gray")
    plt.title(
        f"Pixels perdus entre les seuils "
        f"{seuil:.3f} et {seuil_suivant:.3f}"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def diagnostiquer_detection_bestiole(
    voir_image,
    afficher_informations,
    image,
    masque_bestiole,
    fond,
    difference,
    seuil,
    bbox_detectee=None,
    bbox_securisee=None,
    ):
    """
    Affiche le masque, la carte des distances, les bounding
    boxes et les informations numériques de diagnostic.
    """

    hauteur, largeur = image.shape[:2]


    if voir_image : 

        # ------------------------------------------------------------------
        # Affichage du masque
        # ------------------------------------------------------------------

        plt.figure(figsize=(12, 8))
        plt.imshow(
            masque_bestiole,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.title("Masque nettoyé de la bestiole")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        #------------------------------------------------------------------
        #Affichage de la carte des distances
        #------------------------------------------------------------------
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
        display.print("Aucune bestiole détectée !!\n", colors['warning'])

        return False

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
    marge_gauche_pct = (marge_gauche / largeur * 100.0)
    marge_droite_pct = (marge_droite / largeur * 100.0)
    marge_haut_pct = (marge_haut / hauteur * 100.0)
    marge_bas_pct = (marge_bas / hauteur * 100.0) 

    # ------------------------------------------------------------------
    # Affichage des informations
    # ------------------------------------------------------------------

    if afficher_informations : 

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
        if marge_min == 0:
            print("Objet déjà recadré au plus près.")

        nombre_bords_touches = sum([
            x_min == 0,
            y_min == 0,
            x_max >= largeur - 1,
            y_max >= hauteur - 1
        ])
        print(f"\n Nombre de bord touchant : {nombre_bords_touches}")


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

    image_rgb = cv2.cvtColor(image_affichage, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title("BBox détectée en rouge, bbox sécurisée en vert")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    return True


# ==================================================================================================
# PROGRAMME PRINCIPAL
# ==================================================================================================

def traiter_image(
    rep,
    voir_image,
    afficher_informations,
    chemin_image,
    seuil=None,
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
        voir_image=voir_image,
        rep= rep,
        image=image,
        seuil=seuil,
        taille_bord=TAILLE_BORD,
        surface_min_pct=SURFACE_MIN_PCT,
        afficher_informations= afficher_informations
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

    detection = diagnostiquer_detection_bestiole(
        voir_image = voir_image,
        afficher_informations = afficher_informations,
        image = image,
        masque_bestiole = masque_bestiole,
        fond = fond,
        difference =  difference,
        seuil = seuil_utilise,
        bbox_detectee = bbox_detectee,
        bbox_securisee = bbox_securisee,
    )


    if detection :

        # ------------------------------------------------------------------
        # Affichage des surfaces
        # ------------------------------------------------------------------

        print()
        display.print(f"Surface image     : {info_occ.surface_image:,.0f} px", colors["info"])
        display.print(f"Surface bestiole  : {info_occ.surface_bestiole:,.0f} px", colors["info"])
        
        display.print(f"Occupation réelle : {info_occ.occupation_pct:.1f} %" , colors["info"])
        display.print( f"Occupation BBox  : {info_image_bboxe * 100:.1f} %" , colors["info"])

        occupation_pct = info_occ.occupation_pct


        # # ------------------------------------------------------------------
        # # Sauvegarde de l"image 
        # # ------------------------------------------------------------------

        display.header_title("Sauvegade de l'image", colors["aqua"], chemin_image.stem)

        menu_saucegarde = menu_c.Menu('SAUVEGARDE',   # Nom du menu
                            style= "rounds",          # Style du menu 
                            theme = menu_c.AQUA_IA    # Theme du menu
                            )
        menu_saucegarde.display_menu()
        choix = menu_saucegarde.selection()

        # Sauvegarde Bbox Rougr
        if choix == 1 :

            fichier = sauvegarder_bbox(
                image=image,
                bbox=bbox_detectee,
                chemin_image=chemin_image,
                suffixe="Bbox_Rouge"
            )

            print(f"Image sauvegardée : {fichier}\n")

        # Sauvegarde Bbox Verte
        if choix == 2 :

            fichier = sauvegarder_bbox(
                image=image,
                bbox=bbox_securisee,
                chemin_image=chemin_image,
                suffixe="Bbox_Verte"
            )

            print(f"Image sauvegardée : {fichier}\n")

        # Pas de Sauvegarde
        if choix == 3 :
            print("Sauvegarde annulée.\n")


    return detection
    # return {
    #     "detection" : detection,
    #     "image": image,
    #     "masque": masque_bestiole,
    #     "fond": fond,
    #     "difference": difference,
    #     "bbox_detectee": bbox_detectee,
    #     "bbox_securisee": bbox_securisee,
    #     "occupation": info_occ,
    # }


def main():
    """
    Boucle principale du programme.
    """

    syst.clear_screen()

    display.titre("Info & Crop d'imaghes", colors["aqua"])

    # Saisie du répertoire contenant les images à traiter
    print()
    repertoire = util.get_directory_color("Répertoire à utiliser")

    liste_images = [
        fichier
        for fichier in repertoire.iterdir()
        if fichier.is_file() and fichier.suffix.lower() in cst.IMAGE_EXT
        ]

    nb_images = len(liste_images)
    
    print()
    if nb_images == 0:
        
        display.print("Aucune image trouvée dans le répertoire.", colors["error"])
        util.sortie_de_programme()

    else :

        display.print(f"Nombre d'images à traiter : {len(liste_images)}", colors['info'])


    voir_image = util.answer_yes_or_no("Voulez-vous voir les images intermédiares", False)
    afficher_informations = util.answer_yes_or_no("Voulez-vous voir les informations de calculs", False)

    pas_de_detection = 0
    images_sans_detection = []
    erreurs_lecture = 0
    images_erreur_lecture = []

    # Début de la boucle de traitement des images
    nb_images = len(liste_images)
    for i, img in enumerate(liste_images, start=1):
 
        try:

            en_tete = f"{i}/{nb_images}"
            display.header_title(img.stem, colors["aqua"], en_tete)

            detec = traiter_image(
                repertoire,             # Répertoire contenant les images
                voir_image,             # Afficher les images intermédiaires
                afficher_informations,  # Afficher les informations de calculs
                chemin_image=  img,     # image à traiter         
                seuil=SEUIL_DETECTION   # Seuil de détection 
            )

            # Pas de détection de bestioles
            if not detec :
                pas_de_detection +=1
                images_sans_detection.append(img.name)

        except ErreurLectureImage  as erreur:
            erreurs_lecture += 1
            images_erreur_lecture.append(img.name)
            afficher_erreur(erreur)

        except Exception as erreur:
            afficher_erreur(erreur)

    nb_ok = nb_images - pas_de_detection - erreurs_lecture


    # ----- Résumé d'exécution ----------------------------------------------------------
   # Status de l'exécution pour affichage
    if erreurs_lecture != 0 :
        status = 'error'
    elif pas_de_detection != 0 :
        status = 'warning'
    else :
        status = 'ok'


    display.titre("Résumé d'exécution", colors[status])
    print()

    # Total d'images traitées sur le total avec status d'exécution
    display.print(f"Nombre d'images traitées       : {nb_ok} / {nb_images}", colors[status])
    


    # Total d'images avec détection
    if nb_ok != nb_images :
        display.print(f"Nombre d'images avec détection : {nb_ok}", colors["ok"])
        display.print(f"Nombre d'images non traitées   : {nb_images - nb_ok}", colors["warning"])

    # Pas de détection de bestioles
    if pas_de_detection != 0 :
        display.print(f"Nombre d'images sans détection : {pas_de_detection}", colors["warning"])
        
        sauvegarder_liste_erreurs(
            repertoire = repertoire,
            fichier = "images_sans_detection.txt",
            liste = images_sans_detection
            )

    # Erreur de lecture du fichier image
    if images_erreur_lecture:
        display.print(f"Nombre d'erreurs de lecture    : {erreurs_lecture}", colors["error"])

        sauvegarder_liste_erreurs(
            repertoire = repertoire,
            fichier = "images_erreur_lecture.txt",
            liste = images_erreur_lecture
            )


    print()
    display.print("Fin du traitement !!!", colors['goodbye'])


if __name__ == "__main__":
    main()
