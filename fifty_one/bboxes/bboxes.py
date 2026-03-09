import os

from fiftyone import ViewField as F

from tools import utility as util
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors


display = util.DisplayColor()

# --- Détection détaillée des bbox problématiques ---
def detect_bbox_problemes_detail_tolere(dataset, bbox_tol=1e-6):
    """
    Détection détaillée des bbox problématiques avec tolérance pour
    les petits négatifs.

    bbox_tol : valeur seuil sous laquelle un négatif est considéré comme tolérable.
    """
    erreurs = {
        "manquantes": [],
        "longueur_incorrecte": [],
        "coord_negatives": [],
        "taille_non_positive": [],
        "hors_limites": []
    }

    # Bbox manquantes
    missing = dataset.filter_labels(
        "ground_truth",
        F("bounding_box") == None
    )
    erreurs["manquantes"] = sorted(missing.values("filepath"))
    util.display_and_save_errors(
        erreurs["manquantes"],
        "images_bbox_manquantes.txt",
        "Images sans bbox"
    )

    # Bbox existantes
    bbox_existantes = dataset.filter_labels(
        "ground_truth",
        F("bounding_box") != None
    )

    # Longueur incorrecte
    longueur_incorrecte = bbox_existantes.filter_labels(
        "ground_truth",
        F("bounding_box").length() != 4
    )
    erreurs["longueur_incorrecte"] = sorted(longueur_incorrecte.values("filepath"))
    util.display_and_save_errors(
        erreurs["longueur_incorrecte"],
        "images_bbox_longueur_incorrecte.txt",
        "Images avec bbox de longueur incorrecte"
    )

    # Coordonnées négatives (avec tolérance)
    coord_neg = bbox_existantes.filter_labels(
        "ground_truth",
        (F("bounding_box")[0] < -bbox_tol) | (F("bounding_box")[1] < -bbox_tol)
    )
    erreurs["coord_negatives"] = sorted(coord_neg.values("filepath"))
    util.display_and_save_errors(
        erreurs["coord_negatives"],
        "images_bbox_coord_negatives.txt",
        "Images avec bbox coordonnées négatives"
    )

    # Taille non positive
    taille_non_positive = bbox_existantes.filter_labels(
        "ground_truth",
        (F("bounding_box")[2] <= 0) | (F("bounding_box")[3] <= 0)
    )
    erreurs["taille_non_positive"] = sorted(taille_non_positive.values("filepath"))
    util.display_and_save_errors(
        erreurs["taille_non_positive"],
        "images_bbox_taille_non_positive.txt",
        "Images avec bbox largeur/hauteur ≤ 0"
    )

    # Hors limites
    threshold = 1 + ct.threshold_bounding_box / 100
    hors_limites = bbox_existantes.filter_labels(
        "ground_truth",
        (F("bounding_box")[0] + F("bounding_box")[2] > threshold) |
        (F("bounding_box")[1] + F("bounding_box")[3] > threshold)
    )
    erreurs["hors_limites"] = sorted(hors_limites.values("filepath"))
    util.display_and_save_errors(
        erreurs["hors_limites"],
        "images_bbox_hors_limites.txt",
        f"Images avec bbox hors limites (> {ct.threshold_bounding_box}% de dépassement) "
    )

    return erreurs


def afficher_bbox_erreurs_compact(bbox_erreurs, noms_par_ligne=5):
    """
    Affiche les erreurs de bbox par catégorie, plusieurs noms d'images par ligne.
    
    bbox_erreurs : dict retourné par detect_bbox_problemes_detail()
    noms_par_ligne : nombre de noms d'images affichés par ligne
    """
    # Calculer largeur max pour aligner les catégories
    categorie_max_len = max(len(cat) for cat in bbox_erreurs.keys())
    
    display.print("Erreurs de bbox détectées :", colors["warning"])
    
    for categorie, chemins in bbox_erreurs.items():
        if not chemins:
            continue  # ignorer les catégories sans erreur
        
        # Titre catégorie + nombre d'images
        print(f"{categorie.capitalize().ljust(categorie_max_len)} ({len(chemins)} images) :")
        
        # Extraire uniquement les noms de fichiers
        noms_images = [os.path.basename(chemin) for chemin in chemins]
        
        # Affichage en blocs de plusieurs noms par ligne
        for i in range(0, len(noms_images), noms_par_ligne):
            ligne = "  ".join(noms_images[i:i+noms_par_ligne])
            print(f"{' ' * (categorie_max_len + 3)}{ligne}")
        
        print()  # ligne vide entre catégories


