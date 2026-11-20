import sys
import numpy as np
from pathlib import Path

import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QImage,
    QPixmap,
)


from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QScrollArea,
    QMessageBox,
)

from config import constants as cst
from tools import utility as util
from ihm.moteur_info_crop import (
    ErreurLectureImage,
    lire_image, 
    creer_masque_bestiole,
    detecter_zone_depuis_masque,
    ajouter_marge_bbox,
    calculer_occupation_depuis_masque,
    calculer_occupation_bbox,
    sauvegarder_bbox,
    sauvegarder_liste_erreurs,

)


# =================================================================================================
# PARAMÈTRES
# ==================================================================================================

SEUIL_DETECTION = None
PAS_SEUIL = 5
TAILLE_BORD = 20
SURFACE_MIN_PCT = 0.001

MARGE_BBOX_PCT = 0.03
MARGE_BBOX_MIN_PX = 5

# ==================================================================================================
# ÉTATS DES IMAGES
# ==================================================================================================

ETAT_A_TRAITER = "a_traiter"

ETAT_SAUVEGARDEE_ROUGE = "sauvegardee_rouge"
ETAT_SAUVEGARDEE_VERTE = "sauvegardee_verte"

ETAT_SANS_SAUVEGARDE = "sans_sauvegarde"
ETAT_SANS_DETECTION = "sans_detection"
ETAT_ERREUR_LECTURE = "erreur_lecture"


SYMBOLES_ETAT = {
    ETAT_A_TRAITER: "○",
    ETAT_SAUVEGARDEE_ROUGE: "✓R",
    ETAT_SAUVEGARDEE_VERTE: "✓V",
    ETAT_SANS_SAUVEGARDE: "─",
    ETAT_SANS_DETECTION: "⚠",
    ETAT_ERREUR_LECTURE: "⚠",
}

LIBELLES_ETAT = {
    ETAT_A_TRAITER: "",
    ETAT_SAUVEGARDEE_ROUGE: "BBox rouge sauvegardée",
    ETAT_SAUVEGARDEE_VERTE: "BBox verte sauvegardée",
    ETAT_SANS_SAUVEGARDE: "Sans sauvegarde",
    ETAT_SANS_DETECTION: "Sans détection",
    ETAT_ERREUR_LECTURE: "Erreur de lecture",
}

# ==================================================================================================
# COULEURS IHM AQUA-IA
# ==================================================================================================

COULEUR_AQUA_LIGHT = "#66FFCC"
COULEUR_AQUA = "#00CC99"
COULEUR_AQUA_DARK = "#006666"

COULEUR_AQUA_TEXTE = "#003333"
COULEUR_AQUA_PRESSED = "#009977"
COULEUR_AQUA_DARK_PRESSED = "#004C4C"

COULEUR_TEXTE = "#404040"
COULEUR_TEXTE_SECONDAIRE = "#505050"

COULEUR_FOND_IMAGE = "#202020"
COULEUR_TEXTE_IMAGE = "#D0D0D0"
COULEUR_BORDURE_IMAGE = "#808080"

COULEUR_BBOX_ROUGE = "#CC0000"
COULEUR_BBOX_VERTE = "#009600"

COULEUR_BOUTON_NEUTRE = "#707070"
COULEUR_BOUTON_NEUTRE_HOVER = "#909090"
COULEUR_BOUTON_NEUTRE_PRESSED = "#505050"

COULEUR_DESACTIVE = "#D0D0D0"
COULEUR_TEXTE_DESACTIVE = "#888888"
COULEUR_BORDURE_DESACTIVE = "#BBBBBB"


# ==================================================================================================
# COULEURS D'ÉTAT
# ==================================================================================================

COULEUR_SUCCES_TEXTE = "#155724"
COULEUR_SUCCES_FOND = "#D4EDDA"
COULEUR_SUCCES_BORDURE = "#C3E6CB"



#==================================================================================================

class InterfaceInfoCrop(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Aqua-IA - Info & Crop d'images"
        )

        self.resize(
            900,
            650
        )


        # ----------------------------------------------------------
        # Image et BBox actuellement affichées
        # ----------------------------------------------------------
        self.image_courante = None
        self.chemin_image_courante = None

        self.bbox_detectee_courante = None
        self.bbox_securisee_courante = None

        self.informations_seuil_courant = None

        # ----------------------------------------------------------
        # Diagnostics de l’image courante
        # ----------------------------------------------------------

        self.masque_courant = None
        self.difference_courante = None
        self.seuil_courant = None
        self.informations_seuil_courant = None

        self.etat_images = {}

        self._creer_interface()

        self.image_rgb_affichage = None


    # ----------------------------------------------------------
    # Création Label Affichage
    # ----------------------------------------------------------
    def _creer_label_affichage(
    self,
    texte,
    largeur_min=600,
    hauteur_min=200
):
        """
        Crée une zone standard d'affichage d'image.
        """

        label = QLabel(texte)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(
            largeur_min,
            hauteur_min
        )

        label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {COULEUR_FOND_IMAGE};
                color: {COULEUR_TEXTE_IMAGE};
                border: 1px solid {COULEUR_BORDURE_IMAGE};
            }}
            """
        )

        return label

    def _afficher_image_gris(
        self,
        image,
        label
    ):
        """
        Affiche une image NumPy en niveaux de gris
        dans un QLabel en conservant ses proportions.

        Parameters
        ----------
        image : np.ndarray
            Image uint8 à un seul canal.

        label : QLabel
            QLabel dans lequel afficher l'image.
        """

        if image is None:
            return

        if image.ndim != 2:

            raise ValueError(
                "L'image transmise à _afficher_image_gris() "
                "doit comporter un seul canal."
            )

        # ----------------------------------------------------------
        # Conversion en uint8 si nécessaire
        # ----------------------------------------------------------

        if image.dtype != np.uint8:

            image = image.astype(np.uint8)

        # ----------------------------------------------------------
        # Dimensions
        # ----------------------------------------------------------

        hauteur, largeur = (image.shape)
        octets_par_ligne = (image.strides[0])

        # ----------------------------------------------------------
        # NumPy -> QImage
        # ----------------------------------------------------------

        image_qt = QImage(
            image.data,
            largeur,
            hauteur,
            octets_par_ligne,
            QImage.Format_Grayscale8
        ).copy()

        # ----------------------------------------------------------
        # QImage -> QPixmap
        # ----------------------------------------------------------

        pixmap = QPixmap.fromImage(image_qt)

        # ----------------------------------------------------------
        # Adaptation au QLabel
        # ----------------------------------------------------------

        pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------

        label.setPixmap(pixmap)

    def _afficher_image_rgb(
        self,
        image,
        label
    ):
        """
        Affiche une image NumPy RGB dans un QLabel
        en conservant ses proportions.

        Parameters
        ----------
        image : np.ndarray
            Image uint8 RGB à trois canaux.

        label : QLabel
            QLabel dans lequel afficher l'image.
        """

        if image is None:
            return

        if image.ndim != 3 or image.shape[2] != 3:

            raise ValueError(
                "L'image transmise à _afficher_image_rgb() "
                "doit comporter trois canaux."
            )

        # ----------------------------------------------------------
        # Conversion en uint8 si nécessaire
        # ----------------------------------------------------------

        if image.dtype != np.uint8:

            image = image.astype(np.uint8)

        # ----------------------------------------------------------
        # Dimensions
        # ----------------------------------------------------------

        hauteur, largeur, _ = (image.shape)
        octets_par_ligne = (image.strides[0])

        # ----------------------------------------------------------
        # NumPy -> QImage
        # ----------------------------------------------------------

        image_qt = QImage(
            image.data,
            largeur,
            hauteur,
            octets_par_ligne,
            QImage.Format_RGB888
        ).copy()

        # ----------------------------------------------------------
        # QImage -> QPixmap
        # ----------------------------------------------------------

        pixmap = QPixmap.fromImage(image_qt)

        # ----------------------------------------------------------
        # Adaptation au QLabel
        # ----------------------------------------------------------

        pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------
        label.setPixmap(pixmap)


    def _creer_entete(
        self,
        layout_principal
    ):
        """
        Crée le titre et le sous-titre de l'interface.
        """

        # ----------------------------------------------------------
        # Titre principal
        # ----------------------------------------------------------
        titre = QLabel("AQUA-IA")
        titre.setAlignment(Qt.AlignCenter)
        titre.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_AQUA_DARK};
                font-size: 24px;
                font-weight: bold;
                padding-top: 5px;
            }}
            """
        )

        layout_principal.addWidget(titre)

        # ----------------------------------------------------------
        # Sous-titre
        # ----------------------------------------------------------
        sous_titre = QLabel("Analyse et recadrage d'images")
        sous_titre.setAlignment(Qt.AlignCenter)
        sous_titre.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE_SECONDAIRE};
                font-size: 14px;
                padding-bottom: 8px;
            }}
            """
        )

        layout_principal.addWidget(sous_titre)

    def _creer_selection_images(
        self,
        layout_principal
    ):
        """
        Crée la sélection du répertoire,
        la sélection de l'image et les boutons de navigation.
        """

        layout_selection = QGridLayout()

        # ==========================================================
        # Répertoire
        # ==========================================================

        label_repertoire = QLabel("Répertoire :")
        self.champ_repertoire = QLineEdit()
        self.champ_repertoire.setReadOnly(True)
        self.champ_repertoire.setPlaceholderText("Sélectionnez le répertoire contenant les images")
        self.bouton_parcourir = QPushButton("Parcourir...")

        # ==========================================================
        # Image
        # ==========================================================

        label_image = QLabel("Image :")
        self.combo_images = QComboBox()
        self.combo_images.setEnabled(False)

        # ==========================================================
        # Position          
        # ==========================================================

        self.label_position_image = QLabel("0 / 0")
        self.label_position_image.setAlignment(Qt.AlignCenter)
        self.label_position_image.setMinimumWidth(70)
        self.label_position_image.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_AQUA_DARK};  
                font-weight: bold;
                padding: 4px;
            }}
            """
        )

        # ==========================================================
        # Navigation
        # ==========================================================

        self.bouton_precedente = QPushButton("◀ Précédente")
        self.bouton_suivante = QPushButton("Suivante ▶")
        self.bouton_precedente.setEnabled(False)
        self.bouton_suivante.setEnabled(False)

        # Style des boutons de navigation
        style_navigation = self._style_bouton(
            couleur_fond=COULEUR_AQUA_DARK,
            couleur_texte="white",
            couleur_bordure=COULEUR_AQUA_DARK_PRESSED,
            couleur_hover=COULEUR_AQUA,
            couleur_texte_hover=COULEUR_AQUA_TEXTE,
            couleur_pressed=COULEUR_AQUA_DARK_PRESSED
        )

        self.bouton_precedente.setStyleSheet(style_navigation)
        self.bouton_suivante.setStyleSheet(style_navigation)

        # ==========================================================
        # Ligne Répertoire
        # ==========================================================

        layout_selection.addWidget(
            label_repertoire,
            0,
            0
        )

        layout_selection.addWidget(
            self.champ_repertoire,
            0,
            1
        )

        layout_selection.addWidget(
            self.bouton_parcourir,
            0,
            2
        )

        # ==========================================================
        # Ligne Image
        # ==========================================================

        layout_navigation = QHBoxLayout()

        layout_navigation.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout_navigation.addWidget(self.bouton_precedente)
        layout_navigation.addWidget(self.combo_images, 1)
        layout_navigation.addWidget(self.label_position_image)
        layout_navigation.addWidget(self.bouton_suivante)
        layout_selection.addWidget(
            label_image,
            1,
            0
        )

        layout_selection.addLayout(
            layout_navigation,
            1,
            1,
            1,
            2
        )

        # ==========================================================
        # Gestion des colonnes
        # ==========================================================

        layout_selection.setColumnStretch(
            0,
            0
        )

        layout_selection.setColumnStretch(
            1,
            1
        )

        layout_selection.setColumnStretch(
            2,
            0
        )

        # ==========================================================
        # IMPORTANT : ajout à l'interface principale
        # ==========================================================

        layout_principal.addLayout(layout_selection)

    def _creer_options_affichage(
        self,
        layout_principal
    ):
        """
        Crée les options d'affichage des diagnostics.
        """

        self.checkbox_diagnostics = QCheckBox("Afficher les diagnostics intermédiaires")
        self.checkbox_diagnostics.setChecked(False)
        self.checkbox_diagnostics.setStyleSheet(
            f"""
            QCheckBox {{
                color: {COULEUR_AQUA_DARK};
                font-weight: bold;
                padding: 4px;
            }}

            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            """
        )

        layout_principal.addWidget(
            self.checkbox_diagnostics,
            alignment=Qt.AlignLeft
        )

    def _creer_resume(
        self,
        layout_principal
    ):
        """
        Crée la zone de résumé du traitement.
        """

        self.resume_traitement = QLabel(
            "Images : 0    |    "
            "✓ Sauvegardées : 0    |    "
            "─ Sans sauvegarde : 0    |    "
            "⚠ Sans détection : 0    |    "
            "⚠ Erreurs lecture : 0    |    "
            "○ Restantes : 0"
        )

        self.resume_traitement.setAlignment(Qt.AlignCenter)

        self.resume_traitement.setStyleSheet(
            """
            QLabel {
                padding: 6px;
                font-weight: bold;
            }
            """
        )

        layout_principal.addWidget(self.resume_traitement)

    def _creer_zone_image(
        self,
        layout_principal
    ):
        """
        Crée la zone d'affichage de l'image principale
        et la légende des bounding boxes.
        """

        style_legende = f"""
            QLabel {{
                color: {COULEUR_TEXTE};
                font-weight: bold;
                padding: 4px;
            }}
        """


        # ----------------------------------------------------------
        # Image principale
        # ----------------------------------------------------------

        self.affichage_image = self._creer_label_affichage(
            texte="Sélectionnez une image",
            largeur_min=600,
            hauteur_min=320
        )

        layout_principal.addWidget(
            self.affichage_image
        )

        # ----------------------------------------------------------
        # Légende des Bounding Boxes
        # ----------------------------------------------------------

        layout_legende = QHBoxLayout()

        legende_rouge = QLabel(
            f'<span style="color:{COULEUR_BBOX_ROUGE}; '
            f'font-size:16px;">■</span>'
            '&nbsp;&nbsp;BBox rouge : zone détectée'
        )

        legende_verte = QLabel(
            f'<span style="color:{COULEUR_BBOX_VERTE}; '
            f'font-size:16px;">■</span>'
            '&nbsp;&nbsp;BBox verte : zone sécurisée avec marge'
        )

        legende_rouge.setStyleSheet(style_legende)
        legende_verte.setStyleSheet(style_legende)

        # ----------------------------------------------------------
        # Placement de la légende
        # ----------------------------------------------------------

        layout_legende.addStretch()
        layout_legende.addWidget(legende_rouge)
        layout_legende.addSpacing(30)
        layout_legende.addWidget(legende_verte)
        layout_legende.addStretch()
        layout_principal.addLayout(layout_legende)

    def _creer_diagnostics(
        self,
        layout_principal
    ):
        """
        Crée la zone des diagnostics intermédiaires :

        - masque ;
        - carte des distances ;
        - histogramme ;
        - pixels critiques ;
        - informations sur le calcul du seuil.
        """

        # ==========================================================
        # Conteneur principal des diagnostics
        # ==========================================================

        self.onglets_diagnostics = QTabWidget()

        self.onglets_diagnostics.setVisible(
            False
        )

        # ==========================================================
        # 1. MASQUE
        # ==========================================================

        self.widget_masque = QWidget()
        layout_masque = QVBoxLayout(self.widget_masque)
        layout_masque.setContentsMargins(
            5,
            5,
            5,
            5
        )

        aide_masque = self._creer_aide_diagnostic(
            "Blanc : composante retenue comme organisme. "
            "Noir : fond ou pixels rejetés."
        )

        layout_masque.addWidget(aide_masque)

        self.affichage_masque = self._creer_label_affichage(texte="Aucun masque disponible")
        layout_masque.addWidget(self.affichage_masque)
        self.onglets_diagnostics.addTab(self.widget_masque, "Masque")

        # ==========================================================
        # 2. DISTANCES
        # ==========================================================

        self.widget_distances = QWidget()
        layout_distances = QVBoxLayout(self.widget_distances)
        layout_distances.setContentsMargins(
            5,
            5,
            5,
            5
        )

        aide_distances = self._creer_aide_diagnostic(
            "La couleur représente la distance à la couleur du fond : "
            "les zones les plus chaudes correspondent aux pixels "
            "les plus différents du fond estimé."
        )

        layout_distances.addWidget(aide_distances)
        self.affichage_distances = self._creer_label_affichage(texte="Aucune carte des distances disponible")
        layout_distances.addWidget(self.affichage_distances)
        self.onglets_diagnostics.addTab(self.widget_distances, "Distances")

        # ==========================================================
        # 3. HISTOGRAMME
        # ==========================================================

        self.widget_histogramme = QWidget()
        layout_histogramme = QVBoxLayout(self.widget_histogramme)

        layout_histogramme.setContentsMargins(
            5,
            5,
            5,
            5
        )

        self.figure_histogramme = Figure(figsize=(8, 4))
        aide_histogramme = self._creer_aide_diagnostic(
            "Distribution des distances au fond. "
            "La ligne rouge indique le seuil retenu ; "
            "P90, P95 et P99 indiquent les percentiles de la distribution."
        )

        layout_histogramme.addWidget(aide_histogramme)
        self.canvas_histogramme = FigureCanvas(self.figure_histogramme)
        self.axes_histogramme = (self.figure_histogramme.add_subplot(111))
        layout_histogramme.addWidget(self.canvas_histogramme)
        self.onglets_diagnostics.addTab(self.widget_histogramme,"Histogramme")

        # ==========================================================
        # 4. PIXELS CRITIQUES
        # ==========================================================

        self.widget_pixels_critiques = QWidget()
        layout_pixels_critiques = QVBoxLayout(self.widget_pixels_critiques)

        layout_pixels_critiques.setContentsMargins(
            5,
            5,
            5,
            5
        )

        # ----------------------------------------------------------
        # Informations numériques
        # ----------------------------------------------------------

        self.informations_pixels_critiques = QLabel(
            "Seuil actuel : -    |    "
            "Seuil testé : -    |    "
            "Pixels critiques : -    |    "
            "Pourcentage : -"
        )

        self.informations_pixels_critiques.setAlignment(Qt.AlignCenter)
        self.informations_pixels_critiques.setStyleSheet(
            self._style_label_information(
                couleur=COULEUR_AQUA_DARK
            )
        )

        aide_pixels_critiques = self._creer_aide_diagnostic(
            "Blanc : pixels actuellement détectés qui seraient perdus "
            f"si le seuil augmentait de {PAS_SEUIL}."
        )

        layout_pixels_critiques.addWidget(aide_pixels_critiques)
        layout_pixels_critiques.addWidget(self.informations_pixels_critiques)

        # ----------------------------------------------------------
        # Image des pixels critiques
        # ----------------------------------------------------------

        self.affichage_pixels_critiques = (
            self._creer_label_affichage(
                texte="Aucun pixel critique disponible"
            )
        )

        layout_pixels_critiques.addWidget(self.affichage_pixels_critiques)
        self.onglets_diagnostics.addTab(self.widget_pixels_critiques, "Pixels critiques")

        # ==========================================================
        # 5. CALCUL DU SEUIL
        # ==========================================================

        self.widget_calcul_seuil = QWidget()
        layout_calcul_seuil = QGridLayout(self.widget_calcul_seuil)

        aide_calcul_seuil = self._creer_aide_diagnostic(
            "Détail du calcul automatique du seuil à partir "
            "des variations observées dans la zone de fond."
        )

        layout_calcul_seuil.addWidget(
            aide_calcul_seuil,
            0,
            0,
            1,
            2
        )


        layout_calcul_seuil.setContentsMargins(
            30,
            20,
            30,
            20
        )

        layout_calcul_seuil.setHorizontalSpacing(8)
        layout_calcul_seuil.setVerticalSpacing(8)

        # ----------------------------------------------------------
        # Champs affichés
        # ----------------------------------------------------------

        self.valeurs_calcul_seuil = {}

        libelles = [
            ("mediane_fond", "Médiane du fond"),
            ("mad_fond", "MAD du fond"),
            ("sigma_robuste_fond", "Sigma robuste"),
            ("seuil_mad", "Seuil MAD"),
            ("seuil_percentile", "Seuil percentile"),
            (
                "rapport_percentile_mad",
                "Rapport percentile / MAD"
            ),
            (
                "percentile_suspect",
                "Percentile suspect"
            ),
            (
                "methode_seuil",
                "Méthode retenue"
            ),
            (
                "percentile_fond",
                "Percentile utilisé"
            ),
            (
                "taille_bord",
                "Taille du bord"
            ),
            (
                "nombre_pixels_fond",
                "Pixels de fond analysés"
            ),
            (
                "seuil_final",
                "Seuil automatique final"
            ),
        ]

        # ----------------------------------------------------------
        # Création des lignes
        # ----------------------------------------------------------

        for ligne, (cle, texte) in enumerate(
            libelles,
            start=1
        ):

            label = QLabel(
                f"{texte} :"
            )

            label.setAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            label.setStyleSheet(
                f"""
                QLabel {{
                    color: {COULEUR_AQUA_DARK};
                    font-weight: bold;
                }}
                """
            )

            valeur = QLabel(
                "-"
            )

            valeur.setAlignment(
                Qt.AlignLeft | Qt.AlignVCenter
            )

            self.valeurs_calcul_seuil[
                cle
            ] = valeur

            layout_calcul_seuil.addWidget(
                label,
                ligne,
                0
            )

            layout_calcul_seuil.addWidget(
                valeur,
                ligne,
                1
            )

        # ----------------------------------------------------------
        # Organisation des colonnes
        # ----------------------------------------------------------

        layout_calcul_seuil.setColumnStretch(
            0,
            0
        )

        layout_calcul_seuil.setColumnStretch(
            1,
            1
        )

        layout_calcul_seuil.setRowStretch(
            len(libelles),
            1
        )

        self.onglets_diagnostics.addTab(
            self.widget_calcul_seuil,
            "Calcul du seuil"
        )

        # ==========================================================
        # Ajout dans l'interface principale
        # ==========================================================

        layout_principal.addWidget(
            self.onglets_diagnostics
        )

    def _creer_informations(
        self,
        layout_principal
    ):
        """
        Crée la grille présentant les informations
        de détection et d'occupation.
        """

        layout_informations = QGridLayout()

        layout_informations.setContentsMargins(
            10,
            5,
            10,
            5
        )

        layout_informations.setHorizontalSpacing(12)
        layout_informations.setVerticalSpacing(5)

        # ==========================================================
        # Colonne gauche : Détection
        # ==========================================================

        label_detection = QLabel("Détection :")
        label_seuil = QLabel("Seuil :")
        label_fond = QLabel("Fond BGR :")

        # ==========================================================
        # Colonne droite : Occupation
        # ==========================================================

        label_surface_image = QLabel("Surface image :")
        label_surface_bestiole = QLabel("Surface bestiole :")
        label_occupation_reelle = QLabel("Occupation réelle :")
        label_occupation_bbox = QLabel("Occupation BBox :")

        # ==========================================================
        # Valeurs
        # ==========================================================

        self.valeur_detection = QLabel("-")
        self.valeur_seuil = QLabel("-")
        self.valeur_fond = QLabel("-")
        self.valeur_surface_image = QLabel("-")
        self.valeur_surface_bestiole = QLabel("-")
        self.valeur_occupation_reelle = QLabel("-")
        self.valeur_occupation_bbox = QLabel("-")

        # ==========================================================
        # Style des libellés
        # ==========================================================

        labels = (
            label_detection,
            label_seuil,
            label_fond,
            label_surface_image,
            label_surface_bestiole,
            label_occupation_reelle,
            label_occupation_bbox,
        )

        for label in labels:

            label.setAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            label.setStyleSheet(
                f"""
                QLabel {{
                    color: {COULEUR_AQUA_DARK};
                    font-weight: bold;
                }}
                """
            )


        valeurs = (
            self.valeur_detection,
            self.valeur_seuil,
            self.valeur_fond,
            self.valeur_surface_image,
            self.valeur_surface_bestiole,
            self.valeur_occupation_reelle,
            self.valeur_occupation_bbox,
        )

        for valeur in valeurs:

            valeur.setAlignment(
                Qt.AlignLeft | Qt.AlignVCenter
            )


        # ==========================================================
        # Partie Détection
        # ==========================================================

        layout_informations.addWidget(
            label_detection,
            0,
            0
        )

        layout_informations.addWidget(
            self.valeur_detection,
            0,
            1
        )

        layout_informations.addWidget(
            label_seuil,
            1,
            0
        )

        layout_informations.addWidget(
            self.valeur_seuil,
            1,
            1
        )

        layout_informations.addWidget(
            label_fond,
            2,
            0
        )

        layout_informations.addWidget(
            self.valeur_fond,
            2,
            1
        )

        # ==========================================================
        # Partie Occupation
        # ==========================================================

        layout_informations.addWidget(
            label_surface_image,
            0,
            2
        )

        layout_informations.addWidget(
            self.valeur_surface_image,
            0,
            3
        )

        layout_informations.addWidget(
            label_surface_bestiole,
            1,
            2
        )

        layout_informations.addWidget(
            self.valeur_surface_bestiole,
            1,
            3
        )

        layout_informations.addWidget(
            label_occupation_reelle,
            2,
            2
        )

        layout_informations.addWidget(
            self.valeur_occupation_reelle,
            2,
            3
        )

        layout_informations.addWidget(
            label_occupation_bbox,
            3,
            2
        )

        layout_informations.addWidget(
            self.valeur_occupation_bbox,
            3,
            3
        )

        layout_informations.setColumnStretch(
            0,
            0
        )

        layout_informations.setColumnStretch(
            1,
            1
        )

        layout_informations.setColumnStretch(
            2,
            0
        )

        layout_informations.setColumnStretch(
            3,
            1
        )

        layout_principal.addLayout(
            layout_informations
        )

    def _creer_zone_sauvegarde(
        self,
        layout_principal
    ):
        """
        Crée les boutons permettant à l'opérateur
        de choisir l'action à effectuer sur l'image.
        """

        layout_sauvegarde = QHBoxLayout()

        # ----------------------------------------------------------
        # État de la décision opérateur
        # ----------------------------------------------------------

        self.label_decision = QLabel("Sélectionnez une image")
        self.label_decision.setAlignment(Qt.AlignCenter)
        self.label_decision.setStyleSheet(self._style_label_information(couleur=COULEUR_AQUA_DARK))

        layout_principal.addWidget(self.label_decision)


        # ==========================================================
        # BBox rouge
        # ==========================================================
        self.bouton_bbox_rouge = QPushButton("Sauvegarder BBox rouge")

        self.bouton_bbox_rouge.setStyleSheet(
            self._style_bouton(
                couleur_fond=COULEUR_AQUA,
                couleur_texte=COULEUR_AQUA_TEXTE,
                couleur_bordure=COULEUR_AQUA_PRESSED,
                couleur_hover=COULEUR_AQUA_LIGHT,
                couleur_texte_hover=COULEUR_AQUA_TEXTE,
                couleur_pressed=COULEUR_AQUA_PRESSED
            )
        )


        # ==========================================================
        # BBox verte
        # ==========================================================
        self.bouton_bbox_verte = QPushButton("Sauvegarder BBox verte")

        self.bouton_bbox_verte.setStyleSheet(
            self._style_bouton(
                couleur_fond=COULEUR_AQUA_DARK,
                couleur_texte="white",
                couleur_bordure=COULEUR_AQUA_DARK_PRESSED,
                couleur_hover=COULEUR_AQUA,
                couleur_texte_hover=COULEUR_AQUA_TEXTE,
                couleur_pressed=COULEUR_AQUA_DARK_PRESSED
            )
        )

        # ==========================================================
        # Sans sauvegarde
        # ==========================================================
        self.bouton_sans_sauvegarde = QPushButton("Ne pas sauvegarder")

        self.bouton_sans_sauvegarde.setStyleSheet(
                self._style_bouton(
                    couleur_fond=COULEUR_BOUTON_NEUTRE,
                    couleur_texte="white",
                    couleur_bordure=COULEUR_BOUTON_NEUTRE_PRESSED,
                    couleur_hover=COULEUR_BOUTON_NEUTRE_HOVER,
                    couleur_texte_hover="white",
                    couleur_pressed=COULEUR_BOUTON_NEUTRE_PRESSED
                )
            )

        # ==========================================================
        # État initial
        # ==========================================================

        self.bouton_bbox_rouge.setEnabled(False)
        self.bouton_bbox_verte.setEnabled(False)
        self.bouton_sans_sauvegarde.setEnabled(False)

        # ==========================================================
        # Placement
        # ==========================================================

        layout_sauvegarde.addStretch()

        layout_sauvegarde.addWidget(self.bouton_bbox_rouge)
        layout_sauvegarde.addWidget(self.bouton_bbox_verte)
        layout_sauvegarde.addWidget(self.bouton_sans_sauvegarde)

        layout_sauvegarde.addStretch()
        layout_principal.addLayout(layout_sauvegarde)


        # ----------------------------------------------------------
        # Bouton Quitter
        # ----------------------------------------------------------

        self.bouton_quitter = QPushButton("Quitter")

        self.bouton_quitter.setStyleSheet(
            self._style_bouton(
                couleur_fond=COULEUR_BOUTON_NEUTRE,
                couleur_texte="white",
                couleur_bordure=COULEUR_BOUTON_NEUTRE_PRESSED,
                couleur_hover=COULEUR_BOUTON_NEUTRE_HOVER,
                couleur_texte_hover="white",
                couleur_pressed=COULEUR_BOUTON_NEUTRE_PRESSED
            )
        )

        layout_quitter = QHBoxLayout()

        layout_quitter.addStretch()

        layout_quitter.addWidget(
            self.bouton_quitter
        )

        layout_principal.addLayout(
            layout_quitter
        )

    def _connecter_signaux(self):
        """
        Connecte les signaux de l'interface
        aux méthodes correspondantes.
        """

        # ----------------------------------------------------------
        # Sélection du répertoire
        # ----------------------------------------------------------
        self.bouton_parcourir.clicked.connect(self.selectionner_repertoire)

        # ----------------------------------------------------------
        # Sélection de l'image
        # ----------------------------------------------------------
        self.combo_images.currentIndexChanged.connect(self.selectionner_image)

        # ----------------------------------------------------------
        # Navigation
        # ----------------------------------------------------------
        self.bouton_precedente.clicked.connect(self.image_precedente)
        self.bouton_suivante.clicked.connect(self.image_navigation_suivante)

        # ----------------------------------------------------------
        # Diagnostics intermédiaires
        # ----------------------------------------------------------
        self.checkbox_diagnostics.toggled.connect(self.afficher_diagnostics)

        # ----------------------------------------------------------
        # Décisions opérateur
        # ----------------------------------------------------------
        self.bouton_bbox_rouge.clicked.connect(self.sauvegarder_bbox_rouge)
        self.bouton_bbox_verte.clicked.connect(self.sauvegarder_bbox_verte)
        self.bouton_sans_sauvegarde.clicked.connect(self.ne_pas_sauvegarder)
        self.bouton_quitter.clicked.connect(self.close)

    def _style_bouton(
        self,
        couleur_fond,
        couleur_texte,
        couleur_bordure,
        couleur_hover,
        couleur_texte_hover,
        couleur_pressed,
        couleur_texte_pressed="white"
    ):
        """
        Génère le style CSS standard d'un bouton Aqua-IA.
        """

        return f"""
            QPushButton {{
                background-color: {couleur_fond};
                color: {couleur_texte};
                border: 1px solid {couleur_bordure};
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: {couleur_hover};
                color: {couleur_texte_hover};
            }}

            QPushButton:pressed {{
                background-color: {couleur_pressed};
                color: {couleur_texte_pressed};
            }}

            QPushButton:disabled {{
                background-color: {COULEUR_DESACTIVE};
                color: {COULEUR_TEXTE_DESACTIVE};
                border-color: {COULEUR_BORDURE_DESACTIVE};
            }}
        """

    def _style_label_information(
        self,
        couleur=None
    ):
        """
        Retourne le style standard utilisé pour
        les labels d'informations.
        """

        style_couleur = (
            f"color: {couleur};"
            if couleur is not None
            else ""
        )

        return f"""
            QLabel {{
                {style_couleur}
                padding: 6px;
                font-weight: bold;
            }}
        """

    def _mettre_a_jour_position_image(self):
        """
        Met à jour la position de l'image actuellement
        sélectionnée dans le QComboBox.
        """

        total = self.combo_images.count()

        index = self.combo_images.currentIndex()

        # ----------------------------------------------------------
        # Aucune image
        # ----------------------------------------------------------

        if (
            total == 0
            or index < 0
        ):

            self.label_position_image.setText("0 / 0")
            self.bouton_precedente.setEnabled(False)
            self.bouton_suivante.setEnabled(False)

            return


        # QComboBox commence à 0, affichage opérateur à 1
        position = index + 1

        self.label_position_image.setText(
            f"{position} / {total}"
        )

        # ----------------------------------------------------------
        # État de la navigation
        # ----------------------------------------------------------

        self.bouton_precedente.setEnabled(index > 0)
        self.bouton_suivante.setEnabled(index < total - 1)

    def image_precedente(self):
        """
        Sélectionne l'image précédente dans le QComboBox,
        sans modifier son état.
        """

        index = self.combo_images.currentIndex()

        if index <= 0:
            return

        self.combo_images.setCurrentIndex(
            index - 1
        )

    def image_navigation_suivante(self):
        """
        Sélectionne l'image suivante dans le QComboBox,
        sans modifier son état.
        """

        index = self.combo_images.currentIndex()

        total = self.combo_images.count()

        if index < 0:
            return

        if index >= total - 1:
            return

        self.combo_images.setCurrentIndex(
            index + 1
        )

    def _mettre_a_jour_decision(
        self,
        chemin_image
    ):
        """
        Met à jour le texte décrivant l'état opérateur
        de l'image courante.
        """

        if chemin_image is None:

            self.label_decision.setText(
                "Sélectionnez une image"
            )

            return

        chemin_image = Path(
            chemin_image
        )

        etat = self.etat_images.get(
            chemin_image,
            ETAT_A_TRAITER
        )

        messages = {
            ETAT_A_TRAITER:
                "Décision opérateur requise",

            ETAT_SAUVEGARDEE_ROUGE:
                "✓ BBox rouge sauvegardée",

            ETAT_SAUVEGARDEE_VERTE:
                "✓ BBox verte sauvegardée",

            ETAT_SANS_SAUVEGARDE:
                "─ Image examinée sans sauvegarde",

            ETAT_SANS_DETECTION:
                "⚠ Aucune détection",

            ETAT_ERREUR_LECTURE:
                "⚠ Erreur de lecture",
        }

        self.label_decision.setText(
            messages.get(
                etat,
                "État inconnu"
            )
        )






#////////////////////////////////////////////////////////////////////
    def _reinitialiser_image_courante(self):
        """
        Réinitialise l'état de l'IHM avant le traitement
        d'une nouvelle image.
        """

        # ----------------------------------------------------------
        # Image courante
        # ----------------------------------------------------------

        self.image_courante = None
        self.chemin_image_courante = None
        self.image_rgb_affichage = None

        # ----------------------------------------------------------
        # Bounding boxes
        # ----------------------------------------------------------

        self.bbox_detectee_courante = None
        self.bbox_securisee_courante = None

        # ----------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------

        self.masque_courant = None
        self.difference_courante = None
        self.seuil_courant = None
        self.informations_seuil_courant = None

        # ----------------------------------------------------------
        # Boutons
        # ----------------------------------------------------------

        self.bouton_bbox_rouge.setEnabled(
            False
        )

        self.bouton_bbox_verte.setEnabled(
            False
        )

        self.bouton_sans_sauvegarde.setEnabled(
            False
        )

        # ----------------------------------------------------------
        # Masque / distances / pixels critiques
        # ----------------------------------------------------------

        self.affichage_masque.clear()
        self.affichage_distances.clear()
        self.affichage_pixels_critiques.clear()

        # ----------------------------------------------------------
        # Informations pixels critiques
        # ----------------------------------------------------------

        self.informations_pixels_critiques.setText(
            "Seuil actuel : -    |    "
            "Seuil testé : -    |    "
            "Pixels critiques : -    |    "
            "Pourcentage : -"
        )

        # ----------------------------------------------------------
        # Histogramme
        # ----------------------------------------------------------

        self.axes_histogramme.clear()

        self.axes_histogramme.text(
            0.5,
            0.5,
            "Aucun histogramme disponible",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.axes_histogramme.transAxes
        )

        self.axes_histogramme.set_axis_off()

        self.canvas_histogramme.draw()

        # ----------------------------------------------------------
        # Calcul du seuil
        # ----------------------------------------------------------

        for valeur in self.valeurs_calcul_seuil.values():

            valeur.setText(
                "-"
            )

        self._reinitialiser_informations()


    def _gerer_erreur_lecture(
        self,
        chemin_image,
        erreur
    ):
        """
        Gère une erreur de lecture ou de décodage
        de l'image sélectionnée.
        """

        chemin_image = Path(
            chemin_image
        )

        # ----------------------------------------------------------
        # Mémorisation du chemin
        # ----------------------------------------------------------

        # On conserve le chemin même si l'image n'a pas pu
        # être décodée, notamment pour le suivi opérateur.
        self.chemin_image_courante = chemin_image

        # ----------------------------------------------------------
        # Image principale
        # ----------------------------------------------------------

        self.affichage_image.clear()

        self.affichage_image.setText(
            str(erreur)
        )

        # ----------------------------------------------------------
        # État de l'image
        # ----------------------------------------------------------

        self.definir_etat_image(
            chemin_image,
            ETAT_ERREUR_LECTURE
        )

        # ----------------------------------------------------------
        # Boutons opérateur
        # ----------------------------------------------------------

        self.bouton_bbox_rouge.setEnabled(
            False
        )

        self.bouton_bbox_verte.setEnabled(
            False
        )

        # L'opérateur doit pouvoir confirmer qu'il passe
        # cette image et continuer vers la suivante.
        self.bouton_sans_sauvegarde.setEnabled(
            True
        )

        # ----------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------

        if self.checkbox_diagnostics.isChecked():

            self.affichage_masque.setText(
                "Aucun masque disponible\n"
                "Erreur de lecture de l'image"
            )

            self.affichage_distances.setText(
                "Aucune carte des distances disponible\n"
                "Erreur de lecture de l'image"
            )

            self.mettre_a_jour_histogramme()
            self.mettre_a_jour_pixels_critiques()
            self.mettre_a_jour_calcul_seuil()

        # ----------------------------------------------------------
        # Informations générales
        # ----------------------------------------------------------

        self._reinitialiser_informations()

        # ----------------------------------------------------------
        # Console
        # ----------------------------------------------------------

        print(
            f'Erreur de lecture : "{chemin_image.name}"'
        )

        print(
            f"    {erreur}"
        )


    def _reinitialiser_informations(self):
        """
        Remet à zéro les informations de détection
        et d'occupation.
        """

        self.valeur_detection.setText("-")
        self.valeur_seuil.setText("-")
        self.valeur_fond.setText("-")
        self.valeur_surface_image.setText("-")
        self.valeur_surface_bestiole.setText("-")
        self.valeur_occupation_reelle.setText("-")
        self.valeur_occupation_bbox.setText("-")



    def _calculer_detection(
        self,
        image,
        chemin_image
    ):
        """
        Calcule la détection de l'organisme pour l'image courante.

        Returns
        -------
        tuple
            masque_bestiole,
            fond,
            seuil_utilise,
            bbox_detectee,
            bbox_securisee
        """

        # ----------------------------------------------------------
        # Création du masque
        # ----------------------------------------------------------

        (
            masque_bestiole,
            fond,
            difference,
            seuil_utilise,
            informations_seuil
        ) = creer_masque_bestiole(
            voir_image=False,
            rep=chemin_image.parent,
            image=image,
            seuil=SEUIL_DETECTION,
            taille_bord=TAILLE_BORD,
            surface_min_pct=SURFACE_MIN_PCT,
            afficher_informations=False
        )

        # ----------------------------------------------------------
        # Mémorisation des diagnostics
        # ----------------------------------------------------------

        self.masque_courant = masque_bestiole
        self.difference_courante = difference
        self.seuil_courant = seuil_utilise

        self.informations_seuil_courant = (
            informations_seuil
        )

        # ----------------------------------------------------------
        # Bounding box détectée
        # ----------------------------------------------------------

        bbox_detectee = detecter_zone_depuis_masque(
            masque_bestiole
        )

        # ----------------------------------------------------------
        # Bounding box sécurisée
        # ----------------------------------------------------------

        bbox_securisee = ajouter_marge_bbox(
            image=image,
            bbox=bbox_detectee,
            marge_pct=MARGE_BBOX_PCT,
            marge_min_px=MARGE_BBOX_MIN_PX
        )

        # ----------------------------------------------------------
        # Actualisation des diagnostics
        # ----------------------------------------------------------

        if self.checkbox_diagnostics.isChecked():

            self.mettre_a_jour_masque()
            self.mettre_a_jour_distances()
            self.mettre_a_jour_histogramme()
            self.mettre_a_jour_pixels_critiques()
            self.mettre_a_jour_calcul_seuil()

        # ----------------------------------------------------------
        # Résultat
        # ----------------------------------------------------------

        return (
            masque_bestiole,
            fond,
            seuil_utilise,
            bbox_detectee,
            bbox_securisee
        )

    def _gerer_etat_detection(
        self,
        image,
        chemin_image,
        bbox_detectee,
        bbox_securisee
    ):
        """
        Mémorise le résultat de la détection et configure
        l'état des boutons selon qu'une BBox a été détectée.
        """

        chemin_image = Path(
            chemin_image
        )

        # ----------------------------------------------------------
        # Mémorisation du résultat courant
        # ----------------------------------------------------------

        self.image_courante = image

        self.chemin_image_courante = (
            chemin_image
        )

        self.bbox_detectee_courante = (
            bbox_detectee
        )

        self.bbox_securisee_courante = (
            bbox_securisee
        )

        # ----------------------------------------------------------
        # Détection réussie
        # ----------------------------------------------------------

        if bbox_detectee is not None:

            self.bouton_bbox_rouge.setEnabled(
                True
            )

            self.bouton_bbox_verte.setEnabled(
                True
            )

            self.bouton_sans_sauvegarde.setEnabled(
                True
            )

            return

        # ----------------------------------------------------------
        # Aucune détection
        # ----------------------------------------------------------

        self.bouton_bbox_rouge.setEnabled(
            False
        )

        self.bouton_bbox_verte.setEnabled(
            False
        )

        self.bouton_sans_sauvegarde.setEnabled(
            True
        )

        self.definir_etat_image(
            chemin_image,
            ETAT_SANS_DETECTION
        )

        print(
            f'Aucune bestiole détectée : '
            f'"{chemin_image.name}"'
        )


    def _afficher_informations_image(
    self,
    bbox_detectee,
    fond,
    seuil_utilise,
    surface_image,
    surface_bestiole,
    occupation_pct,
    occupation_bbox_pct
):
        """
        Met à jour les informations de détection
        et d'occupation de l'image courante.
        """

        # ----------------------------------------------------------
        # Détection
        # ----------------------------------------------------------

        detection = (
            "Oui"
            if bbox_detectee is not None
            else "NON"
        )

        self.valeur_detection.setText(
            detection
        )

        self.valeur_seuil.setText(
            f"{seuil_utilise:.3f}"
        )

        self.valeur_fond.setText(
            str(fond)
        )

        # ----------------------------------------------------------
        # Surfaces
        # ----------------------------------------------------------

        self.valeur_surface_image.setText(
            f"{util.format_nombre(surface_image)} px"
        )

        self.valeur_surface_bestiole.setText(
            f"{util.format_nombre(surface_bestiole)} px"
        )

        # ----------------------------------------------------------
        # Occupation
        # ----------------------------------------------------------

        self.valeur_occupation_reelle.setText(
            f"{occupation_pct:.1f} %"
        )

        self.valeur_occupation_bbox.setText(
            f"{occupation_bbox_pct:.1f} %"
        )




    def _calculer_occupation(
        self,
        image,
        masque_bestiole,
        bbox_securisee
    ):
        """
        Calcule les informations d'occupation de l'organisme
        et de la BBox sécurisée.

        Returns
        -------
        tuple
            surface_image,
            surface_bestiole,
            occupation_pct,
            occupation_bbox_pct
        """

        (
            surface_image,
            surface_bestiole,
            occupation_pct
        ) = calculer_occupation_depuis_masque(
            masque_bestiole
        )

        occupation_bbox_pct = calculer_occupation_bbox(
            image=image,
            bbox=bbox_securisee
        )

        return (
            surface_image,
            surface_bestiole,
            occupation_pct,
            occupation_bbox_pct
        )


    def _afficher_resultat_image(
        self,
        image,
        bbox_detectee,
        bbox_securisee
    ):
        """
        Affiche l'image analysée avec :

        - la BBox détectée en rouge ;
        - la BBox sécurisée en vert.

        Parameters
        ----------
        image : np.ndarray
            Image OpenCV originale au format BGR.

        bbox_detectee : tuple | None
            Bounding box détectée :
            (x_min, y_min, x_max, y_max).

        bbox_securisee : tuple | None
            Bounding box avec marge :
            (x_min, y_min, x_max, y_max).
        """

        # ----------------------------------------------------------
        # Copie de l'image originale
        # ----------------------------------------------------------

        image_affichage = image.copy()

        # ----------------------------------------------------------
        # BBox détectée : rouge
        # ----------------------------------------------------------

        if bbox_detectee is not None:

            (
                x_min,
                y_min,
                x_max,
                y_max
            ) = bbox_detectee

            cv2.rectangle(
                image_affichage,
                (x_min, y_min),
                (x_max, y_max),
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------------
        # BBox sécurisée : verte
        # ----------------------------------------------------------

        if bbox_securisee is not None:

            (
                x_min,
                y_min,
                x_max,
                y_max
            ) = bbox_securisee

            cv2.rectangle(
                image_affichage,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------------
        # OpenCV BGR -> RGB
        # ----------------------------------------------------------

        image_rgb = cv2.cvtColor(
            image_affichage,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------------
        # Mémorisation pour les futurs redimensionnements
        # ----------------------------------------------------------

        self.image_rgb_affichage = (
            image_rgb
        )

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------

        self._afficher_image_rgb(
            image=self.image_rgb_affichage,
            label=self.affichage_image
        )


#////////////////////////////////////////////////////////////////////


    #============================================================================================
    def _creer_interface(self):
        """
        Construit l'interface graphique principale.
        """

        # ==========================================================
        # Widget central
        # ==========================================================

        widget_central = QWidget()

        self.setCentralWidget(widget_central)
        layout_central = QVBoxLayout(widget_central)
        layout_central.setContentsMargins(
            0,
            0,
            0,
            0
        )

        # ==========================================================
        # Zone défilable
        # ==========================================================

        self.zone_defilable = QScrollArea()
        self.zone_defilable.setWidgetResizable(True)
        self.zone_defilable.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.zone_defilable.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.zone_defilable.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout_central.addWidget(self.zone_defilable)

        # ==========================================================
        # Contenu de l'IHM
        # ==========================================================

        widget_principal = QWidget()

        widget_principal.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred
        )

        self.zone_defilable.setWidget(widget_principal)
        layout_principal = QVBoxLayout(widget_principal)
        layout_principal.setAlignment(Qt.AlignTop)

        # ==========================================================
        # Construction
        # ==========================================================

        self._creer_entete(layout_principal)
        self._creer_selection_images(layout_principal)
        self._creer_options_affichage(layout_principal)
        self._creer_resume(layout_principal)
        self._creer_zone_image(layout_principal)
        self._creer_informations(layout_principal)
        self._creer_zone_sauvegarde(layout_principal)
        self._creer_diagnostics(layout_principal)
        

        # ==========================================================
        # Signaux
        # ==========================================================
        self._connecter_signaux()


    def selectionner_repertoire(self):

        repertoire = QFileDialog.getExistingDirectory(
            self,
            "Sélectionner le répertoire contenant les images"
        )

        if not repertoire:
            return

        # Conversion en Path
        repertoire = Path(repertoire)

        # Affichage du répertoire
        self.champ_repertoire.setText(str(repertoire))

        # Recherche des images
        liste_images = sorted(
            fichier
            for fichier in repertoire.iterdir()
            if (
                fichier.is_file()
                and fichier.suffix.lower() in cst.IMAGE_EXT
            )
        )

        # Nettoyage de l'ancienne liste
        self.combo_images.clear()
        self.label_position_image.setText("0 / 0")


        # ----------------------------------------------------------
        # Aucune image
        # ----------------------------------------------------------
        if not liste_images:

            self.combo_images.setEnabled(False)
            self.mettre_a_jour_resume()
            self.label_position_image.setText("0 / 0")

            return


        # ----------------------------------------------------------
        # Remplissage du QComboBox
        # ----------------------------------------------------------

        self.etat_images.clear()

        # Évite le déclenchement de currentIndexChanged
        # pendant l'ajout progressif des images.
        self.combo_images.blockSignals(True)

        for image in liste_images:

            self.etat_images[
                image
            ] = ETAT_A_TRAITER

            self.combo_images.addItem(
                f"{SYMBOLES_ETAT[ETAT_A_TRAITER]}  {image.name}",
                image
            )

        # Première image sélectionnée
        self.combo_images.setCurrentIndex(0)

        # Réactivation des signaux
        self.combo_images.blockSignals(False)
        self.combo_images.setEnabled(True)

        # ----------------------------------------------------------
        # Mise à jour initiale
        # ----------------------------------------------------------

        self._mettre_a_jour_position_image()
        self._mettre_a_jour_decision(None)
        self.mettre_a_jour_resume()

        # Lance explicitement l'affichage de la première image
        self.selectionner_image(0)

    def selectionner_image(
        self,
        index
    ):

        if index < 0:

            self._mettre_a_jour_position_image()
            self._mettre_a_jour_decision(None)

            return

        chemin_image = self.combo_images.itemData(index)

        if chemin_image is None:

            self._mettre_a_jour_position_image()
            self._mettre_a_jour_decision(None)

            return

        # ----------------------------------------------------------
        # Position dans la liste
        # ----------------------------------------------------------

        self._mettre_a_jour_position_image()

        # ----------------------------------------------------------
        # Analyse / affichage de l'image
        # ----------------------------------------------------------

        print(
            f"Image sélectionnée : {chemin_image}"
        )

        self.afficher_image(
            chemin_image
        )

        # ----------------------------------------------------------
        # État opérateur APRÈS analyse
        # ----------------------------------------------------------

        self._mettre_a_jour_decision(
            chemin_image
        )

    def afficher_image(
        self,
        chemin_image
    ):

        # ----------------------------------------------------------
        # Réinitialisation de l'image courante
        # ----------------------------------------------------------
        self._reinitialiser_image_courante()
        

        # ----------------------------------------------------------
        # Lecture de l'image
        # ----------------------------------------------------------
        try:

            image = lire_image(chemin_image)

        except ErreurLectureImage as erreur:

            self._gerer_erreur_lecture(
                chemin_image,
                erreur
            )

            return


        # ----------------------------------------------------------
        # Détection
        # ----------------------------------------------------------

        (
            masque_bestiole,
            fond,
            seuil_utilise,
            bbox_detectee,
            bbox_securisee
        ) = self._calculer_detection(
            image=image,
            chemin_image=chemin_image
        )

        # ----------------------------------------------------------
        # État de la détection
        # ----------------------------------------------------------

        self._gerer_etat_detection(
            image=image,
            chemin_image=chemin_image,
            bbox_detectee=bbox_detectee,
            bbox_securisee=bbox_securisee
        )

        # ----------------------------------------------------------
        # Calcul de l'occupation
        # ----------------------------------------------------------

        (
            surface_image,
            surface_bestiole,
            occupation_pct,
            occupation_bbox_pct
        ) = self._calculer_occupation(
            image=image,
            masque_bestiole=masque_bestiole,
            bbox_securisee=bbox_securisee
        )

        # ----------------------------------------------------------
        # Informations
        # ----------------------------------------------------------

        self._afficher_informations_image(
            bbox_detectee=bbox_detectee,
            fond=fond,
            seuil_utilise=seuil_utilise,
            surface_image=surface_image,
            surface_bestiole=surface_bestiole,
            occupation_pct=occupation_pct,
            occupation_bbox_pct=occupation_bbox_pct
        )


        # ----------------------------------------------------------
        # Affichage du résultat
        # ----------------------------------------------------------

        self._afficher_resultat_image(
            image=image,
            bbox_detectee=bbox_detectee,
            bbox_securisee=bbox_securisee
        )

    def definir_etat_image(
        self,
        chemin_image,
        etat
    ):

        chemin_image = Path(
            chemin_image
        )

        if chemin_image not in self.etat_images:
            return

        if etat not in SYMBOLES_ETAT:
            return

        # Mémorisation
        self.etat_images[
            chemin_image
        ] = etat

        symbole = SYMBOLES_ETAT[
            etat
        ]

        # Recherche de l'image dans le QComboBox
        for index in range(
            self.combo_images.count()
        ):

            chemin = self.combo_images.itemData(
                index
            )

            if chemin == chemin_image:

                if etat in (
                    ETAT_SANS_DETECTION,
                    ETAT_ERREUR_LECTURE
                ):

                    texte = (
                        f"{symbole}  "
                        f"{chemin_image.name}  "
                        f"[{LIBELLES_ETAT[etat]}]"
                    )

                else:

                    texte = (
                        f"{symbole}  "
                        f"{chemin_image.name}"
                    )

                self.combo_images.setItemText(
                    index,
                    texte
                )

                break

        self.mettre_a_jour_resume()

        if (
            self.chemin_image_courante is not None
            and chemin_image == self.chemin_image_courante
        ):

            self._mettre_a_jour_decision(chemin_image)

    def sauvegarder_bbox_rouge(self):

        if (
            self.image_courante is None
            or self.bbox_detectee_courante is None
            or self.chemin_image_courante is None
        ):
            return

        fichier = sauvegarder_bbox(
            image=self.image_courante,
            bbox=self.bbox_detectee_courante,
            chemin_image=self.chemin_image_courante,
            suffixe="Bbox_Rouge"
        )

        if fichier is not None:

            print(f"Image sauvegardée : {fichier}")

            self.definir_etat_image(
                self.chemin_image_courante,
                ETAT_SAUVEGARDEE_ROUGE
            )

            self.image_suivante()

        else:

            print("Erreur pendant la sauvegarde de la BBox rouge.")

    def sauvegarder_bbox_verte(self):

        if (
            self.image_courante is None
            or self.bbox_securisee_courante is None
            or self.chemin_image_courante is None
        ):
            return

        fichier = sauvegarder_bbox(
            image=self.image_courante,
            bbox=self.bbox_securisee_courante,
            chemin_image=self.chemin_image_courante,
            suffixe="Bbox_Verte"
        )

        if fichier is not None:

            print(f"Image sauvegardée : {fichier}")

            self.definir_etat_image(
                self.chemin_image_courante,
                ETAT_SAUVEGARDEE_VERTE
            )

            self.image_suivante()

        else:

            print("Erreur pendant la sauvegarde de la BBox verte.")

    def ne_pas_sauvegarder(self):

        if self.chemin_image_courante is None:
            return

        print(
            f"Image non sauvegardée : "
            f"{self.chemin_image_courante.name}"
        )

       # ----------------------------------------------------------
       # Conserver l'état anomalie s'il existe déjà
       # ----------------------------------------------------------

        etat_actuel = self.etat_images.get(
            self.chemin_image_courante
        )

        if etat_actuel not in (
            ETAT_SANS_DETECTION,
            ETAT_ERREUR_LECTURE
        ):

            self.definir_etat_image(
                self.chemin_image_courante,
                ETAT_SANS_SAUVEGARDE
            )

        self.image_suivante()

    def image_suivante(self):
        """
        Sélectionne la prochaine image restant à examiner.

        La recherche commence après l'image actuelle puis,
        si nécessaire, reprend depuis le début du QComboBox.
        """

        nombre_images = (self.combo_images.count())

        if nombre_images == 0:
            return

        index_actuel = (self.combo_images.currentIndex())


        # ----------------------------------------------------------
        # Recherche après l'image actuelle
        # ----------------------------------------------------------
        for index in range(
            index_actuel + 1,
            nombre_images
        ):

            chemin_image = (
                self.combo_images.itemData(
                    index
                )
            )

            if (self.etat_images.get(chemin_image) == ETAT_A_TRAITER):

                self.combo_images.setCurrentIndex(index)

                return

        # ----------------------------------------------------------
        # Recherche depuis le début
        # ----------------------------------------------------------

        for index in range(
            0,
            index_actuel
        ):

            chemin_image = (
                self.combo_images.itemData(
                    index
                )
            )

            if (
                self.etat_images.get(
                    chemin_image
                )
                == ETAT_A_TRAITER
            ):

                self.combo_images.setCurrentIndex(index)

                return

        # ----------------------------------------------------------
        # Plus aucune image à traiter
        # ----------------------------------------------------------

        self.afficher_recapitulatif_final()

    def afficher_diagnostics(
        self,
        actif
    ):
        """
        Affiche ou masque la zone des diagnostics.
        """

        self.onglets_diagnostics.setVisible(
            actif
        )

        if actif:

            self.mettre_a_jour_masque()
            self.mettre_a_jour_distances()
            self.mettre_a_jour_histogramme()
            self.mettre_a_jour_pixels_critiques()
            self.mettre_a_jour_calcul_seuil()

    def mettre_a_jour_masque(self):
        """
        Affiche le masque de détection de l'image courante.
        """

        if self.masque_courant is None:

            self.affichage_masque.clear()

            self.affichage_masque.setText(
                "Aucun masque disponible"
            )

            return

        self._afficher_image_gris(
            image=self.masque_courant,
            label=self.affichage_masque
        )

    def mettre_a_jour_resume(self):
        """
        Met à jour le résumé du traitement à partir
        de l'état actuel des images.
        """

        total = len(
            self.etat_images
        )

        sauvegardees_rouges = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SAUVEGARDEE_ROUGE
        )

        sauvegardees_vertes = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SAUVEGARDEE_VERTE
        )

        sauvegardees_rouges_form = util.format_nombre(sauvegardees_rouges)
        sauvegardees_vertes_form = util.format_nombre(sauvegardees_vertes)

        sans_sauvegarde = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SANS_SAUVEGARDE
        )

        sans_detection = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SANS_DETECTION
        )

        erreurs_lecture = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_ERREUR_LECTURE
        )

        restantes = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_A_TRAITER
        )

        # ----------------------------------------------------------
        # Formatage des nombres
        # ----------------------------------------------------------

        total_form = util.format_nombre(total)
        sauvegardees_rouges_form = util.format_nombre(sauvegardees_rouges)
        sauvegardees_vertes_form = util.format_nombre(sauvegardees_vertes)
        sans_sauvegarde_form = util.format_nombre(sans_sauvegarde)
        sans_detection_form = util.format_nombre(sans_detection)
        erreurs_lecture_form = util.format_nombre(erreurs_lecture)
        restantes_form = util.format_nombre(restantes)

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------

        texte = (
            f"Images : {total_form}    |    "
            f"✓R Rouge : {sauvegardees_rouges_form}    |    "
            f"✓V Verte : {sauvegardees_vertes_form}    |    "
            f"─ Sans sauvegarde : {sans_sauvegarde_form}    |    "
            f"⚠ Sans détection : {sans_detection_form}    |    "
            f"⚠ Erreurs lecture : {erreurs_lecture_form}    |    "
            f"○ Restantes : {restantes_form}"
        )

        self.resume_traitement.setText(texte)

    def afficher_recapitulatif_final(self):
        """
        Affiche le récapitulatif final lorsque toutes les images
        ont été examinées.
        """

        total = len(
            self.etat_images
        )

        sauvegardees_rouges = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SAUVEGARDEE_ROUGE
        )

        sauvegardees_vertes = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SAUVEGARDEE_VERTE
        )



        sans_sauvegarde = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SANS_SAUVEGARDE
        )

        sans_detection = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SANS_DETECTION
        )

        erreurs_lecture = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_ERREUR_LECTURE
        )

        sans_detection_form = util.format_nombre(sans_detection)
        erreurs_lecture_form = util.format_nombre(erreurs_lecture)

        restantes = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_A_TRAITER
        )

        # ----------------------------------------------------------
        # Le récapitulatif final ne doit apparaître que lorsque
        # toutes les images ont été examinées.
        # ----------------------------------------------------------
        if restantes != 0:
            return

        # ----------------------------------------------------------
        # Génération des rapports
        # ----------------------------------------------------------
        self.generer_rapports()


        # ----------------------------------------------------------
        # Formatage des nombres
        # ----------------------------------------------------------

        total_form = util.format_nombre(total)
        sauvegardees_rouges_form = util.format_nombre(sauvegardees_rouges)
        sauvegardees_vertes_form = util.format_nombre(sauvegardees_vertes)
        sans_sauvegarde_form = util.format_nombre(sans_sauvegarde)


        # ----------------------------------------------------------
        # Texte final
        # ----------------------------------------------------------
        texte = (
            "TRAITEMENT TERMINÉ\n\n"
            f"Images examinées       : {total_form} / {total_form}\n"
            f"BBox rouge sauvegardée : {sauvegardees_rouges_form}\n"
            f"BBox verte sauvegardée : {sauvegardees_vertes_form}\n"
            f"Sans sauvegarde        : {sans_sauvegarde_form}\n"
            f"Sans détection         : {sans_detection_form}\n"
            f"Erreurs de lecture     : {erreurs_lecture_form}"
        )

        self.resume_traitement.setText(texte)

        self.resume_traitement.setAlignment(Qt.AlignCenter)

        self.resume_traitement.setStyleSheet(
            f"""
            QLabel {{
                padding: 12px;
                margin: 4px;
                font-weight: bold;
                font-size: 14px;

                color: {COULEUR_SUCCES_TEXTE};
                background-color: {COULEUR_SUCCES_FOND};

                border: 1px solid {COULEUR_SUCCES_BORDURE};
                border-radius: 6px;
            }}
            """
        )

    def generer_rapports(self):
        """
        Génère les fichiers de rapport à partir
        des états mémorisés dans self.etat_images.
        """

        if not self.etat_images:
            return

        # ----------------------------------------------------------
        # Répertoire
        # ----------------------------------------------------------

        premier_chemin = next(
            iter(
                self.etat_images
            )
        )

        repertoire = (
            premier_chemin.parent
        )

        # ----------------------------------------------------------
        # Images sans détection
        # ----------------------------------------------------------

        images_sans_detection = sorted(
            chemin.name
            for chemin, etat
            in self.etat_images.items()
            if etat == ETAT_SANS_DETECTION
        )

        # ----------------------------------------------------------
        # Images avec erreur de lecture
        # ----------------------------------------------------------

        images_erreur_lecture = sorted(
            chemin.name
            for chemin, etat
            in self.etat_images.items()
            if etat == ETAT_ERREUR_LECTURE
        )

        # ----------------------------------------------------------
        # Rapport : sans détection
        # ----------------------------------------------------------

        nom_fichier = f"{repertoire.name}_images_sans_detection.txt"

        fichier_sans_detection = (
            sauvegarder_liste_erreurs(
                repertoire=repertoire,
                nom_fichier=nom_fichier,
                liste=images_sans_detection
            )
        )

        # ----------------------------------------------------------
        # Rapport : erreurs de lecture
        # ----------------------------------------------------------

        nom_fichier = f"{repertoire.name}_images_erreur_lecture.txt"

        fichier_erreurs_lecture = (
            sauvegarder_liste_erreurs(
                repertoire=repertoire,
                nom_fichier=nom_fichier,
                liste=images_erreur_lecture
            )
        )

        # ----------------------------------------------------------
        # Informations console provisoires
        # ----------------------------------------------------------

        if fichier_sans_detection is not None:

            print(
                f"Rapport créé : "
                f"{fichier_sans_detection}"
            )

        if fichier_erreurs_lecture is not None:

            print(
                f"Rapport créé : "
                f"{fichier_erreurs_lecture}"
            )

    def mettre_a_jour_distances(self):
        """
        Affiche la carte des distances à la couleur du fond.
        """

        # ----------------------------------------------------------
        # Aucune donnée disponible
        # ----------------------------------------------------------

        if self.difference_courante is None:

            self.affichage_distances.clear()

            self.affichage_distances.setText(
                "Aucune carte des distances disponible"
            )

            return

        difference = self.difference_courante

        # ----------------------------------------------------------
        # Normalisation entre 0 et 255
        # ----------------------------------------------------------

        difference_normalisee = cv2.normalize(
            difference,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------------
        # Application de la palette thermique
        # ----------------------------------------------------------

        carte_bgr = cv2.applyColorMap(
            difference_normalisee,
            cv2.COLORMAP_INFERNO
        )

        # ----------------------------------------------------------
        # Conversion BGR -> RGB
        # ----------------------------------------------------------

        carte_rgb = cv2.cvtColor(
            carte_bgr,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------

        self._afficher_image_rgb(
            image=carte_rgb,
            label=self.affichage_distances
        )

    def mettre_a_jour_histogramme(self):
        """
        Affiche l'histogramme des distances à la couleur du fond.
        """

        # ----------------------------------------------------------
        # Effacement de l'ancien graphique
        # ----------------------------------------------------------

        self.axes_histogramme.clear()

        # ----------------------------------------------------------
        # Aucune donnée
        # ----------------------------------------------------------

        if self.difference_courante is None:

            self.axes_histogramme.text(
                0.5,
                0.5,
                "Aucun histogramme disponible",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.axes_histogramme.transAxes
            )

            self.axes_histogramme.set_axis_off()

            self.canvas_histogramme.draw()

            return

        difference = self.difference_courante

        # On remet les axes, au cas où le graphique précédent
        # correspondait à une image illisible.
        self.axes_histogramme.set_axis_on()

        # ----------------------------------------------------------
        # Histogramme
        # ----------------------------------------------------------

        self.axes_histogramme.hist(
            difference.ravel(),
            bins=100,
            color="steelblue",
            alpha=0.7
        )

        # ----------------------------------------------------------
        # Seuil utilisé
        # ----------------------------------------------------------

        if self.seuil_courant is not None:

            self.axes_histogramme.axvline(
                self.seuil_courant,
                color="red",
                linewidth=2,
                label=f"Seuil = {self.seuil_courant:.1f}"
            )

        # ----------------------------------------------------------
        # Percentiles
        # ----------------------------------------------------------

        p90 = np.percentile(
            difference,
            90
        )

        p95 = np.percentile(
            difference,
            95
        )

        p99 = np.percentile(
            difference,
            99
        )

        self.axes_histogramme.axvline(
            p90,
            color="green",
            linestyle="--",
            label=f"P90 = {p90:.1f}"
        )

        self.axes_histogramme.axvline(
            p95,
            color="orange",
            linestyle="--",
            label=f"P95 = {p95:.1f}"
        )

        self.axes_histogramme.axvline(
            p99,
            color="purple",
            linestyle="--",
            label=f"P99 = {p99:.1f}"
        )

        # ----------------------------------------------------------
        # Présentation
        # ----------------------------------------------------------

        self.axes_histogramme.set_xlabel(
            "Distance à la couleur du fond"
        )

        self.axes_histogramme.set_ylabel(
            "Nombre de pixels"
        )

        self.axes_histogramme.set_title(
            "Histogramme des distances au fond"
        )

        self.axes_histogramme.legend()

        self.axes_histogramme.grid(
            alpha=0.3
        )

        self.figure_histogramme.tight_layout()

        # ----------------------------------------------------------
        # Rafraîchissement du Canvas Qt
        # ----------------------------------------------------------

        self.canvas_histogramme.draw()

    def mettre_a_jour_pixels_critiques(self):
        """
        Affiche les pixels conservés avec le seuil actuel
        mais qui seraient supprimés avec seuil + PAS_SEUIL.
        """

        # ----------------------------------------------------------
        # Vérification des données
        # ----------------------------------------------------------

        if (
            self.difference_courante is None
            or self.seuil_courant is None
        ):

            self.informations_pixels_critiques.setText(
                "Seuil actuel : -    |    "
                "Seuil testé : -    |    "
                "Pixels critiques : -    |    "
                "Pourcentage : -"
            )

            self.affichage_pixels_critiques.clear()

            self.affichage_pixels_critiques.setText(
                "Aucun pixel critique disponible"
            )

            return

        # ----------------------------------------------------------
        # Données courantes
        # ----------------------------------------------------------

        difference = self.difference_courante

        seuil = float(
            self.seuil_courant
        )

        seuil_suivant = (
            seuil + PAS_SEUIL
        )

        # ----------------------------------------------------------
        # Calcul des pixels critiques
        # ----------------------------------------------------------

        pixels_critiques = (
            (difference > seuil)
            & (difference <= seuil_suivant)
        ).astype(
            np.uint8
        ) * 255

        # ----------------------------------------------------------
        # Statistiques
        # ----------------------------------------------------------

        nombre_pixels_critiques = int(
            np.count_nonzero(
                pixels_critiques
            )
        )

        nombre_pixels_total = int(
            pixels_critiques.size
        )

        if nombre_pixels_total == 0:

            pourcentage_critique = 0.0

        else:

            pourcentage_critique = (
                nombre_pixels_critiques
                / nombre_pixels_total
                * 100.0
            )

        # ----------------------------------------------------------
        # Formatage
        # ----------------------------------------------------------

        nombre_pixels_form = util.format_nombre(
            nombre_pixels_critiques
        )

        # ----------------------------------------------------------
        # Mise à jour des informations
        # ----------------------------------------------------------

        texte_informations = (
            f"Seuil actuel : {seuil:.3f}    |    "
            f"Seuil testé : {seuil_suivant:.3f}    |    "
            f"Pixels critiques : {nombre_pixels_form}    |    "
            f"{pourcentage_critique:.3f} % de l'image"
        )

        self.informations_pixels_critiques.setText(
            texte_informations
        )

        self._afficher_image_gris(
            image=pixels_critiques,
            label=self.affichage_pixels_critiques
        )

    def mettre_a_jour_calcul_seuil(self):
        """
        Affiche les informations détaillées du calcul
        automatique du seuil.
        """

        informations = (
            self.informations_seuil_courant
        )

        # ----------------------------------------------------------
        # Pas d'informations disponibles
        # ----------------------------------------------------------

        if informations is None:

            for valeur in (
                self.valeurs_calcul_seuil.values()
            ):

                valeur.setText(
                    "-"
                )

            return

        # ----------------------------------------------------------
        # Valeurs numériques
        # ----------------------------------------------------------

        self.valeurs_calcul_seuil[
            "mediane_fond"
        ].setText(
            f"{informations['mediane_fond']:.3f}"
        )

        self.valeurs_calcul_seuil[
            "mad_fond"
        ].setText(
            f"{informations['mad_fond']:.3f}"
        )

        self.valeurs_calcul_seuil[
            "sigma_robuste_fond"
        ].setText(
            f"{informations['sigma_robuste_fond']:.3f}"
        )

        self.valeurs_calcul_seuil[
            "seuil_mad"
        ].setText(
            f"{informations['seuil_mad']:.3f}"
        )

        self.valeurs_calcul_seuil[
            "seuil_percentile"
        ].setText(
            f"{informations['seuil_percentile']:.3f}"
        )

        rapport = informations[
            "rapport_percentile_mad"
        ]

        if np.isfinite(
            rapport
        ):

            texte_rapport = (
                f"{rapport:.3f}"
            )

        else:

            texte_rapport = "Infini"

        self.valeurs_calcul_seuil[
            "rapport_percentile_mad"
        ].setText(
            texte_rapport
        )

        self.valeurs_calcul_seuil[
            "percentile_suspect"
        ].setText(
            "Oui"
            if informations["percentile_suspect"]
            else "Non"
        )

        self.valeurs_calcul_seuil[
            "methode_seuil"
        ].setText(
            informations["methode_seuil"]
        )

        self.valeurs_calcul_seuil[
            "percentile_fond"
        ].setText(
            f"{informations['percentile_fond']:.1f} %"
        )

        self.valeurs_calcul_seuil[
            "taille_bord"
        ].setText(
            f"{informations['taille_bord']} px"
        )

        nombre_pixels = util.format_nombre(
            informations[
                "nombre_pixels_fond"
            ]
        )

        self.valeurs_calcul_seuil[
            "nombre_pixels_fond"
        ].setText(
            nombre_pixels
        )

        self.valeurs_calcul_seuil[
            "seuil_final"
        ].setText(
            f"{informations['seuil_final']:.3f}"
        )

    def _creer_aide_diagnostic(
        self,
        texte
    ):
        """
        Crée une courte aide explicative pour
        un onglet de diagnostic.
        """

        label = QLabel(texte)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE_SECONDAIRE};
                padding: 4px 6px;
                font-style: italic;
            }}
            """
        )

        return label


    def resizeEvent(
        self,
        event
    ):
        """
        Réadapte les images affichées lorsque
        la fenêtre change de dimensions.
        """

        # Laisse QMainWindow effectuer son traitement normal.
        super().resizeEvent(
            event
        )

        # ----------------------------------------------------------
        # Image principale
        # ----------------------------------------------------------

        if self.image_rgb_affichage is not None:

            self._afficher_image_rgb(
                image=self.image_rgb_affichage,
                label=self.affichage_image
            )


    def closeEvent(
        self,
        event
    ):
        """
        Confirme la fermeture et génère les rapports
        d'anomalies avant de quitter.
        """

        # ----------------------------------------------------------
        # Nombre d'images restant à examiner
        # ----------------------------------------------------------

        restantes = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_A_TRAITER
        )

        # ----------------------------------------------------------
        # Confirmation
        # ----------------------------------------------------------

        if restantes > 0:

            restantes_form = util.format_nombre(
                restantes
            )

            reponse = QMessageBox.question(
                self,
                "Quitter Aqua-IA",
                (
                    f"Il reste {restantes_form} image(s) "
                    "à examiner.\n\n"
                    "Les rapports des anomalies déjà détectées "
                    "seront sauvegardés avant la fermeture.\n\n"
                    "Voulez-vous vraiment quitter ?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reponse != QMessageBox.Yes:

                event.ignore()

                return

        # ----------------------------------------------------------
        # Génération des rapports
        # ----------------------------------------------------------

        if self.etat_images:

            try:

                self.generer_rapports()

            except Exception as erreur:

                print(
                    "Erreur lors de la génération "
                    f"des rapports : {erreur}"
                )

        event.accept()


#----------------------------------------------------------------------------------------
def main():

    application = QApplication(sys.argv)

    interface = InterfaceInfoCrop()

    interface.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()

    