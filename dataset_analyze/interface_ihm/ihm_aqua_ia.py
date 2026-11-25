import sys
from pathlib import Path
import yaml
import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
        QFont,

)


from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QSizePolicy,
    QLabel,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QProgressBar,
    QHBoxLayout,
    QComboBox,
    QCheckBox,

)


from config import constants as cst
from tools import utility_ihm as ihm



#==================================================================================================

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

# État : à vérifier
COULEUR_VERIFICATION_TEXTE = "#856404"
COULEUR_VERIFICATION_FOND = "#FFF3CD"
COULEUR_VERIFICATION_BORDURE = "#FFCC00"

# État : incorrect
COULEUR_ERREUR_TEXTE = "#721C24"
COULEUR_ERREUR_FOND = "#F8D7DA"
COULEUR_ERREUR_BORDURE = "#CC0000"



COULEUR_DESACTIVE = "#D0D0D0"
COULEUR_TEXTE_DESACTIVE = "#888888"
COULEUR_BORDURE_DESACTIVE = "#BBBBBB"

COULEUR_FOND_BOX = "#F8F8F8"
COULEUR_BORDURE_BOX = "#B8B8B8"
COULEUR_TITRE_BOX = COULEUR_AQUA_DARK


# ==================================================================================================
# COULEURS D'ÉTAT
# ==================================================================================================

COULEUR_SUCCES_TEXTE = "#155724"
COULEUR_SUCCES_FOND = "#D4EDDA"
COULEUR_SUCCES_BORDURE = "#C3E6CB"


# ==========================================================
# Style commun des barres de progression
# ==========================================================
STYLE_PROGRESSION = f"""
    QProgressBar {{
        border: 1px solid {COULEUR_AQUA_DARK};
        border-radius: 5px;
        background-color: #E8E8E8;
        color: {COULEUR_TEXTE};
        text-align: center;
        font-weight: bold;
    }}

    QProgressBar::chunk {{
        background-color: {COULEUR_AQUA};
        border-radius: 4px;
    }}
"""








#==================================================================================================
#==================================================================================================

class IhmAquaIa(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aqua-IA - Détection")
        self.resize(900, 650)

        self.etat_images = {}

        # ==========================================================
        # Mémorisation du filtre utilisé pour chaque zone
        # ==========================================================
        self.filtre_par_zone = {}

        self.repertoire_selectionne = None
        self.repertoire_annotations = None
        
        self.liste_images = []
        self.liste_repertoires = []
        
        # Correspondance :
        # numéro de classe YOLO -> nom de classe
        self.classes_yolo = {}


        self._creer_interface()


#========================================================================================
#  Fonctions génériques 
#========================================================================================

    # Création Label Affichage
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
            label.setMinimumSize( largeur_min, hauteur_min)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

    def _creer_box(
        self,
        titre
    ):
        """
        Crée une box standard avec un cadre fin et un titre.
        """

        box = QGroupBox(titre)

        box.setStyleSheet(
            f"""
            QGroupBox {{
                background-color: {COULEUR_FOND_BOX};
                color: {COULEUR_TITRE_BOX};
                border: 1px solid {COULEUR_BORDURE_BOX};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 5px;
                background-color: {COULEUR_FOND_BOX};
                color: {COULEUR_TITRE_BOX};
            }}
            """
        )

        return box

    def image_precedente(self):
        """
        Sélectionne l'image précédente.
        """

        index_actuel = self.combo_image.currentIndex()

        # index 0 = "Choisir une image..."
        # index 1 = première vraie image
        if index_actuel <= 1:
            return

        self.combo_image.setCurrentIndex(
            index_actuel - 1
        )

    def image_suivante(self):
        """
        Sélectionne l'image suivante.
        """

        index_actuel = self.combo_image.currentIndex()
        dernier_index = self.combo_image.count() - 1

        if index_actuel <= 0:
            return

        if index_actuel >= dernier_index:
            return

        self.combo_image.setCurrentIndex(
            index_actuel + 1
        )


#========================================================================================

#----------------------------------------------------------------------------------------
# Fonction gestion du graphique
#----------------------------------------------------------------------------------------
 
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
        sous_titre = QLabel("***** Titre à Définir *****")
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

    def _creer_selection_repertoire(
        self,
        layout_principal
    ):
        """
        Crée la zone de sélection du répertoire.
        """

        # ==========================================================
        # Box principale
        # ==========================================================
        box_selection = self._creer_box(
            "Sélection du répertoire"
        )

        layout_selection = QGridLayout(box_selection)

        layout_selection.setContentsMargins(
            12,
            18,
            12,
            12
        )

        layout_selection.setHorizontalSpacing(10)
        layout_selection.setVerticalSpacing(8)

        # ==========================================================
        # Répertoire
        # ==========================================================
        label_repertoire = QLabel("Répertoire :")

        self.champ_repertoire = QLineEdit()
        self.champ_repertoire.setReadOnly(True)
        self.champ_repertoire.setPlaceholderText(
            "Sélectionnez le répertoire contenant les images"
        )

        self.bouton_parcourir = QPushButton("Parcourir...")

        self.bouton_parcourir.setStyleSheet(
            self._style_bouton(
                couleur_fond=COULEUR_AQUA,
                couleur_texte=COULEUR_AQUA_TEXTE,
                couleur_bordure=COULEUR_AQUA_DARK,
                couleur_hover=COULEUR_AQUA_LIGHT,
                couleur_texte_hover=COULEUR_AQUA_TEXTE,
                couleur_pressed=COULEUR_AQUA_PRESSED
            )
        )

        # ==========================================================
        # Placement
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

        layout_selection.setColumnStretch(
            1,
            1
        )

        layout_principal.addWidget(box_selection)

    def selectionner_repertoire(self):
        """
        Sélectionne le répertoire racine et recherche les images
        dans celui-ci ainsi que dans ses sous-répertoires.
        """

        repertoire = QFileDialog.getExistingDirectory(
            self,
            "Sélectionner le répertoire contenant les images"
        )

        if not repertoire:
            return

        # ==========================================================
        # Conversion et mémorisation
        # ==========================================================
        self.repertoire_selectionne = Path(repertoire)

        self.champ_repertoire.setText(
            str(self.repertoire_selectionne)
        )

        # ==========================================================
        # Recherche récursive des images
        # ==========================================================
        self.liste_images = sorted(
            fichier
            for fichier in self.repertoire_selectionne.rglob("*")
            if (
                fichier.is_file()
                and fichier.suffix.lower() in cst.IMAGE_EXT
            )
        )

        # ==========================================================
        # Répertoires contenant au moins une image
        # ==========================================================
        self.liste_repertoires = sorted(
            {
                image.parent
                for image in self.liste_images
            }
        )

        nombre_images = len(self.liste_images)
        nombre_repertoires = len(self.liste_repertoires)

        # ==========================================================
        # Initialisation des barres
        # ==========================================================
        self.progression_repertoires.setRange(
            0,
            max(1, nombre_repertoires)
        )

        self.progression_repertoires.setValue(0)

        self.progression_images.setRange(
            0,
            1
        )

        self.progression_images.setValue(0)

        # ==========================================================
        # Mise à jour de l'état
        # ==========================================================
        if nombre_images == 0:
            self.bouton_lancer_detection.setEnabled(False)

            self.label_etat_detection.setText(
                "Aucune image compatible trouvée dans le répertoire."
            )

            return

        self.bouton_lancer_detection.setEnabled(True)

        self.label_etat_detection.setText(
            f"{nombre_images} image(s) trouvée(s) "
            f"dans {nombre_repertoires} répertoire(s)."
        )


        # ==========================================================
        # Mise à jour de la consultation des annotations
        # ==========================================================
        self._charger_zones_annotations()


    def _creer_zone_execution(
        self,
        layout_principal
    ):
        """
        Crée la zone d'exécution de la détection :
        - bouton de lancement ;
        - répertoire en cours ;
        - progression des répertoires ;
        - image en cours ;
        - progression des images ;
        - état général du traitement.
        """

        # ==========================================================
        # Box principale
        # ==========================================================
        box_execution = self._creer_box(
            "Exécution de la détection"
        )

        layout_execution = QGridLayout(box_execution)

        layout_execution.setContentsMargins(
            12,
            18,
            12,
            12
        )

        layout_execution.setHorizontalSpacing(10)
        layout_execution.setVerticalSpacing(10)

        # ==========================================================
        # Bouton de lancement
        # ==========================================================
        self.bouton_lancer_detection = QPushButton(
            "Lancer la détection"
        )

        self.bouton_lancer_detection.setEnabled(False)

        self.bouton_lancer_detection.setStyleSheet(
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
        # Répertoire courant
        # ==========================================================
        self.label_repertoire_courant = QLabel(
            "Répertoire : —"
        )

        self.label_repertoire_courant.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE};
                font-weight: bold;
                padding-top: 4px;
            }}
            """
        )

        # ==========================================================
        # Progression des répertoires
        # ==========================================================
        self.progression_repertoires = QProgressBar()

        self.progression_repertoires.setRange(
            0,
            1
        )

        self.progression_repertoires.setValue(0)

        self.progression_repertoires.setFormat(
            "%v / %m répertoires"
        )

        self.progression_repertoires.setFixedHeight(24)

        self.progression_repertoires.setStyleSheet(
            STYLE_PROGRESSION
        )

        # ==========================================================
        # Image courante
        # ==========================================================
        self.label_image_courante = QLabel(
            "Image : —"
        )

        self.label_image_courante.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE};
                font-weight: bold;
                padding-top: 8px;
            }}
            """
        )

        # ==========================================================
        # Progression des images
        # ==========================================================
        self.progression_images = QProgressBar()

        self.progression_images.setRange(
            0,
            1
        )

        self.progression_images.setValue(0)

        self.progression_images.setFormat(
            "%v / %m images"
        )

        self.progression_images.setFixedHeight(24)

        self.progression_images.setStyleSheet(
            STYLE_PROGRESSION
        )

        # ==========================================================
        # État général du traitement
        # ==========================================================
        self.label_etat_detection = QLabel(
            "En attente de la sélection d’un répertoire."
        )

        self.label_etat_detection.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE_SECONDAIRE};
                font-style: italic;
                padding: 4px;
            }}
            """
        )

        # ==========================================================
        # Placement des widgets
        # ==========================================================

        # Bouton
        layout_execution.addWidget(
            self.bouton_lancer_detection,
            0,
            0,
            1,
            2,
            Qt.AlignCenter
        )

        # Répertoire courant
        layout_execution.addWidget(
            self.label_repertoire_courant,
            1,
            0,
            1,
            2
        )

        # Barre de progression des répertoires
        layout_execution.addWidget(
            self.progression_repertoires,
            2,
            0,
            1,
            2
        )

        # Image courante
        layout_execution.addWidget(
            self.label_image_courante,
            3,
            0,
            1,
            2
        )

        # Barre de progression des images
        layout_execution.addWidget(
            self.progression_images,
            4,
            0,
            1,
            2
        )

        # État général
        layout_execution.addWidget(
            self.label_etat_detection,
            5,
            0,
            1,
            2
        )

        # ==========================================================
        # Redimensionnement
        # ==========================================================
        layout_execution.setColumnStretch(
            0,
            1
        )

        layout_execution.setColumnStretch(
            1,
            1
        )

        # ==========================================================
        # Ajout au layout principal
        # ==========================================================
        layout_principal.addWidget(
            box_execution
        )

    def _creer_consultation_annotations(
    self,
    layout_principal
):
        """
        Crée la zone permettant :
        - de sélectionner le répertoire des annotations ;
        - de sélectionner une zone ;
        - de sélectionner et parcourir les images ;
        - d'afficher l'image originale ;
        - d'afficher l'image annotée ;
        - d'afficher les classes présentes.
        """

        # ==========================================================
        # Box principale
        # ==========================================================
        box_annotations = self._creer_box(
            "Consultation des annotations"
        )

        layout_annotations = QGridLayout(
            box_annotations
        )

        layout_annotations.setContentsMargins(
            12,
            18,
            12,
            12
        )

        layout_annotations.setHorizontalSpacing(12)
        layout_annotations.setVerticalSpacing(10)

        # ==========================================================
        # Répertoire des annotations
        # ==========================================================
        label_repertoire_annotations = QLabel(
            "Répertoire :"
        )

        self.champ_repertoire_annotations = QLineEdit()
        self.champ_repertoire_annotations.setReadOnly(True)

        self.champ_repertoire_annotations.setPlaceholderText(
            "Sélectionnez le répertoire contenant les annotations"
        )

        self.bouton_parcourir_annotations = QPushButton("Parcourir...")
        self.bouton_parcourir_annotations.setFixedWidth(130)




        self.bouton_parcourir_annotations.setStyleSheet(
            self._style_bouton(
                couleur_fond=COULEUR_AQUA,
                couleur_texte=COULEUR_AQUA_TEXTE,
                couleur_bordure=COULEUR_AQUA_DARK,
                couleur_hover=COULEUR_AQUA_LIGHT,
                couleur_texte_hover=COULEUR_AQUA_TEXTE,
                couleur_pressed=COULEUR_AQUA_PRESSED
            )
        )

        # ==========================================================
        # Sélection de la zone
        # ==========================================================
        label_zone = QLabel(
            "Zone :"
        )

        self.combo_zone = QComboBox()
        self.combo_zone.setEnabled(False)

        self.combo_zone.setPlaceholderText(
            "Sélectionnez une zone"
        )

        # ==========================================================
        # Filtre des images
        # ==========================================================
        label_filtre_image = QLabel(
            "Afficher :"
        )

        self.combo_filtre_etat = QComboBox()

        self.combo_filtre_etat.addItems([
            "Toutes les images",
            "Non contrôlées",
            "Correctes",
            "À vérifier",
            "Incorrectes",
        ])

        self.combo_filtre_etat.setCurrentIndex(0)

        
        # ==========================================================
        # Sélection de l'image
        # ==========================================================
        label_image = QLabel(
            "Image :"
        )

        self.combo_image = QComboBox()
        self.combo_image.setEnabled(False)

        self.combo_image.setPlaceholderText(
            "Sélectionnez une image"
        )

        # ==========================================================
        # Boutons de navigation
        # ==========================================================
        self.bouton_image_precedente = QPushButton ("◀ Précédente")
        self.bouton_image_suivante = QPushButton("Suivante ▶")

        self.bouton_image_precedente.setEnabled(False)
        self.bouton_image_suivante.setEnabled(False)

        # ==========================================================
        # Indicateur de position dans les images
        # ==========================================================
        self.label_position_image = QLabel("0 / 0")
        self.label_position_image.setAlignment(Qt.AlignCenter)
        self.label_position_image.setMinimumWidth(70)
        self.label_position_image.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_AQUA_DARK};
                font-weight: bold;
                padding: 5px;
            }}
            """
        )


        # ==========================================================
        # Style des boutons de navigation
        # ==========================================================
        style_navigation = self._style_bouton(
            couleur_fond=COULEUR_BOUTON_NEUTRE,
            couleur_texte="white",
            couleur_bordure=COULEUR_BOUTON_NEUTRE_PRESSED,
            couleur_hover=COULEUR_BOUTON_NEUTRE_HOVER,
            couleur_texte_hover="white",
            couleur_pressed=COULEUR_BOUTON_NEUTRE_PRESSED
        )

        self.bouton_image_precedente.setStyleSheet(
            style_navigation
        )

        self.bouton_image_suivante.setStyleSheet(
            style_navigation
        )

        # ==========================================================
        # Layout de navigation des images
        # ==========================================================
        layout_navigation_images = QHBoxLayout()

        layout_navigation_images.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout_navigation_images.addWidget(
            self.bouton_image_precedente
        )

        layout_navigation_images.addWidget(
            self.label_position_image
        )

        layout_navigation_images.addWidget(
            self.combo_image,
            1
        )

        layout_navigation_images.addWidget(
            self.bouton_image_suivante
        )

        # ==========================================================
        # Titre image originale
        # ==========================================================
        titre_image_originale = QLabel(
            "Image originale"
        )

        titre_image_originale.setAlignment(
            Qt.AlignCenter
        )

        titre_image_originale.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_AQUA_DARK};
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
            }}
            """
        )

        # ==========================================================
        # Titre image annotée
        # ==========================================================
        titre_image_annotee = QLabel(
            "Image avec annotations"
        )

        titre_image_annotee.setAlignment(
            Qt.AlignCenter
        )

        titre_image_annotee.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_AQUA_DARK};
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
            }}
            """
        )

        # ==========================================================
        # Image originale
        # ==========================================================
        self.label_image_originale = (
            self._creer_label_affichage(
                "Aucune image sélectionnée",
                largeur_min=400,
                hauteur_min=350
            )
        )

        # ==========================================================
        # Image annotée
        # ==========================================================
        self.label_image_annotee = (
            self._creer_label_affichage(
                "Aucune annotation à afficher",
                largeur_min=400,
                hauteur_min=350
            )
        )

        # ==========================================================
        # Titre des classes
        # ==========================================================
        titre_classes = QLabel(
            "Classes présentes"
        )

        titre_classes.setAlignment(
            Qt.AlignCenter
        )

        titre_classes.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_AQUA_DARK};
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
            }}
            """
        )

        # ==========================================================
        # Affichage des classes
        # ==========================================================
        self.label_classes_annotation = QLabel(
            "Aucune classe à afficher"
        )

        self.label_classes_annotation.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

        self.label_classes_annotation.setWordWrap(
            True
        )

        self.label_classes_annotation.setTextFormat(
            Qt.RichText
        )

        self.label_classes_annotation.setMinimumWidth(
            250
        )

        self.label_classes_annotation.setStyleSheet(
            f"""
            QLabel {{
                background-color: white;
                color: {COULEUR_TEXTE};
                border: 1px solid {COULEUR_BORDURE_IMAGE};
                border-radius: 4px;
                padding: 10px;
            }}
            """
        )

        # ==========================================================
        # Colonne Classes
        # ==========================================================
        layout_classes = QVBoxLayout()

        layout_classes.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout_classes.setSpacing(5)

        layout_classes.addWidget(
            titre_classes
        )

        layout_classes.addWidget(
            self.label_classes_annotation
        )

        layout_classes.addStretch()

        # ==========================================================
        # Informations fichier de labels
        # ==========================================================
        self.label_fichier_annotation = QLabel(
            "Fichier de labels : —"
        )

        self.label_fichier_annotation.setWordWrap(
            True
        )

        self.label_fichier_annotation.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE_SECONDAIRE};
                padding: 5px;
                font-style: italic;
            }}
            """
        )

        
        # ==========================================================
        #  Création des trois boutons de contrôle
        # ==========================================================
        label_controle = QLabel("Contrôle :")

        self.bouton_annotation_correcte = QPushButton("✓ Annotation correcte")
        self.bouton_annotation_verifier = QPushButton("⚠ À vérifier")
        self.bouton_annotation_incorrecte = QPushButton("✕ Annotation incorrecte")


        # ==========================================================
        # Les boutons conservent leur état visuel
        # ==========================================================
        self.bouton_annotation_correcte.setCheckable(True)
        self.bouton_annotation_verifier.setCheckable(True)
        self.bouton_annotation_incorrecte.setCheckable(True)

        # ==========================================================
        # Désactivation initiale
        # ==========================================================
        self.bouton_annotation_correcte.setEnabled(False)
        self.bouton_annotation_verifier.setEnabled(False)
        self.bouton_annotation_incorrecte.setEnabled(False)

        # ==========================================================
        # Styles des boutons de contrôle
        # ==========================================================

        # ----------------------------------------------------------
        # Annotation correcte
        # ----------------------------------------------------------
        self.bouton_annotation_correcte.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COULEUR_SUCCES_FOND};
                color: {COULEUR_SUCCES_TEXTE};
                border: 1px solid {COULEUR_SUCCES_BORDURE};
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: #C3E6CB;
                border: 2px solid {COULEUR_SUCCES_TEXTE};
            }}

            QPushButton:checked {{
                background-color: #70C080;
                color: #0B3D18;
                border: 3px solid {COULEUR_SUCCES_TEXTE};
            }}

            QPushButton:disabled {{
                background-color: {COULEUR_DESACTIVE};
                color: {COULEUR_TEXTE_DESACTIVE};
                border: 1px solid {COULEUR_BORDURE_DESACTIVE};
            }}
            """
        )

        # ----------------------------------------------------------
        # Annotation à vérifier
        # ----------------------------------------------------------
        self.bouton_annotation_verifier.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COULEUR_VERIFICATION_FOND};
                color: {COULEUR_VERIFICATION_TEXTE};
                border: 1px solid {COULEUR_VERIFICATION_BORDURE};
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: #FFE69C;
                border: 2px solid {COULEUR_VERIFICATION_TEXTE};
            }}

            QPushButton:checked {{
                background-color: #FFD966;
                color: #5C4500;
                border: 3px solid {COULEUR_VERIFICATION_TEXTE};
            }}

            QPushButton:disabled {{
                background-color: {COULEUR_DESACTIVE};
                color: {COULEUR_TEXTE_DESACTIVE};
                border: 1px solid {COULEUR_BORDURE_DESACTIVE};
            }}
            """
        )

        # ----------------------------------------------------------
        # Annotation incorrecte
        # ----------------------------------------------------------
        self.bouton_annotation_incorrecte.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COULEUR_ERREUR_FOND};
                color: {COULEUR_ERREUR_TEXTE};
                border: 1px solid {COULEUR_ERREUR_BORDURE};
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: #F1AEB5;
                border: 2px solid {COULEUR_ERREUR_TEXTE};
            }}

            QPushButton:checked {{
                background-color: #E88888;
                color: #4A0F14;
                border: 3px solid {COULEUR_ERREUR_TEXTE};
            }}

            QPushButton:disabled {{
                background-color: {COULEUR_DESACTIVE};
                color: {COULEUR_TEXTE_DESACTIVE};
                border: 1px solid {COULEUR_BORDURE_DESACTIVE};
            }}
            """
        )


        # ==========================================================
        # Layout des boutons de contrôle
        # ==========================================================
        layout_controle = QHBoxLayout()

        layout_controle.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout_controle.setSpacing(10)
        layout_controle.addWidget(self.bouton_annotation_correcte)
        layout_controle.addWidget(self.bouton_annotation_verifier)
        layout_controle.addWidget(self.bouton_annotation_incorrecte)
        layout_controle.addStretch()


        # ==========================================================
        # Récapitulatif des états de contrôle
        # ==========================================================
        self.label_recap_controle = QLabel(
            "✓ Correctes : 0    "
            "⚠ À vérifier : 0    "
            "✕ Incorrectes : 0    "
            "○ Non contrôlées : 0"
        )
        self.label_recap_controle.setAlignment(Qt.AlignCenter)
        self.label_recap_controle.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_TEXTE};
                background-color: #F5F5F5;
                border: 1px solid {COULEUR_BORDURE_IMAGE};
                border-radius: 5px;
                padding: 7px;
                font-weight: bold;
            }}
            """
        )

        # ==========================================================
        # Barre de progression du contrôle
        # ==========================================================
        self.progress_controle = QProgressBar()

        self.progress_controle.setRange(
            0,
            100
        )

        self.progress_controle.setValue(0)

        self.progress_controle.setFormat(
            "Avancement de la zone : %p %"
        )

        self.progress_controle.setTextVisible(True)

        self.progress_controle.setFixedHeight(24)

        self.progress_controle.setStyleSheet(
            STYLE_PROGRESSION
        )



        # ==========================================================
        # Passage automatique à l'image suivante
        # ==========================================================
        self.check_image_suivante_auto = QCheckBox(
            "Passer automatiquement à l'image suivante après validation"
        )

        self.check_image_suivante_auto.setChecked(
            False
        )

        self.check_image_suivante_auto.setStyleSheet(
            f"""
            QCheckBox {{
                color: {COULEUR_TEXTE};
                font-weight: bold;
                padding: 4px;
            }}

            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
            }}
            """
        )



        # ==========================================================
        # PLACEMENT
        # ==========================================================

        # ----------------------------------------------------------
        # Ligne 0 : Répertoire
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            label_repertoire_annotations,
            0,
            0
        )

        layout_annotations.addWidget(
            self.champ_repertoire_annotations,
            0,
            1
        )

        layout_annotations.addWidget(
            self.bouton_parcourir_annotations,
            0,
            2
        )

        # ----------------------------------------------------------
        # Ligne 1 : Zone
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            label_zone,
            1,
            0
        )

        layout_annotations.addWidget(
            self.combo_zone,
            1,
            1,
            1,
            2
        )

        # ----------------------------------------------------------
        # Ligne 2 : Filtre des images
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            label_filtre_image,
            2,
            0
        )

        layout_annotations.addWidget(
            self.combo_filtre_etat,
            2,
            1,
            1,
            2
        )



        # ----------------------------------------------------------
        # Ligne 3 : Navigation Images
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            label_image,
            3,
            0
        )

        layout_annotations.addLayout(
            layout_navigation_images,
            3,
            1,
            1,
            2
        )

        # ----------------------------------------------------------
        # Ligne 4 : Titres
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            titre_image_originale,
            4,
            0
        )

        layout_annotations.addWidget(
            titre_image_annotee,
            4,
            1
        )

        # ----------------------------------------------------------
        # Lignes 4 : Classes
        # ----------------------------------------------------------
        layout_annotations.addLayout(
            layout_classes,
            4,
            2,
            2,
            1
        )

        # ----------------------------------------------------------
        # Ligne 5 : Images
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            self.label_image_originale,
            5,
            0
        )

        layout_annotations.addWidget(
            self.label_image_annotee,
            5,
            1
        )

        # ----------------------------------------------------------
        # Ligne 6 : Contrôle
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            label_controle,
            6,
            0
        )

        layout_annotations.addLayout(
            layout_controle,
            6,
            1,
            1,
            2
        )

        # ----------------------------------------------------------
        # Ligne 7 : Navigation automatique
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            self.check_image_suivante_auto,
            7,
            1,
            1,
            2
        )

        # ----------------------------------------------------------
        # Ligne 8 : Récapitulatif
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            self.label_recap_controle,
            8,
            0,
            1,
            3
        )

        # ----------------------------------------------------------
        # Ligne 9 : Progression du contrôle
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            self.progress_controle,
            9,
            0,
            1,
            3
        )

        # ----------------------------------------------------------
        # Ligne 10 : Informations labels
        # ----------------------------------------------------------
        layout_annotations.addWidget(
            self.label_fichier_annotation,
            10,
            0,
            1,
            3
        )


        # ==========================================================
        # Ajout de la box au layout principal
        # ==========================================================
        layout_principal.addWidget(
            box_annotations
        )



    def selectionner_repertoire_annotations(self):
        """
        Sélectionne le répertoire contenant les résultats
        de la détection et recherche les zones disponibles.
        """

        repertoire = QFileDialog.getExistingDirectory(
            self,
            "Sélectionner le répertoire contenant les annotations"
        )

        if not repertoire:
            return

        # ==========================================================
        # Mémorisation
        # ==========================================================
        self.repertoire_annotations = Path(repertoire)
        self.champ_repertoire_annotations.setText(str(self.repertoire_annotations))

        # ==========================================================
        # Chargement des noms de classes YOLO
        # ==========================================================
        self._charger_classes_yolo(self.repertoire_annotations)

        # ==========================================================
        # Chargement des états de contrôle sauvegardés
        # ==========================================================
        self._charger_etats_controle()


        # ==========================================================
        # Chargement des zones
        # ==========================================================
        self._charger_zones_annotations()





    def _mettre_a_jour_navigation_images(self):
        """
        Met à jour :
        - les boutons Précédente / Suivante ;
        - la position de l'image dans la liste.
        """

        index_actuel = self.combo_image.currentIndex()
        dernier_index = self.combo_image.count() - 1

        # ==========================================================
        # Aucune image sélectionnée
        # ==========================================================
        if index_actuel <= 0:

            self.bouton_image_precedente.setEnabled(False)
            self.bouton_image_suivante.setEnabled(False)

            self.label_position_image.setText(
                f"0 / {max(0, dernier_index)}"
            )

            return

        # ==========================================================
        # Bouton Précédente
        # ==========================================================
        self.bouton_image_precedente.setEnabled(
            index_actuel > 1
        )

        # ==========================================================
        # Bouton Suivante
        # ==========================================================
        self.bouton_image_suivante.setEnabled(
            index_actuel < dernier_index
        )

        # ==========================================================
        # Position
        # ==========================================================
        self.label_position_image.setText(
            f"{index_actuel} / {dernier_index}"
        )

    def _charger_zones_annotations(self):
        """
        Recherche les zones disponibles dans le répertoire
        d'annotations sélectionné.
        """

        # ==========================================================
        # Désactivation temporaire des signaux
        # ==========================================================
        self.combo_zone.blockSignals(True)
        self.combo_image.blockSignals(True)

        # ==========================================================
        # Réinitialisation
        # ==========================================================
        self.combo_zone.clear()
        self.combo_image.clear()

        self.combo_zone.setEnabled(False)
        self.combo_image.setEnabled(False)

        self.liste_zones = []

        # ==========================================================
        # Vérification du répertoire
        # ==========================================================
        if self.repertoire_annotations is None:

            self.combo_zone.blockSignals(False)
            self.combo_image.blockSignals(False)

            return

        # ==========================================================
        # Recherche des zones
        # ==========================================================
        for sous_repertoire in sorted(
            self.repertoire_annotations.iterdir()
        ):

            if not sous_repertoire.is_dir():
                continue

            repertoire_images = (
                sous_repertoire
                / "images"
                / "Train2017"
            )

            repertoire_labels = (
                sous_repertoire
                / "labels"
                / "train2017"
            )

            # ------------------------------------------------------
            # Une zone est valide si les deux répertoires existent
            # ------------------------------------------------------
            if (
                repertoire_images.is_dir()
                and repertoire_labels.is_dir()
            ):
                self.liste_zones.append(
                    sous_repertoire
                )

        # ==========================================================
        # Alimentation de la ComboBox
        # ==========================================================
        if self.liste_zones:

            self.combo_zone.addItem(
                "Choisir une zone...",
                None
            )

            for repertoire_zone in self.liste_zones:

                self.combo_zone.addItem(
                    repertoire_zone.name,
                    repertoire_zone
                )

            self.combo_zone.setEnabled(True)

        else:

            self.combo_zone.addItem(
                "Aucune zone disponible",
                None
            )

            self.combo_zone.setEnabled(False)

        # ==========================================================
        # Réactivation des signaux
        # ==========================================================
        self.combo_zone.blockSignals(False)
        self.combo_image.blockSignals(False)


    def selectionner_zone(
        self,
        index
    ):
        """
        Charge les images disponibles dans la zone sélectionnée.
        """

        # ==========================================================
        # Récupération de la zone sélectionnée
        # ==========================================================
        repertoire_zone = self.combo_zone.itemData(
            index
        )

        # ==========================================================
        # Blocage temporaire du signal Image
        # ==========================================================
        self.combo_image.blockSignals(True)

        # ==========================================================
        # Réinitialisation
        # ==========================================================
        self.combo_image.clear()
        self.combo_image.setEnabled(False)

        self.bouton_image_precedente.setEnabled(False)
        self.bouton_image_suivante.setEnabled(False)

        self.liste_images_zone = []

        # ==========================================================
        # Réinitialisation de l'affichage
        # ==========================================================
        self._reinitialiser_affichage_image()


        # ==========================================================
        # Aucune zone réellement sélectionnée
        # ==========================================================
        if repertoire_zone is None:

            self.repertoire_zone = None
            self.repertoire_images_zone = None
            self.repertoire_labels_zone = None

            self.combo_image.addItem(
                "Sélectionnez une zone",
                None
            )

            self._reinitialiser_recap_controle()

            self.combo_image.blockSignals(False)

            return

        # ==========================================================
        # Zone sélectionnée
        # ==========================================================
        self.repertoire_zone = Path(
            repertoire_zone
        )

        # ==========================================================
        # Répertoire des images
        # ==========================================================
        self.repertoire_images_zone = (
            self.repertoire_zone
            / "images"
            / "Train2017"
        )

        # ==========================================================
        # Répertoire des labels
        # ==========================================================
        self.repertoire_labels_zone = (
            self.repertoire_zone
            / "labels"
            / "train2017"
        )

        # ==========================================================
        # Vérification du répertoire des images
        # ==========================================================
        if not self.repertoire_images_zone.is_dir():

            self.combo_image.addItem(
                "Répertoire images introuvable",
                None
            )

            self._reinitialiser_recap_controle()

            self.combo_image.blockSignals(False)

            return

        # ==========================================================
        # Recherche des images
        # ==========================================================
        self.liste_images_zone = sorted(
            fichier
            for fichier in self.repertoire_images_zone.iterdir()
            if (
                fichier.is_file()
                and fichier.suffix.lower() in cst.IMAGE_EXT
            )
        )

        # ==========================================================
        # Aucune image
        # ==========================================================
        if not self.liste_images_zone:

            self.combo_image.addItem(
                "Aucune image trouvée",
                None
            )

            self._reinitialiser_recap_controle()

            self.combo_image.blockSignals(False)

            return

        # ==========================================================
        # Images disponibles
        # ==========================================================
        self.combo_image.addItem(
            "Choisir une image...",
            None
        )

        for chemin_image in self.liste_images_zone:

            self.combo_image.addItem(
                chemin_image.name,
                chemin_image
            )

        self.combo_image.setEnabled(True)

        self.label_position_image.setText(
            f"0 / {len(self.liste_images_zone)}"
        )
     
        self._mettre_a_jour_recap_controle()


        # ==========================================================
        # Restauration du filtre mémorisé pour la zone
        # ==========================================================
        cle_zone = str(
            Path(self.repertoire_zone)
        )

        filtre_memorise = self.filtre_par_zone.get(
            cle_zone,
            "Toutes les images"
        )

        # ==========================================================
        # Restauration sans déclencher les signaux
        # ==========================================================
        self.combo_filtre_etat.blockSignals(True)

        self.combo_filtre_etat.setCurrentText(
            filtre_memorise
        )

        self.combo_filtre_etat.blockSignals(False)

        # ==========================================================
        # Application explicite du filtre
        # ==========================================================
        self._appliquer_filtre_images()

        # ==========================================================
        # Réactivation du signal
        # ==========================================================
        self.combo_image.blockSignals(False)


    def _reinitialiser_affichage_image(
            self,
            reinitialiser_compteur=True
        ):
        """
        Réinitialise toute la zone d'affichage lorsqu'aucune
        image n'est sélectionnée.
        """

        # ==========================================================
        # Image originale
        # ==========================================================
        self.label_image_originale.clear()

        self.label_image_originale.setText(
            "Aucune image sélectionnée"
        )

        # ==========================================================
        # Image annotée
        # ==========================================================
        self.label_image_annotee.clear()

        self.label_image_annotee.setText(
            "Aucune annotation à afficher"
        )

        # ==========================================================
        # Classes
        # ==========================================================
        self.label_classes_annotation.setText(
            "Aucune classe à afficher"
        )

        # ==========================================================
        # Fichier de labels
        # ==========================================================
        self.label_fichier_annotation.setText(
            "Fichier de labels : —"
        )

        # ==========================================================
        # Compteur
        # ==========================================================
        if reinitialiser_compteur:

            self.label_position_image.setText(
                "0 / 0"
            )

        # ==========================================================
        # Navigation
        # ==========================================================
        self.bouton_image_precedente.setEnabled(False)
        self.bouton_image_suivante.setEnabled(False)

        # ==========================================================
        # Boutons de contrôle
        # ==========================================================
        self.bouton_annotation_correcte.setChecked(False)
        self.bouton_annotation_verifier.setChecked(False)
        self.bouton_annotation_incorrecte.setChecked(False)

        self.bouton_annotation_correcte.setEnabled(False)
        self.bouton_annotation_verifier.setEnabled(False)
        self.bouton_annotation_incorrecte.setEnabled(False)


    def selectionner_image_zone(
        self,
        index
    ):
        """
        Gère la sélection d'une image dans la zone choisie.

        Affiche :
        - l'image originale à gauche ;
        - l'image avec les bounding boxes à droite.
        """

         # ==========================================================
        # Navigation
        # ==========================================================
        self._mettre_a_jour_navigation_images()



        # ==========================================================
        # Récupération de l'image sélectionnée
        # ==========================================================
        chemin_image = self.combo_image.itemData(
            index
        )

        # ==========================================================
        # Aucun choix valide
        # ==========================================================
        if chemin_image is None:

            self._reinitialiser_affichage_image(
                reinitialiser_compteur=False
            )

            return

        # ==========================================================
        # Activation des boutons de contrôle
        # ==========================================================
        self.bouton_annotation_correcte.setEnabled(True)
        self.bouton_annotation_verifier.setEnabled(True)
        self.bouton_annotation_incorrecte.setEnabled(True)

        # ==========================================================
        # Restauration de l'état mémorisé
        # ==========================================================
        self._mettre_a_jour_etat_controle()


        # ==========================================================
        # Conversion en Path
        # ==========================================================
        chemin_image = Path(chemin_image)

        # ==========================================================
        # Chargement de l'image originale
        # ==========================================================
        pixmap_original = QPixmap(str(chemin_image))

        # ==========================================================
        # Vérification
        # ==========================================================
        if pixmap_original.isNull():

            self.label_image_originale.clear()

            self.label_image_originale.setText(
                f"Impossible de charger l'image :\n"
                f"{chemin_image.name}"
            )

            self.label_image_annotee.clear()
            self.label_image_annotee.setText("Image indisponible")

            return

        # ==========================================================
        # Recherche du fichier de labels correspondant
        # ==========================================================
        chemin_label = (
            self.repertoire_labels_zone / f"{chemin_image.stem}.txt")

        # ==========================================================
        # Image originale
        # ==========================================================
        pixmap_original_affichage = pixmap_original.scaled(
            self.label_image_originale.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.label_image_originale.clear()

        self.label_image_originale.setPixmap(pixmap_original_affichage)

        # ==========================================================
        # Aucun fichier de labels
        # ==========================================================
        if not chemin_label.is_file():

            self.label_fichier_annotation.setText(
                f"Fichier de labels absent : "
                f"{chemin_image.stem}.txt"
            )

            self.label_image_annotee.clear()
            self.label_image_annotee.setText("Aucun label associé à cette image.")

            return

        # ==========================================================
        # Lecture des annotations YOLO
        # ==========================================================
        annotations = self._lire_labels_yolo(chemin_label)

        # ==========================================================
        # Information fichier
        # ==========================================================
        nombre_annotations = len(annotations)

        self.label_fichier_annotation.setText(
            f"Fichier de labels : {chemin_label.name} "
            f"— {nombre_annotations} annotation(s)"
        )

        # ==========================================================
        # Classes présentes dans l'image
        # ==========================================================
        classes_presentes = sorted(
            {
                annotation[0]
                for annotation in annotations
            }
        )

        lignes_classes = []

        for numero_classe in classes_presentes:

            nom_classe = self.classes_yolo.get(
                numero_classe,
                "Classe inconnue"
            )

            lignes_classes.append(
                f"<b>{numero_classe}</b><br>"
                f"{nom_classe}"
            )

        self.label_classes_annotation.setText(
            "<br><br>".join(lignes_classes)
        )



        # ==========================================================
        # Fichier présent mais vide
        # ==========================================================
        if not annotations:

            self.label_image_annotee.clear()

            self.label_image_annotee.setText(
                "Le fichier de labels ne contient "
                "aucune annotation."
            )

            return

        # ==========================================================
        # Redimensionnement de l'image avant dessin des BBox
        # ==========================================================
        pixmap_annote_affichage = pixmap_original.scaled(
            self.label_image_annotee.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # ==========================================================
        # Dessin des BBox directement à la taille d'affichage
        # ==========================================================
        pixmap_annote_affichage = self._dessiner_annotations_yolo(
            pixmap_annote_affichage,
            annotations
        )

        # ==========================================================
        # Affichage
        # ==========================================================
        self.label_image_annotee.clear()

        self.label_image_annotee.setPixmap(pixmap_annote_affichage)

    def _definir_etat_image(
        self,
        etat
    ):
        """
        Mémorise l'état de contrôle de l'image courante
        et actualise l'affichage.
        """

        # ==========================================================
        # Image actuellement sélectionnée
        # ==========================================================
        chemin_image = self.combo_image.currentData()

        if chemin_image is None:
            return

        # ==========================================================
        # Clé unique de l'image
        # ==========================================================
        cle_image = str(
            Path(chemin_image)
        )

        # ==========================================================
        # Mémorisation / remplacement de l'état
        # ==========================================================
        self.etat_images[cle_image] = etat

        # ==========================================================
        # Mise à jour de l'état visuel
        # ==========================================================
        self._mettre_a_jour_etat_controle()

        # ==========================================================
        # Mise à jour immédiate du récapitulatif
        # ==========================================================
        self._mettre_a_jour_recap_controle()

        # ==========================================================
        # Sauvegarde des états dans le fichier JSON
        # ==========================================================
        self._sauvegarder_etats_controle()

        # ==========================================================
        # Vérification du filtre courant
        # ==========================================================
        image_correspond_filtre = (
            self._image_correspond_filtre(
                chemin_image
            )
        )

        # ==========================================================
        # L'image ne correspond plus au filtre
        # ==========================================================
        if not image_correspond_filtre:

            # ======================================================
            # Mémorisation de la position actuelle
            # ======================================================
            index_actuel = self.combo_image.currentIndex()

            # ======================================================
            # Reconstruction de la liste filtrée
            # ======================================================
            self._appliquer_filtre_images()

            # ======================================================
            # Nombre d'éléments restant dans la ComboBox
            # index 0 = "Choisir une image..."
            # ======================================================
            nombre_elements = self.combo_image.count()

            # ======================================================
            # Sélection de l'image qui prend sa place
            # ======================================================
            if nombre_elements > 1:

                nouvel_index = min(
                    index_actuel,
                    nombre_elements - 1
                )

                self.combo_image.setCurrentIndex(
                    nouvel_index
                )

            return

        # ==========================================================
        # L'image appartient toujours au filtre
        # Passage automatique éventuel à l'image suivante
        # ==========================================================
        if self.check_image_suivante_auto.isChecked():

            self.image_suivante()
        

    def _chemin_fichier_controle(self):
        """
        Retourne le chemin du fichier JSON de contrôle.
        """

        if self.repertoire_annotations is None:
            return None

        nom_json = (
            f"{self.repertoire_annotations.name}"
            "_controle_annotations.json"
        )

        return (self.repertoire_annotations / nom_json)



    def _charger_etats_controle(self):
        """
        Recharge les états de contrôle depuis le fichier JSON.
        """

        self.etat_images = {}

        if self.repertoire_annotations is None:
            return

        chemin_json = self._chemin_fichier_controle()

        # ==========================================================
        # Premier contrôle : aucun JSON existant
        # ==========================================================
        if not chemin_json.is_file():
            return

        # ==========================================================
        # Lecture
        # ==========================================================
        try:

            with chemin_json.open("r", encoding="utf-8") as fichier:

                donnees = json.load(fichier)

        except (OSError, json.JSONDecodeError) as erreur:

            print(f"Erreur lecture contrôle : {erreur}")

            return

        # ==========================================================
        # Reconstruction des chemins complets
        # ==========================================================
        for zone, images in donnees.items():

            if not isinstance(images, dict):
                continue

            for nom_image, etat in images.items():

                chemin_image = (
                    self.repertoire_annotations
                    / zone
                    / "images"
                    / "Train2017"
                    / nom_image
                )

                self.etat_images[
                    str(chemin_image)
                ] = etat


    def _mettre_a_jour_etat_controle(self):
        """
        Restaure l'état visuel des boutons de contrôle
        pour l'image actuellement sélectionnée.
        """

        chemin_image = self.combo_image.currentData()

        # ==========================================================
        # Aucune image sélectionnée
        # ==========================================================
        if chemin_image is None:

            self.bouton_annotation_correcte.setChecked(False)
            self.bouton_annotation_verifier.setChecked(False)
            self.bouton_annotation_incorrecte.setChecked(False)

            self.bouton_annotation_correcte.setEnabled(False)
            self.bouton_annotation_verifier.setEnabled(False)
            self.bouton_annotation_incorrecte.setEnabled(False)

            return

        # ==========================================================
        # Image sélectionnée
        # ==========================================================
        self.bouton_annotation_correcte.setEnabled(True)
        self.bouton_annotation_verifier.setEnabled(True)
        self.bouton_annotation_incorrecte.setEnabled(True)

        cle_image = str(Path(chemin_image))

        # ==========================================================
        # Recherche de l'état mémorisé
        # ==========================================================
        etat = self.etat_images.get(cle_image)

        # ==========================================================
        # Etat visuel
        # ==========================================================
        self.bouton_annotation_correcte.setChecked(etat == "correcte")
        self.bouton_annotation_verifier.setChecked(etat == "a_verifier")
        self.bouton_annotation_incorrecte.setChecked(etat == "incorrecte")

    def _mettre_a_jour_recap_controle(self):
        """
        Met à jour le récapitulatif des états pour
        les images de la zone actuellement sélectionnée.
        """

        # ==========================================================
        # Compteurs
        # ==========================================================
        nombre_correctes = 0
        nombre_a_verifier = 0
        nombre_incorrectes = 0

        nombre_images = len(
            self.liste_images_zone
        )

        # ==========================================================
        # Parcours des images de la zone courante
        # ==========================================================
        for chemin_image in self.liste_images_zone:

            cle_image = str(
                Path(chemin_image)
            )

            etat = self.etat_images.get(
                cle_image
            )

            if etat == "correcte":

                nombre_correctes += 1

            elif etat == "a_verifier":

                nombre_a_verifier += 1

            elif etat == "incorrecte":

                nombre_incorrectes += 1

        # ==========================================================
        # Nombre d'images déjà contrôlées
        # ==========================================================
        nombre_controlees = (
            nombre_correctes
            + nombre_a_verifier
            + nombre_incorrectes
        )

        # ==========================================================
        # Images non contrôlées
        # ==========================================================
        nombre_non_controlees = (
            nombre_images
            - nombre_controlees
        )

        # ==========================================================
        # Avancement du contrôle
        # ==========================================================
        if nombre_images > 0:

            pourcentage_controle = (
                nombre_controlees
                / nombre_images
                * 100
            )

        else:

            pourcentage_controle = 0

        # ==========================================================
        # Affichage du récapitulatif
        # ==========================================================
        self.label_recap_controle.setText(
            f"✓ Correctes : {nombre_correctes}    "
            f"⚠ À vérifier : {nombre_a_verifier}    "
            f"✕ Incorrectes : {nombre_incorrectes}    "
            f"○ Non contrôlées : {nombre_non_controlees}"
        )

        # ==========================================================
        # Mise à jour de la barre de progression
        # ==========================================================
        self.progress_controle.setValue(
            round(pourcentage_controle)
        )

        self.progress_controle.setFormat(
            f"Avancement de la zone : "
            f"{nombre_controlees} / {nombre_images} "
            f"({pourcentage_controle:.0f} %)"
        )

    def _reinitialiser_recap_controle(self):
        """
        Réinitialise le récapitulatif et la progression
        du contrôle des annotations.
        """

        self.label_recap_controle.setText(
            "✓ Correctes : 0    "
            "⚠ À vérifier : 0    "
            "✕ Incorrectes : 0    "
            "○ Non contrôlées : 0"
        )

        self.progress_controle.setValue(0)

        self.progress_controle.setFormat(
            "Avancement de la zone : 0 / 0 (0 %)"
        )




    def _image_correspond_filtre(self, chemin_image):
        """
        Indique si une image correspond
        au filtre actuellement sélectionné.
        """

        # ==========================================================
        # Filtre actuellement sélectionné
        # ==========================================================
        filtre = self.combo_filtre_etat.currentText()

        # ==========================================================
        # Toutes les images
        # ==========================================================
        if filtre == "Toutes les images":
            return True

        # ==========================================================
        # Recherche de l'état de l'image
        # ==========================================================
        cle_image = str(
            Path(chemin_image)
        )

        etat = self.etat_images.get(
            cle_image
        )

        # ==========================================================
        # Application du filtre
        # ==========================================================
        if filtre == "Non contrôlées":
            return etat is None

        if filtre == "Correctes":
            return etat == "correcte"

        if filtre == "À vérifier":
            return etat == "a_verifier"

        if filtre == "Incorrectes":
            return etat == "incorrecte"

        return True

    def _memoriser_filtre_zone(self, filtre):
        """
        Mémorise le filtre sélectionné pour la zone courante.
        """

        # ==========================================================
        # Vérification de la zone courante
        # ==========================================================
        repertoire_zone = getattr(
            self,
            "repertoire_zone",
            None
        )

        if repertoire_zone is None:
            return

        # ==========================================================
        # Clé unique de la zone
        # ==========================================================
        cle_zone = str(
            Path(repertoire_zone)
        )

        # ==========================================================
        # Mémorisation du filtre
        # ==========================================================
        self.filtre_par_zone[cle_zone] = filtre



    def _appliquer_filtre_images(self):
        """
        Met à jour la liste déroulante des images
        selon le filtre sélectionné.
        """

        # ==========================================================
        # Aucune image dans la zone
        # ==========================================================
        if not self.liste_images_zone:
            return

        # ==========================================================
        # Images correspondant au filtre
        # ==========================================================
        images_filtrees = [
            chemin_image
            for chemin_image in self.liste_images_zone
            if self._image_correspond_filtre(chemin_image)
        ]

        # ==========================================================
        # Mise à jour de la ComboBox
        # ==========================================================
        self.combo_image.blockSignals(True)

        self.combo_image.clear()

        # Élément neutre en première position
        self.combo_image.addItem(
            "Choisir une image...",
            None
        )

        for chemin_image in images_filtrees:

            self.combo_image.addItem(
                Path(chemin_image).name,
                chemin_image
            )

        self.combo_image.blockSignals(False)

        # ==========================================================
        # Mise à jour du compteur après filtrage
        # ==========================================================
        self.label_position_image.setText(
            f"0 / {len(images_filtrees)}"
        )

        # ==========================================================
        # Réinitialisation de l'affichage
        # ==========================================================
        self._reinitialiser_affichage_image(
            reinitialiser_compteur=False
        )


        # ==========================================================
        # Activation de la ComboBox
        # ==========================================================
        self.combo_image.setEnabled(
            bool(images_filtrees)
        )


    def _charger_classes_yolo(
        self,
        repertoire
    ):
        """
        Recherche un fichier YAML dans le répertoire de travail
        et charge la correspondance entre les numéros de classes
        YOLO et leurs noms.

        Exemple :
            0 -> Arthr_Insec_Diptera_Simuliidae_Genus
            1 -> Arthr_Insec_Trichoptera_Hydropsychidae_Hydropsyche
        """

        self.classes_yolo = {}

        repertoire = Path(repertoire)

        # ==========================================================
        # Recherche du fichier YAML
        # ==========================================================
        fichiers_yaml = sorted(
            list(repertoire.glob("*.yaml"))
            + list(repertoire.glob("*.yml"))
        )

        # Aucun YAML
        if not fichiers_yaml:
            print(
                "Aucun fichier YAML trouvé dans :",
                repertoire
            )
            return

        # ==========================================================
        # Premier fichier YAML trouvé
        # ==========================================================
        chemin_yaml = fichiers_yaml[0]

        try:

            with chemin_yaml.open(
                "r",
                encoding="utf-8"
            ) as fichier:

                donnees = yaml.safe_load(
                    fichier
                )

        except (OSError, yaml.YAMLError) as erreur:

            print(
                f"Erreur de lecture YAML : {erreur}"
            )
            return

        # ==========================================================
        # Récupération des classes
        # ==========================================================
        if not isinstance(donnees, dict):

            print(
                "Structure YAML invalide :",
                chemin_yaml
            )
            return

        noms = donnees.get(
            "names"
        )

        if not isinstance(noms, dict):

            print(
                "Section 'names' absente ou invalide :",
                chemin_yaml
            )
            return

        # ==========================================================
        # Conversion des clés en entiers
        # ==========================================================
        try:

            self.classes_yolo = {
                int(numero): str(nom)
                for numero, nom in noms.items()
            }

        except (ValueError, TypeError):

            print(
                "Classes YAML invalides :",
                chemin_yaml
            )

            self.classes_yolo = {}

            return

        print(
            f"Classes YOLO chargées : "
            f"{len(self.classes_yolo)}"
        )

        print(
            f"Fichier YAML : {chemin_yaml.name}"
        )



    def _lire_labels_yolo(
        self,
        chemin_label
    ):
        """
        Lit un fichier de labels au format YOLO.

        Format attendu :
            classe x_centre y_centre largeur hauteur

        Les coordonnées sont normalisées entre 0 et 1.

        Retourne :
            liste de tuples :
            (classe, x_centre, y_centre, largeur, hauteur)
        """

        annotations = []

        # ==========================================================
        # Vérification du fichier
        # ==========================================================
        if not chemin_label.is_file():
            return annotations

        # ==========================================================
        # Lecture
        # ==========================================================
        try:

            with chemin_label.open(
                "r",
                encoding="utf-8"
            ) as fichier:

                for numero_ligne, ligne in enumerate(
                    fichier,
                    start=1
                ):

                    ligne = ligne.strip()

                    # Ligne vide
                    if not ligne:
                        continue

                    elements = ligne.split()

                    # Format YOLO attendu : 5 valeurs
                    if len(elements) != 5:
                        print(
                            f"Label YOLO invalide : "
                            f"{chemin_label.name}, "
                            f"ligne {numero_ligne}"
                        )
                        continue

                    try:

                        classe = int(elements[0])

                        x_centre = float(elements[1])
                        y_centre = float(elements[2])
                        largeur = float(elements[3])
                        hauteur = float(elements[4])

                    except ValueError:

                        print(
                            f"Valeur YOLO invalide : "
                            f"{chemin_label.name}, "
                            f"ligne {numero_ligne}"
                        )

                        continue

                    annotations.append(
                        (
                            classe,
                            x_centre,
                            y_centre,
                            largeur,
                            hauteur
                        )
                    )

        except OSError as erreur:

            print(
                f"Erreur de lecture du fichier "
                f"{chemin_label} : {erreur}"
            )

        return annotations

    def _dessiner_annotations_yolo(
        self,
        pixmap,
        annotations
    ):
        """
        Dessine les bounding boxes YOLO sur une copie
        du QPixmap original.

        Retourne le QPixmap annoté.
        """

        # ==========================================================
        # Copie de l'image
        # ==========================================================
        pixmap_annote = pixmap.copy()

        largeur_image = pixmap_annote.width()
        hauteur_image = pixmap_annote.height()

        # ==========================================================
        # Vérification dimensions
        # ==========================================================
        if largeur_image <= 0 or hauteur_image <= 0:
            return pixmap_annote

        # ==========================================================
        # Création du Painter
        # ==========================================================
        painter = QPainter(
            pixmap_annote
        )

        painter.setRenderHint(
            QPainter.Antialiasing,
            True
        )

        # ==========================================================
        # Crayon des bounding boxes
        # ==========================================================
        pen_bbox = QPen(
            QColor(COULEUR_BBOX_ROUGE)
        )

        # Une ligne assez épaisse pour rester visible
        # après réduction de l'image dans l'IHM
        pen_bbox.setWidth(6)

        painter.setPen(
            pen_bbox
        )

        # ==========================================================
        # Police du numéro de classe
        # ==========================================================
        font_classe = QFont()

        font_classe.setBold(True)
        font_classe.setPointSize(9)

        painter.setFont(
            font_classe
        )



        # ==========================================================
        # Parcours des annotations
        # ==========================================================
        for annotation in annotations:

            (
                classe,
                x_centre,
                y_centre,
                largeur,
                hauteur
            ) = annotation

            # ======================================================
            # Conversion YOLO -> pixels
            # ======================================================
            centre_x_pixels = (
                x_centre * largeur_image
            )

            centre_y_pixels = (
                y_centre * hauteur_image
            )

            largeur_pixels = (
                largeur * largeur_image
            )

            hauteur_pixels = (
                hauteur * hauteur_image
            )

            # ======================================================
            # Coin supérieur gauche
            # ======================================================
            x1 = (
                centre_x_pixels
                - largeur_pixels / 2
            )

            y1 = (
                centre_y_pixels
                - hauteur_pixels / 2
            )


            painter.drawRect(
                int(round(x1)),
                int(round(y1)),
                int(round(largeur_pixels)),
                int(round(hauteur_pixels))
            )

            # ==========================================================
            # Numéro de classe
            # ==========================================================
            texte_classe = str(
                classe
            )

            painter.drawText(
                int(round(x1)),
                max(
                    15,
                    int(round(y1)) - 5
                ),
                texte_classe
            )



        # ==========================================================
        # Fin du Painter
        # ==========================================================
        painter.end()

        return pixmap_annote


    def _sauvegarder_etats_controle(self):
        """
        Sauvegarde les états de contrôle dans un fichier JSON
        situé dans le répertoire des annotations.
        """

        if self.repertoire_annotations is None:
            return

        chemin_json = self._chemin_fichier_controle() 

        # ==========================================================
        # Réorganisation par zone
        # ==========================================================
        donnees = {}

        for chemin_image_str, etat in self.etat_images.items():

            chemin_image = Path(
                chemin_image_str
            )

            try:
                chemin_relatif = chemin_image.relative_to(
                    self.repertoire_annotations
                )

            except ValueError:
                continue

            # Exemple :
            # zone_1/images/Train2017/image.jpg
            if len(chemin_relatif.parts) < 2:
                continue

            zone = chemin_relatif.parts[0]
            nom_image = chemin_image.name

            if zone not in donnees:
                donnees[zone] = {}

            donnees[zone][nom_image] = etat

        # ==========================================================
        # Sauvegarde
        # ==========================================================
        try:

            with chemin_json.open(
                "w",
                encoding="utf-8"
            ) as fichier:

                json.dump(
                    donnees,
                    fichier,
                    indent=4,
                    ensure_ascii=False
                )

        except OSError as erreur:

            print(
                f"Erreur sauvegarde contrôle : {erreur}"
            )

    def _creer_zone_sortie(
        self,
        layout_principal
    ):
        """
        Crée le bouton permettant de quitter l'application.
        Le bouton est placé en bas à droite de l'interface.
        """

        # ==========================================================
        # Layout
        # ==========================================================
        layout_sortie = QHBoxLayout()

        layout_sortie.setContentsMargins(
            0,
            10,
            0,
            5
        )

        # ==========================================================
        # Bouton Quitter
        # ==========================================================
        self.bouton_quitter = QPushButton(
            "Quitter"
        )

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

        # ==========================================================
        # Placement à droite
        # ==========================================================
        layout_sortie.addStretch()

        layout_sortie.addWidget(
            self.bouton_quitter
        )

        layout_principal.addLayout(
            layout_sortie
        )


    def _connecter_signaux(self):
        """
        Connecte les signaux de l'interface
        aux méthodes correspondantes.
        """

        # ----------------------------------------------------------
        # Sélection du répertoire
        # ----------------------------------------------------------
        self.bouton_parcourir.clicked.connect(
            self.selectionner_repertoire
        )

        # ----------------------------------------------------------
        # Lancement de la détection
        # ----------------------------------------------------------
        self.bouton_lancer_detection.clicked.connect(
            self.lancer_detection
        )

        # ----------------------------------------------------------
        # Sélection du répertoire des annotations
        # ----------------------------------------------------------
        self.bouton_parcourir_annotations.clicked.connect(
            self.selectionner_repertoire_annotations
        )

        # ----------------------------------------------------------
        # Sélection d'une zone
        # ----------------------------------------------------------
        self.combo_zone.currentIndexChanged.connect(
            self.selectionner_zone
        )

        # ----------------------------------------------------------
        # Filtrage des images
        # ----------------------------------------------------------
        self.combo_filtre_etat.currentTextChanged.connect(
            self._appliquer_filtre_images
        )

        # ----------------------------------------------------------
        # Mémorisation du filtre par zone
        # ----------------------------------------------------------
        self.combo_filtre_etat.currentTextChanged.connect(
            self._memoriser_filtre_zone
        )




        # ----------------------------------------------------------
        # Image précédente
        # ----------------------------------------------------------
        self.bouton_image_precedente.clicked.connect(
            self.image_precedente
        )

        # ----------------------------------------------------------
        # Image suivante
        # ----------------------------------------------------------
        self.bouton_image_suivante.clicked.connect(
            self.image_suivante
        )


        # ----------------------------------------------------------
        # Sélection d'une image
        # ----------------------------------------------------------
        self.combo_image.currentIndexChanged.connect(
            self.selectionner_image_zone
        )

        # ----------------------------------------------------------
        # Annotation correcte
        # ----------------------------------------------------------
        self.bouton_annotation_correcte.clicked.connect(
            lambda: self._definir_etat_image(
                "correcte"
            )
        )

        # ----------------------------------------------------------
        # Annotation à vérifier
        # ----------------------------------------------------------
        self.bouton_annotation_verifier.clicked.connect(
            lambda: self._definir_etat_image(
                "a_verifier"
            )
        )

        # ----------------------------------------------------------
        # Annotation incorrecte
        # ----------------------------------------------------------
        self.bouton_annotation_incorrecte.clicked.connect(
            lambda: self._definir_etat_image(
                "incorrecte"
            )
        )




        # ----------------------------------------------------------
        # Fermeture de l'application
        # ----------------------------------------------------------
        self.bouton_quitter.clicked.connect(
            self.close
        )


    

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
# Construction l'interface graphique principale
#----------------------------------------------------------------------------------------

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

        self._creer_selection_repertoire(layout_principal)

        self._creer_zone_execution(layout_principal)

        # Consultation des résultats de la détection
        self._creer_consultation_annotations(layout_principal)

        self._creer_zone_sortie(layout_principal)

        # ==========================================================
        # Signaux
        # ==========================================================
        self._connecter_signaux()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def _traiter_image_suivante(self):
        """
        Simule le traitement successif des images.
        """

        # ==========================================================
        # Fin de tous les répertoires
        # ==========================================================
        if self.index_repertoire >= len(self.liste_repertoires):
            self._terminer_detection()
            return

        repertoire_courant = self.liste_repertoires[self.index_repertoire]

        self.label_repertoire_courant.setText(f"Répertoire : {repertoire_courant.name}")

        images_courantes = self.images_par_repertoire[repertoire_courant]


        # ==========================================================
        # Initialisation du répertoire courant
        # ==========================================================
        if self.index_image == 0:
            self.progression_images.setRange(0, len(images_courantes))
            self.progression_images.setValue(0)

        # ==========================================================
        # Traitement d'une image
        # ==========================================================
        if self.index_image < len(images_courantes):
            image_courante = images_courantes[self.index_image]
            self.label_image_courante.setText(f"Image : {image_courante.name}")

            # TODO
            # ------------------------------------------------------
            # La future fonction de détection sera appelée ici
            # ------------------------------------------------------

            self.index_image += 1

            self.progression_images.setValue(self.index_image)

            # Délai uniquement destiné à rendre la simulation visible
            QTimer.singleShot(
                100,
                self._traiter_image_suivante
            )

            return

        # ==========================================================
        # Passage au répertoire suivant
        # ==========================================================
        self.index_repertoire += 1

        self.progression_repertoires.setValue(self.index_repertoire)
        self.index_image = 0
        QTimer.singleShot(0, self._traiter_image_suivante)


    def _terminer_detection(self):
        """
        Termine le traitement et réactive les commandes.
        """

        self.bouton_lancer_detection.setEnabled(True)
        self.bouton_parcourir.setEnabled(True)
        self.label_etat_detection.setText("Détection terminée.")
        self.label_etat_detection.setStyleSheet(
            f"""
            QLabel {{
                color: {COULEUR_SUCCES_TEXTE};
                background-color: {COULEUR_SUCCES_FOND};
                border: 1px solid {COULEUR_SUCCES_BORDURE};
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }}
            """
        )


    def lancer_detection(self):
        """
        Lance provisoirement la simulation du traitement.
        Cette méthode sera remplacée par le véritable processus
        de détection.
        """

        if not self.liste_images:
            return

        # ==========================================================
        # Préparation
        # ==========================================================
        self.bouton_lancer_detection.setEnabled(False)
        self.bouton_parcourir.setEnabled(False)

        self.progression_repertoires.setValue(0)
        self.progression_images.setValue(0)

        # Création des groupes d'images par répertoire
        self.images_par_repertoire = {
            repertoire: [
                image
                for image in self.liste_images
                if image.parent == repertoire
            ]
            for repertoire in self.liste_repertoires
        }

        self.index_repertoire = 0
        self.index_image = 0

        self.label_etat_detection.setText(
            "Détection en cours..."
        )

        # Lance la première étape
        QTimer.singleShot(
            0,
            self._traiter_image_suivante
        )


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
def main():

    application = QApplication(sys.argv)

    interface = IhmAquaIa()

    interface.show()

    sys.exit(application.exec())

#==================================================================================================
if __name__ == "__main__":
    main()

    