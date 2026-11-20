import sys
import numpy as np
from pathlib import Path



from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QImage,
    QPixmap,
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

)


from config import constants as cst
from tools import utility as util



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


#==================================================================================================
#==================================================================================================

class IhmAquaIa(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aqua-IA - Détection")
        self.resize(900, 650)

        self.etat_images = {}

        self.repertoire_selectionne = None
        self.liste_images = []
        self.liste_repertoires = []



        self._creer_interface()


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


#========================================================================================
#  Fonctions génériques 
#========================================================================================

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
        # Style commun des barres de progression
        # ==========================================================
        style_progression = f"""
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
            style_progression
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
            style_progression
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
        # Fermeture de l'application
        # ----------------------------------------------------------
        self.bouton_quitter.clicked.connect(
            self.close
        )




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



        self._creer_zone_sortie(layout_principal)

        # ==========================================================
        # Signaux
        # ==========================================================
        self._connecter_signaux()



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#           Pour les Tests 
#               A Suprimer

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

        repertoire_courant = self.liste_repertoires[
            self.index_repertoire
        ]

        self.label_repertoire_courant.setText(
            f"Répertoire : {repertoire_courant.name}"
        )


        images_courantes = self.images_par_repertoire[
            repertoire_courant
        ]



        # ==========================================================
        # Initialisation du répertoire courant
        # ==========================================================
        if self.index_image == 0:
            self.progression_images.setRange(
                0,
                len(images_courantes)
            )

            self.progression_images.setValue(0)

        # ==========================================================
        # Traitement d'une image
        # ==========================================================
        if self.index_image < len(images_courantes):
            image_courante = images_courantes[
                self.index_image
            ]

            self.label_image_courante.setText(
                f"Image : {image_courante.name}"
            )


            # ------------------------------------------------------
            # La future fonction de détection sera appelée ici
            # ------------------------------------------------------

            self.index_image += 1

            self.progression_images.setValue(
                self.index_image
            )

            # Délai uniquement destiné à rendre la simulation visible
            QTimer.singleShot(
                40,
                self._traiter_image_suivante
            )

            return

        # ==========================================================
        # Passage au répertoire suivant
        # ==========================================================
        self.index_repertoire += 1

        self.progression_repertoires.setValue(
            self.index_repertoire
        )

        self.index_image = 0

        QTimer.singleShot(
            0,
            self._traiter_image_suivante
        )


    def _terminer_detection(self):
        """
        Termine le traitement et réactive les commandes.
        """

        self.bouton_lancer_detection.setEnabled(True)
        self.bouton_parcourir.setEnabled(True)

        self.label_etat_detection.setText(
            "Détection terminée."
        )

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

    