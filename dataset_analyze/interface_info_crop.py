import sys
import numpy as np
from pathlib import Path


import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QImage,
    QPixmap,
)

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout, #  A supprimer si non utilisé
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QCheckBox,
)


from config import constants as cst
from tools import utility as util
from moteur_info_crop import (
    ErreurLectureImage,
    lire_image, 
    couleur_fond,
    selectionner_composante_principale,
    calculer_seuil_depuis_difference,
    creer_masque_bestiole,
    detecter_zone_depuis_masque,
    ajouter_marge_bbox,
    calculer_occupation_depuis_masque,
    calculer_occupation_bbox,
    sauvegarder_bbox,
    sauvegarder_liste_erreurs,

)





#==================================================================================================

# =================================================================================================
# PARAMÈTRES
# ==================================================================================================

SEUIL_DETECTION = None
pas_seuil = 5
TAILLE_BORD = 20
SURFACE_MIN_PCT = 0.001

MARGE_BBOX_PCT = 0.03
MARGE_BBOX_MIN_PX = 5

# ==================================================================================================
# ÉTATS DES IMAGES
# ==================================================================================================

ETAT_A_TRAITER = "a_traiter"
ETAT_SAUVEGARDEE = "sauvegardee"
ETAT_SANS_SAUVEGARDE = "sans_sauvegarde"
ETAT_SANS_DETECTION = "sans_detection"
ETAT_ERREUR_LECTURE = "erreur_lecture"


SYMBOLES_ETAT = {
    ETAT_A_TRAITER: "○",
    ETAT_SAUVEGARDEE: "✓",
    ETAT_SANS_SAUVEGARDE: "─",
    ETAT_SANS_DETECTION: "⚠",
    ETAT_ERREUR_LECTURE: "⚠",
}

LIBELLES_ETAT = {
    ETAT_A_TRAITER: "",
    ETAT_SAUVEGARDEE: "Sauvegardée",
    ETAT_SANS_SAUVEGARDE: "Sans sauvegarde",
    ETAT_SANS_DETECTION: "Sans détection",
    ETAT_ERREUR_LECTURE: "Erreur de lecture",
}



#==================================================================================================

class InterfaceInfoCrop(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Aqua-IA - Info & Crop d'images"
        )

        self.resize(
            1200,
            800
        )


        # ----------------------------------------------------------
        # Image et BBox actuellement affichées
        # ----------------------------------------------------------
        self.image_courante = None
        self.chemin_image_courante = None

        self.bbox_detectee_courante = None
        self.bbox_securisee_courante = None

        self.masque_courant = None


        self.etat_images = {}

        self._creer_interface()

    def _creer_interface(self):

        widget_principal = QWidget()

        self.setCentralWidget(widget_principal)

        layout_principal = QVBoxLayout(widget_principal)

        
        # ----------------------------------------------------------
        # Titre principal
        # ----------------------------------------------------------

        titre = QLabel(
            "AQUA-IA"
        )

        titre.setAlignment(
            Qt.AlignCenter
        )

        titre.setStyleSheet(
            """
            QLabel {
                color: rgb(0, 102, 102);
                font-size: 24px;
                font-weight: bold;
                padding-top: 5px;
            }
            """
        )

        layout_principal.addWidget(
            titre
        )


        # ----------------------------------------------------------
        # Sous-titre
        # ----------------------------------------------------------

        sous_titre = QLabel(
            "Analyse et recadrage d'images"
        )

        sous_titre.setAlignment(
            Qt.AlignCenter
        )

        sous_titre.setStyleSheet(
            """
            QLabel {
                color: #505050;
                font-size: 14px;
                padding-bottom: 8px;
            }
            """
        )

        layout_principal.addWidget(
            sous_titre
        )

        # ---- Sélection du répertoire et des images ----
        # ----------------------------------------------------------
        # Sélection du répertoire et de l'image
        # ----------------------------------------------------------

        layout_selection = QGridLayout()

        # ----------------------------------------------------------
        # Répertoire
        # ----------------------------------------------------------

        label_repertoire = QLabel(
            "Répertoire :"
        )

        self.champ_repertoire = QLineEdit()

        self.champ_repertoire.setReadOnly(
            True
        )

        self.champ_repertoire.setPlaceholderText(
            "Sélectionnez le répertoire contenant les images"
        )

        self.bouton_parcourir = QPushButton(
            "Parcourir..."
        )

        # ----------------------------------------------------------
        # Image
        # ----------------------------------------------------------

        label_image = QLabel(
            "Image :"
        )

        self.combo_images = QComboBox()

        self.combo_images.setEnabled(
            False
        )

        # ----------------------------------------------------------
        # Placement dans la grille
        # ----------------------------------------------------------

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

        layout_selection.addWidget(
            label_image,
            1,
            0
        )

        layout_selection.addWidget(
            self.combo_images,
            1,
            1,
            1,
            2
        )

        # ----------------------------------------------------------
        # Gestion de la largeur des colonnes
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Ajout dans le layout principal
        # ----------------------------------------------------------

        layout_principal.addLayout(
            layout_selection
        )




        # ----------------------------------------------------------
        # Options d'affichage
        # ----------------------------------------------------------

        self.checkbox_diagnostics = QCheckBox(
            "Afficher les diagnostics intermédiaires"
        )

        self.checkbox_diagnostics.setChecked(
            False
        )

        self.checkbox_diagnostics.setStyleSheet(
            """
            QCheckBox {
                color: #006666;
                font-weight: bold;
                padding: 4px;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            """
        )

        layout_principal.addWidget(
            self.checkbox_diagnostics,
            alignment=Qt.AlignLeft
        )


        # ----------------------------------------------------------
        # Affichage optionnel du masque
        # ----------------------------------------------------------

        self.affichage_masque = QLabel(
            "Aucun masque disponible"
        )

        self.affichage_masque.setAlignment(
            Qt.AlignCenter
        )

        self.affichage_masque.setMinimumSize(
            600,
            300
        )

        self.affichage_masque.setStyleSheet(
            """
            QLabel {
                background-color: #202020;
                color: #D0D0D0;
                border: 1px solid #808080;
            }
            """
        )

        # Masqué au démarrage
        self.affichage_masque.setVisible(
            False
        )

  


        # ----------------------------------------------------------
        # Résumé du traitement
        # ----------------------------------------------------------

        self.resume_traitement = QLabel(
            "Images : 0    |    "
            "✓ Sauvegardées : 0    |    "
            "─ Sans sauvegarde : 0    |    "
            "⚠ Anomalies : 0    |    "
            "○ Restantes : 0"
        )

        self.resume_traitement.setAlignment(
            Qt.AlignCenter
        )

        self.resume_traitement.setStyleSheet(
            """
            QLabel {
                padding: 6px;
                font-weight: bold;
            }
            """
        )

        layout_principal.addWidget(
            self.resume_traitement
        )


        # ----------------------------------------------------------
        # Sélection de l'image
        # ----------------------------------------------------------
       

        # ----------------------------------------------------------
        # Affichage de l'image
        # ----------------------------------------------------------
        self.affichage_image = QLabel(
            "Sélectionnez une image"
        )

        self.affichage_image.setAlignment(
            Qt.AlignCenter
        )

        self.affichage_image.setMinimumSize(
            600,
            400
        )

        self.affichage_image.setStyleSheet(
            """
            QLabel {
                background-color: #202020;
                color: #D0D0D0;
                border: 1px solid #808080;
            }
            """
        )

        # Ajout seulement après la création du QLabel
        layout_principal.addWidget(
            self.affichage_image
        )

        layout_principal.addWidget(
            self.affichage_masque
        )



        # ----------------------------------------------------------
        # Légende des Bounding Boxes
        # ----------------------------------------------------------

        layout_legende = QHBoxLayout()

        legende_rouge = QLabel(
            '<span style="color:#CC0000; font-size:16px;">■</span>'
            '&nbsp;&nbsp;BBox rouge : zone détectée'
        )

        legende_verte = QLabel(
            '<span style="color:#009600; font-size:16px;">■</span>'
            '&nbsp;&nbsp;BBox verte : zone sécurisée avec marge'
        )

        legende_rouge.setStyleSheet(
            """
            QLabel {
                color: #404040;
                font-weight: bold;
                padding: 4px;
            }
            """
        )

        legende_verte.setStyleSheet(
            """
            QLabel {
                color: #404040;
                font-weight: bold;
                padding: 4px;
            }
            """
        )


        layout_legende.addStretch()

        layout_legende.addWidget(
            legende_rouge
        )

        layout_legende.addSpacing(
            30
        )

        layout_legende.addWidget(
            legende_verte
        )

        layout_legende.addStretch()

        layout_principal.addLayout(
            layout_legende
        )





        # ----------------------------------------------------------
        # Informations de détection
        # ----------------------------------------------------------
        self.informations_detection = QLabel(
            "Détection : -    |    Seuil : -    |    Fond BGR : -"
        )

        self.informations_detection.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )

        self.informations_detection.setStyleSheet(
            """
            QLabel {
                padding: 6px;
                font-weight: bold;
            }
            """
        )

        layout_principal.addWidget(
            self.informations_detection
        )        


        # ----------------------------------------------------------
        # Informations d'occupation
        # ----------------------------------------------------------
        self.informations_occupation = QLabel(
            "Surface image : -    |    "
            "Surface bestiole : -    |    "
            "Occupation réelle : -    |    "
            "Occupation BBox : -"
        )

        self.informations_occupation.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )

        self.informations_occupation.setStyleSheet(
            """
            QLabel {
                padding: 6px;
                font-weight: bold;
            }
            """
        )

        layout_principal.addWidget(
            self.informations_occupation
        )


        layout_principal.addWidget(
            self.informations_occupation
        )


        # ----------------------------------------------------------
        # Choix de sauvegarde
        # ----------------------------------------------------------
        layout_sauvegarde = QHBoxLayout()

        self.bouton_bbox_rouge = QPushButton(
            "Sauvegarder BBox rouge"
        )

        self.bouton_bbox_rouge.setStyleSheet(
            """
            QPushButton {
                background-color: #00CC99;
                color: #003333;
                border: 1px solid #009977;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #66FFCC;
            }

            QPushButton:pressed {
                background-color: #009977;
                color: white;
            }

            QPushButton:disabled {
                background-color: #D0D0D0;
                color: #888888;
                border-color: #BBBBBB;
            }
            """
        )


        self.bouton_bbox_verte = QPushButton(
            "Sauvegarder BBox verte"
        )

        self.bouton_bbox_verte.setStyleSheet(
            """
            QPushButton {
                background-color: #006666;
                color: white;
                border: 1px solid #004C4C;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #00CC99;
                color: #003333;
            }

            QPushButton:pressed {
                background-color: #004C4C;
                color: white;
            }

            QPushButton:disabled {
                background-color: #D0D0D0;
                color: #888888;
                border-color: #BBBBBB;
            }
            """
        )



        self.bouton_sans_sauvegarde = QPushButton(
            "Ne pas sauvegarder"
        )


        self.bouton_sans_sauvegarde.setStyleSheet(
            """
            QPushButton {
                background-color: #707070;
                color: white;
                border: 1px solid #505050;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #909090;
            }

            QPushButton:pressed {
                background-color: #505050;
            }

            QPushButton:disabled {
                background-color: #D0D0D0;
                color: #888888;
                border-color: #BBBBBB;
            }
            """
        )


        layout_sauvegarde.addStretch()

        layout_sauvegarde.addWidget(
            self.bouton_bbox_rouge
        )

        layout_sauvegarde.addWidget(
            self.bouton_bbox_verte
        )

        layout_sauvegarde.addWidget(
            self.bouton_sans_sauvegarde
        )

        layout_sauvegarde.addStretch()

        layout_principal.addLayout(
            layout_sauvegarde
        )

        self.bouton_bbox_rouge.clicked.connect(
            self.sauvegarder_bbox_rouge
        )

        self.bouton_bbox_verte.clicked.connect(
            self.sauvegarder_bbox_verte
        )

        self.bouton_sans_sauvegarde.clicked.connect(
            self.ne_pas_sauvegarder
        )


        # ----------------------------------------------------------
        # Connexion du bouton
        # ----------------------------------------------------------
        self.bouton_parcourir.clicked.connect(
            self.selectionner_repertoire
        )

        self.combo_images.currentIndexChanged.connect(
            self.selectionner_image
        )


        self.checkbox_diagnostics.toggled.connect(
            self.afficher_diagnostics
        )



        # ----------------------------------------------------------
        # Espace restant
        # ----------------------------------------------------------
        layout_principal.addStretch()

    def selectionner_repertoire(self):

        repertoire = QFileDialog.getExistingDirectory(
            self,
            "Sélectionner le répertoire contenant les images"
        )

        if not repertoire:
            return

        # Conversion en Path
        repertoire = Path(
            repertoire
        )

        # Affichage du répertoire
        self.champ_repertoire.setText(
            str(repertoire)
        )

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


        # ----------------------------------------------------------
        # Aucune image
        # ----------------------------------------------------------
        if not liste_images:

            self.combo_images.setEnabled(False)
            self.mettre_a_jour_resume()

            return


        # ----------------------------------------------------------
        # Remplissage du menu
        # ----------------------------------------------------------
        self.etat_images.clear()

        for image in liste_images:

            self.etat_images[image] = (ETAT_A_TRAITER)

            self.combo_images.addItem(
                f"{SYMBOLES_ETAT[ETAT_A_TRAITER]}  {image.name}",
                image
            )

        self.combo_images.setEnabled(True)
        self.mettre_a_jour_resume()

    def selectionner_image(
        self,
        index
    ):

        if index < 0:
            return

        chemin_image = self.combo_images.itemData(index)

        if chemin_image is None:
            return

        print(f"Image sélectionnée : {chemin_image}")
        self.afficher_image(chemin_image)


    def afficher_image(
        self,
        chemin_image
    ):

        # ----------------------------------------------------------
        # Réinitialisation de l'image courante
        # ----------------------------------------------------------

        self.image_courante = None
        self.chemin_image_courante = None

        self.masque_courant = None

        self.bbox_detectee_courante = None
        self.bbox_securisee_courante = None

        self.bouton_bbox_rouge.setEnabled(False)
        self.bouton_bbox_verte.setEnabled(False)
        self.bouton_sans_sauvegarde.setEnabled(False)

        # ----------------------------------------------------------
        # Réinitialisation des diagnostics
        # ----------------------------------------------------------

        self.masque_courant = None

        self.affichage_masque.clear()

        if self.checkbox_diagnostics.isChecked():

            self.affichage_masque.setText(
                "Chargement du masque..."
            )




        # ----------------------------------------------------------
        # Lecture de l'image
        # ----------------------------------------------------------
        try:

            image = lire_image(chemin_image)

        except ErreurLectureImage as erreur:

            self.affichage_image.clear()

            self.affichage_image.setText(
                str(erreur)
            )

            # ------------------------------------------------------
            # Aucun masque pour cette image
            # ------------------------------------------------------

            self.masque_courant = None

            self.affichage_masque.clear()

            if self.checkbox_diagnostics.isChecked():

                self.affichage_masque.setText(
                    "Aucun masque disponible\n"
                    "Erreur de lecture de l'image"
                )

            # Votre gestion existante
            self.chemin_image_courante = Path(
                chemin_image
            )

            self.definir_etat_image(
                chemin_image,
                ETAT_ERREUR_LECTURE
            )

            self.bouton_bbox_rouge.setEnabled(
                False
            )

            self.bouton_bbox_verte.setEnabled(
                False
            )

            self.bouton_sans_sauvegarde.setEnabled(
                True
            )

            print(
                f"Erreur : {erreur}"
            )

            return



        # ----------------------------------------------------------
        # Création du masque
        # ----------------------------------------------------------
        masque_bestiole, fond, difference, seuil_utilise = (
            creer_masque_bestiole(
                voir_image=False,
                rep=chemin_image.parent,
                image=image,
                seuil=SEUIL_DETECTION,
                taille_bord=TAILLE_BORD,
                surface_min_pct=SURFACE_MIN_PCT,
                afficher_informations=False
            )
        )

        self.masque_courant = (
            masque_bestiole
        )

        if self.checkbox_diagnostics.isChecked():

            self.mettre_a_jour_masque()


        # ----------------------------------------------------------
        # Détection de la BBox
        # ----------------------------------------------------------

        bbox_detectee = detecter_zone_depuis_masque(
            masque_bestiole
        )

        # ----------------------------------------------------------
        # BBox sécurisée
        # ----------------------------------------------------------

        bbox_securisee = ajouter_marge_bbox(
            image=image,
            bbox=bbox_detectee,
            marge_pct=MARGE_BBOX_PCT,
            marge_min_px=MARGE_BBOX_MIN_PX
        )

        # ----------------------------------------------------------
        # Mémorisation du résultat courant
        # ----------------------------------------------------------

        self.image_courante = image

        self.chemin_image_courante = Path(chemin_image)

        self.bbox_detectee_courante = (bbox_detectee)

        self.bbox_securisee_courante = (bbox_securisee)

        if bbox_detectee is not None:

            self.bouton_bbox_rouge.setEnabled(True)
            self.bouton_bbox_verte.setEnabled(True)
            self.bouton_sans_sauvegarde.setEnabled(True)

        else:

            self.bouton_bbox_rouge.setEnabled(False)
            self.bouton_bbox_verte.setEnabled(False)
            self.bouton_sans_sauvegarde.setEnabled(True)

            self.definir_etat_image(chemin_image, ETAT_SANS_DETECTION)


        # ----------------------------------------------------------
        # Calcul de l'occupation
        # ----------------------------------------------------------
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


        # ----------------------------------------------------------
        # Informations de détection
        # ----------------------------------------------------------
        
        detection  = "Oui" if bbox_detectee is not None else "NON"
        texte_detection = (
                f"Détection : {detection}    |    "
                f"Seuil : {seuil_utilise:.3f}    |    "
                f"Fond BGR : {fond}"
            )

        self.informations_detection.setText(
            texte_detection
        )


        # ----------------------------------------------------------
        # Affichage des informations d'occupation
        # ----------------------------------------------------------
        surface_image_form    = util.format_nombre(surface_image)
        surface_bestiole_form = util.format_nombre(surface_bestiole)

        texte_occupation = (
            f"Surface image : {surface_image_form} px    |    "
            f"Surface bestiole : {surface_bestiole_form} px    |    "
            f"Occupation réelle : {occupation_pct:.1f} %    |    "
            f"Occupation BBox : {occupation_bbox_pct:.1f} %"
        )

        self.informations_occupation.setText(
            texte_occupation
        )


        # ----------------------------------------------------------
        # Copie de l'image pour l'affichage
        # ----------------------------------------------------------
        image_affichage = image.copy()


        # ----------------------------------------------------------
        # Dessin de la BBox rouge
        # ----------------------------------------------------------
        if bbox_detectee is not None:

            x_min, y_min, x_max, y_max = (
                bbox_detectee
            )

            cv2.rectangle(
                image_affichage,
                (x_min, y_min),
                (x_max, y_max),
                (0, 0, 255),
                2
            )

        else:

            print(f'Aucune bestiole détectée : "{chemin_image.name}"')


        # ----------------------------------------------------------
        # Dessin de la BBox sécurisée verte
        # ----------------------------------------------------------
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


        # ----------------------------------------------------------
        # Conversion BGR -> RGB
        # ----------------------------------------------------------

        image_rgb = cv2.cvtColor(
            image_affichage,
            cv2.COLOR_BGR2RGB
        )

        hauteur, largeur, canaux = (
            image_rgb.shape
        )

        octets_par_ligne = (
            canaux * largeur
        )


        # ----------------------------------------------------------
        # Conversion OpenCV -> QImage
        # ----------------------------------------------------------
        image_qt = QImage(
            image_rgb.data,
            largeur,
            hauteur,
            octets_par_ligne,
            QImage.Format_RGB888
        )

        # ----------------------------------------------------------
        # Conversion QImage -> QPixmap
        # ----------------------------------------------------------

        pixmap = QPixmap.fromImage(
            image_qt
        )

        # ----------------------------------------------------------
        # Redimensionnement pour la zone d'affichage
        # ----------------------------------------------------------

        pixmap = pixmap.scaled(
            self.affichage_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------

        self.affichage_image.setPixmap(
            pixmap
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
                ETAT_SAUVEGARDEE
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
                ETAT_SAUVEGARDEE
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
        Affiche ou masque les diagnostics intermédiaires.
        """

        self.affichage_masque.setVisible(
            actif
        )

        if actif:
            self.mettre_a_jour_masque()


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

        masque = self.masque_courant

        hauteur, largeur = (
            masque.shape
        )

        # Pour une image uint8 à un canal :
        octets_par_ligne = (
            masque.strides[0]
        )

        image_qt = QImage(
            masque.data,
            largeur,
            hauteur,
            octets_par_ligne,
            QImage.Format_Grayscale8
        ).copy()

        pixmap = QPixmap.fromImage(
            image_qt
        )

        pixmap = pixmap.scaled(
            self.affichage_masque.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.affichage_masque.setPixmap(
            pixmap
        )



    def mettre_a_jour_resume(self):
        """
        Met à jour le résumé du traitement à partir
        de l'état actuel des images.
        """

        total = len(
            self.etat_images
        )

        sauvegardees = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SAUVEGARDEE
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

        restantes = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_A_TRAITER
        )

        # ----------------------------------------------------------
        # Formatage des nombres
        # ----------------------------------------------------------

        total_form = util.format_nombre(total)
        sauvegardees_form = util.format_nombre(sauvegardees)
        sans_sauvegarde_form = util.format_nombre(sans_sauvegarde)
        sans_detection_form = util.format_nombre(sans_detection)
        erreurs_lecture_form = util.format_nombre(erreurs_lecture)
        restantes_form = util.format_nombre(restantes)

        # ----------------------------------------------------------
        # Affichage
        # ----------------------------------------------------------

        texte = (
            f"Images : {total_form}    |    "
            f"✓ Sauvegardées : {sauvegardees_form}    |    "
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

        sauvegardees = sum(
            1
            for etat in self.etat_images.values()
            if etat == ETAT_SAUVEGARDEE
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
        sauvegardees_form = util.format_nombre(sauvegardees)
        sans_sauvegarde_form = util.format_nombre(sans_sauvegarde)


        # ----------------------------------------------------------
        # Texte final
        # ----------------------------------------------------------
        texte = (
            "TRAITEMENT TERMINÉ\n\n"
            f"Images examinées       : {total_form} / {total_form}\n"
            f"Images sauvegardées    : {sauvegardees_form}\n"
            f"Sans sauvegarde        : {sans_sauvegarde_form}\n"
            f"Sans détection         : {sans_detection_form}\n"
            f"Erreurs de lecture     : {erreurs_lecture_form}"
        )

        self.resume_traitement.setText(texte)

        self.resume_traitement.setAlignment(Qt.AlignCenter)

        self.resume_traitement.setStyleSheet(
            """
            QLabel {
                padding: 12px;
                margin: 4px;
                font-weight: bold;
                font-size: 14px;

                color: #155724;
                background-color: #d4edda;

                border: 1px solid #c3e6cb;
                border-radius: 6px;
            }
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


#----------------------------------------------------------------------------------------
def main():

    application = QApplication(
        sys.argv
    )

    interface = InterfaceInfoCrop()

    interface.show()

    sys.exit(
        application.exec()
    )


if __name__ == "__main__":
    main()

    