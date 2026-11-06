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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from config import constants as cst
from tools import utility as util

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
class ErreurLectureImage(Exception):
    """Erreur lors de la lecture ou du décodage d'une image."""
    pass


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

        self.etat_images = {}

        self._creer_interface()

    def _creer_interface(self):

        widget_principal = QWidget()

        self.setCentralWidget(widget_principal)

        layout_principal = QVBoxLayout(widget_principal)

        
        # ----------------------------------------------------------
        # Titre
        # ----------------------------------------------------------
        titre = QLabel(
            "Info & Crop d'images"
        )

        titre.setAlignment(
            Qt.AlignHCenter | Qt.AlignTop
        )

        layout_principal.addWidget(
            titre
        )
        # ----------------------------------------------------------
        # Sélection du répertoire
        # ----------------------------------------------------------
        layout_repertoire = QHBoxLayout()

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

        layout_repertoire.addWidget(
            label_repertoire
        )

        layout_repertoire.addWidget(
            self.champ_repertoire
        )

        layout_repertoire.addWidget(
            self.bouton_parcourir
        )

        # Ajout du répertoire EN PREMIER
        layout_principal.addLayout(
            layout_repertoire
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
        layout_image = QHBoxLayout()

        label_image = QLabel(
            "Image :"
        )

        self.combo_images = QComboBox()

        # Désactivé tant qu'aucun répertoire valide
        # n'a été sélectionné
        self.combo_images.setEnabled(
            False
        )

        layout_image.addWidget(
            label_image
        )

        layout_image.addWidget(
            self.combo_images
        )

        # Ajout des images APRÈS le répertoire
        layout_principal.addLayout(
            layout_image
        )


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

        self.bouton_bbox_verte = QPushButton(
            "Sauvegarder BBox verte"
        )

        self.bouton_sans_sauvegarde = QPushButton(
            "Ne pas sauvegarder"
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

        self.bbox_detectee_courante = None
        self.bbox_securisee_courante = None

        self.bouton_bbox_rouge.setEnabled(False)
        self.bouton_bbox_verte.setEnabled(False)
        self.bouton_sans_sauvegarde.setEnabled(False)


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

            self.bouton_sans_sauvegarde.setEnabled(True)

            self.chemin_image_courante = Path(chemin_image)

            self.definir_etat_image(chemin_image, ETAT_ERREUR_LECTURE)

            print(f"Erreur : {erreur}")

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

        fichier_sans_detection = (
            sauvegarder_liste_erreurs(
                repertoire=repertoire,
                nom_fichier="images_sans_detection.txt",
                liste=images_sans_detection
            )
        )

        # ----------------------------------------------------------
        # Rapport : erreurs de lecture
        # ----------------------------------------------------------

        fichier_erreurs_lecture = (
            sauvegarder_liste_erreurs(
                repertoire=repertoire,
                nom_fichier="images_erreur_lecture.txt",
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


#--------------------------------------------------------------------------------------------------
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

    repertoire = Path(
        repertoire
    )

    fichier_sortie = (
        repertoire
        / nom_fichier
    )

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

    largeur_colonne = max(
        len(nom)
        for nom in liste
    )

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

            ligne = liste[
                i:i + nombre_colonnes
            ]

            ligne_formatee = " | ".join(
                f"{nom:<{largeur_colonne}}"
                for nom in ligne
            )

            fichier.write(
                ligne_formatee.rstrip()
                + "\n"
            )

    return fichier_sortie

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

    