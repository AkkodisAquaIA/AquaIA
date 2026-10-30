from __future__ import annotations

import queue
import threading
import tkinter as tk

from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


from ihm.moteur_augmentation import (
    ConfigurationAugmentation,
    EvenementProgression,
    ResumeTraitement,
    executer_augmentation,

    FLIP_H,
    FLIP_V,
    ROTATION,
    CONTRASTE,
    BRUIT,
    ZOOM,

    ROT_FLIT_H,
    ROT_FLIT_H_C,
    ROT_FLIT_V,
    ROT_FLIT_V_C,
    ZOOM_C,
    ZOOM_ROTATION,
    ZOOM_ROTATION_C,
    CONTRASTE_BRUIT,
)


class InterfaceAugmentation(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Aqua-IA - Génération d'images augmentées")

        # Taille de la fenêre
        # Taille minimale de la fenêtre
        self.minsize(1000, 750)

        # Dimensions de l'écran
        largeur_ecran = self.winfo_screenwidth()
        hauteur_ecran = self.winfo_screenheight()

        # Taille souhaitée
        largeur = min(1200, largeur_ecran - 100)
        hauteur = min(950, hauteur_ecran - 100)

        # Centrage de la fenêtre
        x = (largeur_ecran - largeur) // 2
        y = (hauteur_ecran - hauteur) // 2

        self.geometry(
            f"{largeur}x{hauteur}+{x}+{y}"
        )

        
        self.file_evenements = queue.Queue()
        self.demande_arret = threading.Event()
        self.thread_traitement = None

        self.variables_traitements = {}

        self.repertoire_var = tk.StringVar()
        self.nombre_rotation_var = tk.IntVar(value=1)
        self.nombre_zoom_var = tk.IntVar(value=1)
        self.nombre_autre_var = tk.IntVar(value=1)
        self.seuil_detection_var = tk.IntVar(value=35)

        self.repertoire_courant_var = tk.StringVar(
            value="Aucun traitement en cours"
        )

        self.image_courante_var = tk.StringVar(
            value="-"
        )

        self.images_creees_var = tk.StringVar(value="0")
        self.defauts_var = tk.StringVar(value="0")

        self.creer_styles()
        self.creer_interface()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.fermer_application
        )

        self.after(
            100,
            self.lire_evenements
        )

    # ========================================================
    # STYLE
    # ========================================================

    def creer_styles(self):

        style = ttk.Style(self)

        styles_disponibles = style.theme_names()

        if "vista" in styles_disponibles:
            style.theme_use("vista")

        style.configure(
            "Titre.TLabel",
            font=("Segoe UI", 18, "bold")
        )

        style.configure(
            "SousTitre.TLabel",
            font=("Segoe UI", 10),
            foreground="#4b5563"
        )

        style.configure(
            "Section.TLabelframe.Label",
            font=("Segoe UI", 10, "bold")
        )

        style.configure(
            "Statistique.TLabel",
            font=("Segoe UI", 12, "bold"),
            foreground="#006d77"
        )

    # ========================================================
    # CONSTRUCTION DE L'INTERFACE
    # ========================================================

    def creer_interface(self):

        conteneur = ttk.Frame(
            self,
            padding=15
        )

        conteneur.pack(
            fill="both",
            expand=True
        )

        # ----------------------------------------------------
        # En-tête
        # ----------------------------------------------------

        ttk.Label(
            conteneur,
            text="Génération automatique d'images augmentées",
            style="Titre.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            conteneur,
            text=(
                "Sélectionnez un répertoire, choisissez les "
                "augmentations, puis lancez le traitement."
            ),
            style="SousTitre.TLabel"
        ).pack(
            anchor="w",
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Répertoire
        # ----------------------------------------------------

        cadre_repertoire = ttk.LabelFrame(
            conteneur,
            text="Répertoire de travail",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_repertoire.pack(
            fill="x",
            pady=(0, 10)
        )

        entree_repertoire = ttk.Entry(
            cadre_repertoire,
            textvariable=self.repertoire_var
        )

        entree_repertoire.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 10)
        )

        ttk.Button(
            cadre_repertoire,
            text="Parcourir...",
            command=self.choisir_repertoire
        ).pack(side="right")

        # ----------------------------------------------------
        # Zone centrale
        # ----------------------------------------------------
        # ----------------------------------------------------
        # Zone centrale - Choix des augmentations
        # ----------------------------------------------------

        zone_centrale = ttk.Frame(conteneur)

        zone_centrale.pack(
            fill="x",
            expand=False,
            pady=(0, 10)
        )

        # ====================================================
        # 1 - SYMÉTRIE
        # ====================================================

        cadre_symetrie = ttk.LabelFrame(
            zone_centrale,
            text="Symétrie",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_symetrie.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 5)
        )

        # ====================================================
        # 2 - ROTATION
        # ====================================================

        cadre_rotation = ttk.LabelFrame(
            zone_centrale,
            text="Rotation",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_rotation.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=5
        )

        # ====================================================
        # 3 - ZOOM
        # ====================================================

        cadre_zoom = ttk.LabelFrame(
            zone_centrale,
            text="Zoom",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_zoom.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=5
        )

        # ====================================================
        # 4 - AUTRES
        # ====================================================

        cadre_autres = ttk.LabelFrame(
            zone_centrale,
            text="Autres",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_autres.grid(
            row=0,
            column=3,
            sticky="nsew",
            padx=(5, 0)
        )

        # ----------------------------------------------------
        # Les quatre colonnes occupent la même largeur
        # ----------------------------------------------------

        for colonne in range(4):
            zone_centrale.columnconfigure(
                colonne,
                weight=1,
                uniform="augmentations"
            )


        # ====================================================
        # TRAITEMENTS : SYMÉTRIE
        # ====================================================

        traitements_symetrie = [
            (FLIP_H, "Symétrie horizontale"),
            (FLIP_V, "Symétrie verticale"),
        ]


        # ====================================================
        # TRAITEMENTS : ROTATION
        # ====================================================

        traitements_rotation = [
            (
                ROTATION,
                "Rotation"
            ),
            (
                ROT_FLIT_H,
                "Rotation + Symétrie H"
            ),
            (
                ROT_FLIT_H_C,
                "Rotation + Symétrie H + contraste"
            ),
            (
                ROT_FLIT_V,
                "Rotation + Symétrie V"
            ),
            (
                ROT_FLIT_V_C,
                "Rotation + Symétrie V + contraste"
            ),
        ]


        # ====================================================
        # TRAITEMENTS : ZOOM
        # ====================================================

        traitements_zoom = [
            (
                ZOOM,
                "Zoom ou dézoom"
            ),
            (
                ZOOM_C,
                "Zoom + Contraste"
            ),
            (
                ZOOM_ROTATION,
                "Zoom + Rotation"
            ),
            (
                ZOOM_ROTATION_C,
                "Zoom + Rotation + Contraste"
            ),
        ]


        # ====================================================
        # TRAITEMENTS : AUTRES
        # ====================================================

        traitements_autres = [
            (
                CONTRASTE,
                "Luminosité-Contraste / Gamma"
            ),
            (
                BRUIT,
                "Bruit gaussien"
            ),
            (
                CONTRASTE_BRUIT,
                "Contraste + Bruit"
            ),
        ]


        # ====================================================
        # AJOUT DES CASES À COCHER
        # ====================================================

        self.ajouter_cases_traitements(
            cadre_symetrie,
            traitements_symetrie
        )

        self.ajouter_cases_traitements(
            cadre_rotation,
            traitements_rotation
        )

        self.ajouter_cases_traitements(
            cadre_zoom,
            traitements_zoom
        )

        self.ajouter_cases_traitements(
            cadre_autres,
            traitements_autres
        )

                


        # ----------------------------------------------------
        # Boutons de sélection
        # ----------------------------------------------------

        cadre_selection = ttk.Frame(conteneur)

        cadre_selection.pack(
            fill="x",
            pady=(0, 10)
        )

        ttk.Button(
            cadre_selection,
            text="Tout sélectionner",
            command=self.tout_selectionner
        ).pack(side="left")

        ttk.Button(
            cadre_selection,
            text="Tout désélectionner",
            command=self.tout_deselectionner
        ).pack(
            side="left",
            padx=(8, 0)
        )

        # ----------------------------------------------------
        # Boutons de commande
        # ----------------------------------------------------

        cadre_boutons = ttk.Frame(
            conteneur
        )

        cadre_boutons.pack(
            fill="x",
            pady=(0, 12)
        )

        # Bouton principal placé complètement à droite
        self.bouton_lancer = ttk.Button(
            cadre_boutons,
            text="▶  Lancer l'augmentation",
            command=self.lancer_traitement
        )

        self.bouton_lancer.pack(
            side="right",
            ipadx=12,
            ipady=4
        )

        # Bouton d'arrêt placé à gauche du bouton principal
        self.bouton_arreter = ttk.Button(
            cadre_boutons,
            text="■  Arrêter",
            command=self.arreter_traitement,
            state="disabled"
        )

        self.bouton_arreter.pack(
            side="right",
            padx=(0, 10),
            ipadx=8,
            ipady=4
        )


        # ----------------------------------------------------
        # Paramètres
        # ----------------------------------------------------

        cadre_parametres = ttk.LabelFrame(
            conteneur,
            text="Paramètres",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_parametres.pack(
            fill="x",
            pady=(0, 10)
        )

        self.ajouter_spinbox(
            cadre_parametres,
            colonne=0,
            texte="Rotations :",
            variable=self.nombre_rotation_var,
            minimum=1,
            maximum=10
        )

        self.ajouter_spinbox(
            cadre_parametres,
            colonne=1,
            texte="Zooms :",
            variable=self.nombre_zoom_var,
            minimum=1,
            maximum=10
        )

        self.ajouter_spinbox(
            cadre_parametres,
            colonne=2,
            texte="Autres :",
            variable=self.nombre_autre_var,
            minimum=1,
            maximum=10
        )

        self.ajouter_spinbox(
            cadre_parametres,
            colonne=3,
            texte="Seuil détection :",
            variable=self.seuil_detection_var,
            minimum=1,
            maximum=255
        )

        for colonne in range(4):
            cadre_parametres.columnconfigure(
                colonne,
                weight=1
            )

        # ----------------------------------------------------
        # Progression
        # ----------------------------------------------------

        cadre_progression = ttk.LabelFrame(
            conteneur,
            text="Progression",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_progression.pack(
            fill="x",
            pady=(0, 10)
        )

        ttk.Label(
            cadre_progression,
            textvariable=self.repertoire_courant_var
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.progression_repertoires = ttk.Progressbar(
            cadre_progression,
            mode="determinate"
        )

        self.progression_repertoires.pack(
            fill="x",
            pady=(0, 12)
        )

        ttk.Label(
            cadre_progression,
            textvariable=self.image_courante_var
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        self.progression_images = ttk.Progressbar(
            cadre_progression,
            mode="determinate"
        )

        self.progression_images.pack(
            fill="x",
            pady=(0, 10)
        )

        cadre_statistiques = ttk.Frame(cadre_progression)

        cadre_statistiques.pack(fill="x")

        ttk.Label(
            cadre_statistiques,
            text="Images créées :"
        ).pack(side="left")

        ttk.Label(
            cadre_statistiques,
            textvariable=self.images_creees_var,
            style="Statistique.TLabel"
        ).pack(
            side="left",
            padx=(5, 25)
        )

        ttk.Label(
            cadre_statistiques,
            text="Problèmes détectés :"
        ).pack(side="left")

        ttk.Label(
            cadre_statistiques,
            textvariable=self.defauts_var,
            style="Statistique.TLabel"
        ).pack(
            side="left",
            padx=(5, 0)
        )

        # ----------------------------------------------------
        # Journal
        # ----------------------------------------------------

        cadre_journal = ttk.LabelFrame(
            conteneur,
            text="Journal d'exécution",
            padding=10,
            style="Section.TLabelframe"
        )

        cadre_journal.pack(
            fill="both",
            expand=True,
            pady=(0, 10)
        )

        self.journal = ScrolledText(
            cadre_journal,
            height=10,
            state="disabled",
            font=("Consolas", 9),
            wrap="word"
        )

        self.journal.pack(
            fill="both",
            expand=True
        )

  

    # ========================================================
    # COMPOSANTS
    # ========================================================

    def ajouter_cases_traitements(
        self,
        parent,
        traitements
    ):
        for ligne, (identifiant, libelle) in enumerate(traitements):

            variable = tk.BooleanVar(value=False)

            self.variables_traitements[identifiant] = variable

            ttk.Checkbutton(
                parent,
                text=libelle,
                variable=variable
            ).grid(
                row=ligne,
                column=0,
                sticky="w",
                pady=3
            )

    def ajouter_spinbox(
        self,
        parent,
        colonne,
        texte,
        variable,
        minimum,
        maximum
    ):
        cadre = ttk.Frame(parent)

        cadre.grid(
            row=0,
            column=colonne,
            sticky="w",
            padx=5
        )

        ttk.Label(
            cadre,
            text=texte
        ).pack(side="left")

        ttk.Spinbox(
            cadre,
            from_=minimum,
            to=maximum,
            width=5,
            textvariable=variable
        ).pack(
            side="left",
            padx=(5, 0)
        )

    # ========================================================
    # ACTIONS
    # ========================================================

    def choisir_repertoire(self):

        repertoire = filedialog.askdirectory(
            title="Sélectionnez le répertoire de travail"
        )

        if repertoire:
            self.repertoire_var.set(repertoire)

    def tout_selectionner(self):

        for variable in self.variables_traitements.values():
            variable.set(True)

    def tout_deselectionner(self):

        for variable in self.variables_traitements.values():
            variable.set(False)

    def obtenir_traitements_selectionnes(self) -> set[str]:
        return {
            traitement
            for traitement, variable in self.variables_traitements.items()
            if variable.get()
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    def creer_configuration(
        self
    ) -> ConfigurationAugmentation | None:

        texte_repertoire = self.repertoire_var.get().strip()

        if not texte_repertoire:
            messagebox.showwarning(
                "Répertoire manquant",
                "Sélectionnez un répertoire de travail."
            )

            return None

        repertoire = Path(texte_repertoire)

        if not repertoire.exists():
            messagebox.showerror(
                "Répertoire introuvable",
                "Le répertoire sélectionné n'existe pas."
            )

            return None

        if not repertoire.is_dir():
            messagebox.showerror(
                "Sélection incorrecte",
                "Le chemin sélectionné n'est pas un répertoire."
            )

            return None

        traitements = self.obtenir_traitements_selectionnes()

        if not traitements:
            messagebox.showwarning(
                "Aucune augmentation",
                "Sélectionnez au moins une augmentation."
            )

            return None

        try:
            nombre_rotations = self.nombre_rotation_var.get()
            nombre_zooms = self.nombre_zoom_var.get()
            nombre_autres = self.nombre_autre_var.get()
            seuil_detection = self.seuil_detection_var.get()

        except tk.TclError:
            messagebox.showerror(
                "Paramètre incorrect",
                "Un ou plusieurs paramètres numériques sont incorrects."
            )

            return None

        return ConfigurationAugmentation(
            repertoire=repertoire,
            traitements=traitements,
            nombre_symetries=1,
            nombre_rotations=nombre_rotations,
            nombre_zooms=nombre_zooms,
            nombre_autres=nombre_autres,
            seuil_detection=seuil_detection
        )

    # ========================================================
    # LANCEMENT
    # ========================================================

    def lancer_traitement(self):

        if (
            self.thread_traitement is not None
            and self.thread_traitement.is_alive()
        ):
            return

        configuration = self.creer_configuration()

        if configuration is None:
            return

        self.demande_arret.clear()

        self.progression_repertoires["value"] = 0
        self.progression_images["value"] = 0

        self.images_creees_var.set("0")
        self.defauts_var.set("0")

        self.effacer_journal()
        self.ajouter_journal("Démarrage du traitement...")

        self.bouton_lancer.configure(state="disabled")
        self.bouton_arreter.configure(state="normal")

        self.thread_traitement = threading.Thread(
            target=self.executer_dans_thread,
            args=(configuration,),
            daemon=True
        )

        self.thread_traitement.start()

    def executer_dans_thread(
        self,
        configuration: ConfigurationAugmentation
    ):
        try:
            executer_augmentation(
                configuration=configuration,
                callback=self.file_evenements.put,
                demande_arret=self.demande_arret
            )

        except Exception as erreur:
            self.file_evenements.put(
                EvenementProgression(
                    type="erreur",
                    message=(
                        f"{type(erreur).__name__} : {erreur}"
                    )
                )
            )

    def arreter_traitement(self):

        if (
            self.thread_traitement is None
            or not self.thread_traitement.is_alive()
        ):
            return

        self.demande_arret.set()
        self.bouton_arreter.configure(state="disabled")

        self.ajouter_journal(
            "Demande d'arrêt en cours..."
        )

    # ========================================================
    # GESTION DES ÉVÉNEMENTS
    # ========================================================

    def lire_evenements(self):

        try:
            while True:
                evenement = self.file_evenements.get_nowait()
                self.traiter_evenement(evenement)

        except queue.Empty:
            pass

        self.after(
            100,
            self.lire_evenements
        )

    def traiter_evenement(
        self,
        evenement: EvenementProgression
    ):

        if evenement.type == "initialisation":

            self.progression_repertoires["maximum"] = max(
                evenement.maximum,
                1
            )

            self.ajouter_journal(evenement.message)

        elif evenement.type == "repertoire":

            self.progression_repertoires["maximum"] = max(
                evenement.maximum,
                1
            )

            self.progression_repertoires["value"] = evenement.valeur

            nombre_images = evenement.donnees.get(
                "nombre_images",
                0
            )

            self.progression_images["maximum"] = max(
                nombre_images,
                1
            )

            self.progression_images["value"] = 0

            self.repertoire_courant_var.set(
                f"Répertoire : {evenement.message}"
            )

            self.ajouter_journal(
                f"Traitement du répertoire : {evenement.message} "
                f"({nombre_images} image(s))"
            )

        elif evenement.type == "image":

            self.progression_images["maximum"] = max(
                evenement.maximum,
                1
            )

            self.progression_images["value"] = evenement.valeur

            self.image_courante_var.set(
                f"Image : {evenement.message}"
            )

        elif evenement.type == "statistiques":

            self.images_creees_var.set(
                str(
                    evenement.donnees.get(
                        "images_creees",
                        0
                    )
                )
            )

            self.defauts_var.set(
                str(
                    evenement.donnees.get(
                        "defauts",
                        0
                    )
                )
            )

        elif evenement.type == "journal":

            self.ajouter_journal(
                evenement.message
            )

        elif evenement.type == "annulation":

            self.ajouter_journal(
                evenement.message
            )

        elif evenement.type == "erreur":

            self.fin_interface_traitement()

            self.ajouter_journal(
                f"ERREUR : {evenement.message}"
            )

            messagebox.showerror(
                "Erreur",
                evenement.message
            )

        elif evenement.type == "termine":

            resume = evenement.donnees.get("resume")

            self.fin_interface_traitement()

            if resume is not None:
                self.afficher_resume(resume)

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    def afficher_resume(
            self,
            resume: ResumeTraitement
        ):

            # --------------------------------------------------------
            # Aucun sous-répertoire trouvé
            # --------------------------------------------------------
            if resume.nombre_repertoires == 0:

                self.ajouter_journal(
                    "Traitement terminé : aucun sous-répertoire à traiter."
                )

                messagebox.showwarning(
                    "Aucun sous-répertoire",
                    (
                        "Le répertoire sélectionné ne contient "
                        "aucun sous-répertoire à traiter."
                    )
                )

                return

            # --------------------------------------------------------
            # Résumé d'un traitement normal
            # --------------------------------------------------------
            lignes = [
                (
                    f"Répertoires traités : "
                    f"{resume.nombre_repertoires_traites}"
                    f"/{resume.nombre_repertoires}"
                ),
                (
                    f"Images sources traitées : "
                    f"{resume.nombre_images_sources}"
                ),
                (
                    f"Images augmentées créées : "
                    f"{resume.nombre_images_creees}"
                ),
                (
                    f"Répertoires vides : "
                    f"{len(resume.repertoires_vides)}"
                ),
                (
                    f"Problèmes détectés : "
                    f"{len(resume.defauts)}"
                ),
            ]

            if resume.rapport_defauts is not None:
                lignes.extend([
                    "",
                    "Rapport des défauts :",
                    str(resume.rapport_defauts)
                ])

            texte = "\n".join(lignes)

            self.ajouter_journal("")
            self.ajouter_journal("Résumé d'exécution")
            self.ajouter_journal(texte)

            if resume.annule:

                messagebox.showwarning(
                    "Traitement interrompu",
                    texte
                )

            elif resume.defauts:

                messagebox.showwarning(
                    "Traitement terminé avec des problèmes",
                    texte
                )

            else:

                messagebox.showinfo(
                    "Traitement terminé",
                    texte
                )



    def fin_interface_traitement(self):

        self.bouton_lancer.configure(state="normal")
        self.bouton_arreter.configure(state="disabled")

        self.repertoire_courant_var.set(
            "Traitement terminé"
        )

        self.image_courante_var.set("-")

    # ========================================================
    # JOURNAL
    # ========================================================

    def ajouter_journal(self, texte: str):

        self.journal.configure(state="normal")

        self.journal.insert(
            "end",
            texte + "\n"
        )

        self.journal.see("end")
        self.journal.configure(state="disabled")

    def effacer_journal(self):

        self.journal.configure(state="normal")
        self.journal.delete("1.0", "end")
        self.journal.configure(state="disabled")

    # ========================================================
    # FERMETURE
    # ========================================================

    def fermer_application(self):

        traitement_actif = (
            self.thread_traitement is not None
            and self.thread_traitement.is_alive()
        )

        if traitement_actif:

            reponse = messagebox.askyesno(
                "Traitement en cours",
                (
                    "Un traitement est encore en cours.\n\n"
                    "Voulez-vous demander son arrêt et fermer "
                    "l'application ?"
                )
            )

            if not reponse:
                return

            self.demande_arret.set()

        self.destroy()
