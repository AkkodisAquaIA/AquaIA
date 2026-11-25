


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
