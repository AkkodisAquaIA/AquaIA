from typing import Tuple

from config import constants as ct
#------------------------------------------------------------------------------
def titre_centre(texte, largeur=ct.DISPLAY_WIDTH, remplissage='—'):  # '—' 
    return f" {texte} ".center(largeur, remplissage)

def titre_dataset_centre(
    titre: str,
    dataset: str,
    largeur: int = ct.DISPLAY_WIDTH
) -> str:

    largeur = ct.DISPLAY_WIDTH

    dataset = f"- {dataset} "
    titre = f" {titre} "

    # Position du début du titre pour qu'il soit centré
    debut_titre = (largeur - len(titre)) // 2

    # Nombre de tirets entre le dataset et le titre
    nb_tirets_gauche = max(1, debut_titre - len(dataset))

    # Nombre de tirets après le titre
    nb_tirets_droite = max(
        1,
        largeur - len(dataset) - nb_tirets_gauche - len(titre)
    )

    tag = (
        dataset +
        "—" * nb_tirets_gauche +
        titre +
        "—" * nb_tirets_droite
    )

    return tag

#------------------------------------------------------------------------------
# A color spec is (red, green, blue, prefix_str)
ColorSpec = Tuple[int, int, int, str]

class DisplayColor:
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def __init__(self) -> None:
        """Initialize the DisplayColor helper."""
        # Nothing to do here for now

    def print(self, text: str, color_spec: ColorSpec, name_dataset: str = "",  bold: bool = False) -> None:
        """
        Print `text` in the RGB color and with the prefix defined by `color_spec`.

        Args:
            text: The message to display.
            color_spec: 4-tuple (r, g, b, prefix), where:
                - r, g, b are ints 0-255
                - prefix is a short string (e.g. "[X] ")
            bold: If True, render message in bold.

        Usage:
            self.display.print("Something went wrong", colors["error"])
        """
        # Destructure for clarity
        r, g, b, prefix = color_spec

        # Build ANSI escape codes
        rgb_code  = f"\033[38;2;{r};{g};{b}m"
        bold_code = self.BOLD if bold else ""

        tag = text

        # Special handling for titles (prefix "T") to center them
        if prefix == "T":
            prefix = ""  # No prefix for titles, just centering
            tag = titre_centre(text)
            print()
        
        # Dataset + titre centré
        elif prefix == "DO" or prefix == "DF" :
            prefix = ""
            tag = titre_dataset_centre(text, name_dataset)
            print()

        # Final output
        print(f"{rgb_code}{bold_code}{prefix}{tag}{self.RESET}")


    def colored(self, text: str, color_spec: ColorSpec, bold: bool = False) -> str:
        """
        Renvoie le texte avec les codes ANSI pour couleur et bold (sans print).
        """
        r, g, b, prefix = color_spec
        rgb_code  = f"\033[38;2;{r};{g};{b}m"
        bold_code = self.BOLD if bold else ""
        return f"{rgb_code}{bold_code}{prefix}{text}{self.RESET}"        

#========================================================================================
#-----------------------------------------------------------------------------------
# Display Colors & Prefixes (RGB + label prefix)
#-----------------------------------------------------------------------------------

DISPLAY_COLORS = {
    # Standard statuses
    'error':   (204,  51,   0, "[X] "),    # Red           → critical error        : [X] Message
    'warning': (204, 204,   0, "[!] "),    # Yellow/Orange → warning               : [!] Message
    'input':   (153, 204,  51, "[?] "),    # Light green   → user input            : [?] Message
    'ok':      ( 51, 153,   0, "[√] "),    # Green         → success               : [√] Message
    'info':    ( 51, 102, 255, "[I] "),    # Blue          → informational message : [I] Message
    'wait':    (255, 153,  51, "[...] "),  # Orange        → processing/wait       : [...] Message
    'goodbye': (255,  16, 240, "[<3] "),   # Purple        → exit message          : [<3] Message

    # Custom prefixes for specific message types
    # Titre centrè
    'titre':   ( 0,  204, 153, "T"),   # ———— Titre ————

    # Dataset & Titre centrè
    'data_ok':   (  51, 153,   0, "DO"),    # — data ———— Titre ————
    'data_df':   ( 204, 204,   0, "DF"),    # — data ———— Titre ————

    # Aqua-IA themed colors (blue-green palette)
    'aqua_light': (102, 255, 204, "[~] "), # Light turquoise
    'aqua':       (  0, 204, 153, "[~] "), # Standard teal
    'aqua_dark':  (  0, 102, 102, "[~] "), # Dark blue-green
}


