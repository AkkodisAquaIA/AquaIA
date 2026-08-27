from typing import Tuple

from config import constants as ct
#------------------------------------------------------------------------------
def titre_centre(texte, largeur=ct.DISPLAY_WIDTH, remplissage=ct.TITLE_FILL):  # '—' 
    """
    Return a title centered within a line filled with the specified character.

    Args:
        texte: Title to center.
        largeur: Total line width.
        remplissage: Character used to fill the remaining space.

    Returns:
        A centered title string.
    """

    remplissage = (remplissage or ct.TITLE_FILL)[0]
    
    return f" {texte} ".center(largeur, remplissage)

def titre_dataset_centre(
    titre: str,
    dataset: str,
    remplissage=ct.TITLE_FILL
    ) -> str:
    """
    Return a centered section title prefixed with the dataset name.

    The dataset name is left-aligned while the title remains centered
    within the available display width.

    Args:
        titre: Section title.
        dataset: Dataset name displayed at the beginning of the line.
        largeur: Total line width.

    Returns:
        A formatted header string.
    """

    largeur = ct.DISPLAY_WIDTH

    remplissage = (remplissage or ct.TITLE_FILL)[0]
    dataset = f"├{remplissage} {dataset} "
    titre = f" {titre} "

    # Starting position of the centered title
    debut_titre = (largeur - len(titre)) // 2

    # Number of separator characters between the dataset name and the title
    nb_tirets_gauche = max(1, debut_titre - len(dataset))

    # Number of separator characters after the title
    nb_tirets_droite = max(1, largeur - len(dataset) - nb_tirets_gauche - len(titre))

    tag = (
        dataset +
        remplissage * nb_tirets_gauche +
        titre +
        remplissage * nb_tirets_droite
    )

    return tag

#------------------------------------------------------------------------------
# RGB color definition:
# (red, green, blue, message prefix)
ColorSpec = Tuple[int, int, int, str]


class DisplayColor:
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def __init__(self) -> None:
        """Initialize the DisplayColor helper."""
        # Nothing to do here for now

   
    def _ansi(
    self,
    color_spec: ColorSpec,
    bold: bool = False
) -> tuple[str, str, str]:
        """
        Build ANSI escape sequences from a color specification.
        """
        r, g, b, prefix = color_spec

        rgb_code = f"\033[38;2;{r};{g};{b}m"
        bold_code = self.BOLD if bold else ""

        return rgb_code, bold_code, prefix


    def colored(self, text: str, color_spec: ColorSpec, bold: bool = False, pref: bool = True) -> str:
        """
        Return a string formatted with ANSI escape sequences.

        Unlike `print()`, this method only returns the formatted string
        without displaying it.

        Args:
            text: Message to format.
            color_spec: Tuple (red, green, blue, prefix).
            bold: If True, apply bold formatting.
            pref: If False, omit the message prefix.

        Returns:
            The ANSI-formatted string.
        """


        rgb_code, bold_code, prefix = self._ansi(color_spec, bold)
        if not pref :
            prefix =""

        return f"{rgb_code}{bold_code}{prefix}{text}{self.RESET}"        


    def print(self, text: str, color_spec: ColorSpec, bold: bool = False) -> None:
        """
        Print a colored message using ANSI escape sequences.

        Args:
            text: Message to display.
            color_spec: Tuple (red, green, blue, prefix), where:
                - red, green and blue are integers in the range [0, 255]
                - prefix is prepended to the message (e.g. "[X] ")
            name_dataset: Reserved for future use.
            bold: If True, display the message in bold.

        Example:
            display.print("File successfully loaded.", DISPLAY_COLORS["ok"])
        """

        rgb_code, bold_code, prefix = self._ansi(color_spec, bold)

        # Final output
        print(f"{rgb_code}{bold_code}{prefix}{text}{self.RESET}")

    def titre(self, text: str, color_spec: ColorSpec, bold: bool = False) -> None:
        """
        Print a centered title using the specified color.

        Args:
            text: Title to display.
            color_spec: Tuple (red, green, blue, prefix).
            name_dataset: Reserved for future use.
            bold: If True, display the title in bold.

        Example:
            display.titre("Dataset Validation", DISPLAY_COLORS["info"])
        """

        rgb_code, bold_code, prefix = self._ansi(color_spec, bold)

        tag = titre_centre(text)

        # Display the formatted title
        print()
        print(f"{rgb_code}{bold_code}{tag}{self.RESET}")

    def header_title(self, text: str, color_spec: ColorSpec, header: str = "",  bold: bool = False) -> None:
        """
        Print a centered section header prefixed with a dataset name.

        Args:
            text: Section title.
            color_spec: Tuple (red, green, blue, prefix).
            header: Dataset name displayed at the beginning of the line.
            bold: If True, display the header in bold.

        Example:
            display.header_title(
                "Statistics",
                DISPLAY_COLORS["info"],
                "COCO128"
            )
        """

        rgb_code, bold_code, prefix = self._ansi(color_spec, bold)

        # Build the formatted dataset header
        tag = titre_dataset_centre(text, header)

        # Display the formatted header
        print()
        print(f"{rgb_code}{bold_code}{tag}{self.RESET}")

 
#========================================================================================
# ==============================================================================
# Display Colors
# RGB values and associated message prefixes
# ==============================================================================-

DISPLAY_COLORS = {
    # Standard statuses
    'error':   (204,  51,   0, "[X] "),    # Red           → critical error        : [X] Message
    'warning': (204, 204,   0, "[!] "),    # Yellow/Orange → warning               : [!] Message
    'input':   (153, 204,  51, "[?] "),    # Light green   → user input            : [?] Message
    'ok':      ( 51, 153,   0, "[√] "),    # Green         → success               : [√] Message
    'info':    ( 51, 102, 255, "[I] "),    # Blue          → informational message : [I] Message
    'wait':    (255, 153,  51, "[...] "),  # Orange        → processing/wait       : [...] Message
    'goodbye': (255,  16, 240, "[<3] "),   # Purple        → exit message          : [<3] Message
    'number':  (255,  16, 240, ""),        # Purple        → numeric values 

    # Aqua-IA color palette
    'aqua_light': (102, 255, 204, "[~] "), # Light turquoise
    'aqua':       (  0, 204, 153, "[~] "), # Standard teal
    'aqua_dark':  (  0, 102, 102, "[~] "), # Dark blue-green

}

