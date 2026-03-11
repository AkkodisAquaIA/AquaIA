# tools/display_color.py

from typing import Tuple

# A color spec is (red, green, blue, prefix_str)
ColorSpec = Tuple[int, int, int, str]

class DisplayColor:
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def __init__(self) -> None:
        """Initialize the DisplayColor helper."""
        # Nothing to do here for now

    def print(self, text: str, color_spec: ColorSpec, bold: bool = False) -> None:
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

        # Final output
        print(f"{rgb_code}{bold_code}{prefix}{text}{self.RESET}")
