import socket
from pathlib import Path
import argparse
import fiftyone as fo

from typing import Tuple, List

from tools.constants import DISPLAY_COLORS as colors
from tools import constants as ct

#==========================================================================================

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


# ------------------------------
# Function to find a free port
# ------------------------------
def get_free_port() -> int:
    """
    Returns an available port on the local machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Let OS assign a free port
        return s.getsockname()[1]


# ------------------------------
# Function to launch FiftyOne web interface
# ------------------------------
def launch_fiftyone_interface(dataset: fo.Dataset) -> None:
    """
    Launches the FiftyOne web app for a given dataset.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset to visualize.
    """

    color = DisplayColor()

    color.print("Launching FiftyOne web interface...", colors['info'])
    
    port = get_free_port()
    session = None
    try:
        session = fo.launch_app(dataset, port=port, remote=False)
        color.print(f"tyOne web interface available at: http://127.0.0.1:{port}", colors['info'])
        color.print("Waiting for the web interface to close", colors['wait'], bold=True)
        color.print("Press CTRL+C to continue if needed.", colors['wait'], bold=True)

        # Wait until the session is closed
        try:
            session.wait()
        except KeyboardInterrupt:
            color.print("CTRL+C detected, continuing program...", colors['warning'])

    except Exception as e:
        color   .print("Failed to launch FiftyOne web interface.", colors['error'])
        print("Error:", e)

    finally:
        if session is not None:
            session.close()
            color.print("FiftyOne session closed, continuing program.", colors['info'])


# ------------------------------
# Function to parse command-line arguments
# ------------------------------
def read_param() -> Tuple[Path, Path]:
    """
    Parses command-line arguments for base path and data folder.

    Returns:
        Tuple[Path, Path]: base path and data folder path

    Raises:
        FileNotFoundError: If paths do not exist
        NotADirectoryError: If the data path is not a directory
    """
    parser = argparse.ArgumentParser(description="Program using a base path and a data folder")

    parser.add_argument(
        "-p", "--work_dir",
        required=True,
        help="Base path (e.g., C:/my_data)"
    )

    parser.add_argument(
        "-f", "--folder",
        required=True,
        help="Folder containing the data (inside the base path)"
    )

    args = parser.parse_args()

    base_path = Path(args.work_dir)
    data_path = base_path / args.folder

    # Validate paths
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path}")

    return base_path, data_path


# ------------------------------
# Function to display and save problematic items
# ------------------------------
def display_and_save_errors(
    items: List[str],
    file_name: str,
    title: str,
    sort: bool = True,
    full_path: bool = False,
) -> None:
    """
    Display and save a list of problematic items in a file.

    Args:
        items (List[str]): List of file paths or names
        file_name (str): Output file name
        title (str): Title to display
        sort (bool): Whether to sort items before displaying
        full_path (bool): Display/write full paths if True, else only file names
        n_per_line (int): Number of items per line when printing
    """

    color = DisplayColor()

    if not items:
        color.print(f"{title}: No issues detected.\n", colors['ok'])
        return

    if sort:
        items = sorted(items, key=lambda p: str(p))

    color.print(f"{title}: {len(items)} item(s) detected", colors['warning'], bold=True)
    for i in range(0, len(items), ct.n_per_line):
        line_items = items[i:i + ct.n_per_line]
        line = " | ".join(str(x) if full_path else Path(x).name for x in line_items)
        print(line)
    
    # Save to file
    if ct.REPORT_MODE:
        with open(file_name, "w") as f:
            for x in items:
                f.write(str(x) if full_path else Path(x).name)
                f.write("\n")

        print(f"List saved to '{file_name}'\n")
    else:
        print()
