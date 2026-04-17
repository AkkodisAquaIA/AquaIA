import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections.abc import Sequence
from pathlib import Path
from datetime import datetime

import tools.constants as ct


def save_plot(filename: str, path_user: Path | None) -> None:
    """
    Save the current matplotlib figure if SAVE_PLOT is enabled.

    Args:
        filename (str): Name of the output file (e.g. 'plot.png').
        path_user (Path | None): Destination directory.
    """

    if ct.SAVE_PLOT and path_user is not None:
        # Generate a timestamp (format: YYYYMMDD_HHMMSS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Extract filename without extension and file extension
        stem = Path(filename).stem
        suffix = Path(filename).suffix

        # Build a new filename including the timestamp
        filename_with_timestamp = f"{stem}_{timestamp}{suffix}"

        # Build full file path
        file_path: Path = path_user / filename_with_timestamp

        # Save the current matplotlib figure
        plt.savefig(file_path, dpi=300)


def bbox_overflow(
    values: Sequence[float],
    overflow_warning: float,
    overflow_error: float,
    path_user: Path | None = None
) -> None:
    """
    Plot the distribution of bounding box overflow percentages.

    A histogram is displayed along with vertical threshold lines 
    representing warning and error levels.

    Args:
        values (Sequence[float]):
            List or array of overflow percentage values.
        overflow_warning (float):
            Warning threshold (percentage).
        overflow_error (float):
            Error threshold (percentage).
    """

    plt.figure(figsize=(12, 6))

    counts, bins, patches = plt.hist(
        values,
        bins=50,
        color="skyblue",
        edgecolor="black"
    )

    # Vertical threshold lines with legend labels
    plt.axvline(
        overflow_warning,
        color="orange",
        linestyle="--",
        linewidth=3,
        label=f"Warning ({overflow_warning:.2f}%)"
    )

    plt.axvline(
        overflow_error,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Error ({overflow_error:.2f}%)"
    )

    # Display threshold values at top of the plot
    ymax: float = plt.ylim()[1]

    plt.text(
        overflow_warning,
        ymax * 0.95,
        f"{overflow_warning:.2f}%",
        color="orange",
        fontsize=12,
        rotation=90,
        va="top",
        ha="right"
    )

    plt.text(
        overflow_error,
        ymax * 0.95,
        f"{overflow_error:.2f}%",
        color="red",
        fontsize=12,
        rotation=90,
        va="top",
        ha="right"
    )

    plt.xlabel("Outside ratio (%)")
    plt.ylabel("Number of bounding boxes")
    plt.title("Bounding Box Overflow Distribution")

    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_plot("bbox_overflow.png", path_user)
    plt.show()

def histogram(
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    path_user: Path | None = None
) -> None:
    """
    Display a simple histogram.

    Args:
        values (Sequence[float]):
            Numerical values to plot.
        title (str):
            Plot title.
        x_label (str):
            Label for the X-axis.
        y_label (str):
            Label for the Y-axis.
    """

    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=50)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_plot("histogram.png", path_user)
    plt.show()

def histogram_multiple(
    values: dict[str, int],
    y_label: str,
    fault: str | None = None,
    path_user: Path | None = None
) -> None:
    """
    Display a bar chart for anomaly counts by type.

    Error types containing the word "error" are displayed in red,
    others in orange.

    Args:
        values (dict[str, int]):
            Mapping of anomaly type to occurrence count.
        y_label (str):
            Label for the Y-axis.
        fault (str | None):
            Optional parameter (currently unused, reserved for future use).
    """

    plt.figure(figsize=(12, 6))

    colors: list[str] = [
        "red" if "error" in anomaly_type.lower() else "orange"
        for anomaly_type in values.keys()
    ]

    labels: list[str] = list(values.keys())
    counts: list[int] = list(values.values())

    plt.bar(labels, counts, color=colors)

    plt.xticks(rotation=45, ha="right")
    plt.title("Number of anomalies per type")
    plt.ylabel(y_label)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_plot("histogram_multiple", path_user)
    plt.show()
