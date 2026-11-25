import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections.abc import Sequence
from pathlib import Path
import numpy as np

import tools.utility as util
from config import constants as ct
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors

#-----------------------------------------------------------------------------------
def save_plot(filename: str, cfg) -> None:
    """
    Save the current matplotlib figure if SAVE_PLOT is enabled.

    Args:
        filename (str): Name of the output file (e.g. 'plot.png').
    """

    display = dc.DisplayColor()

    if cfg["SAVE_PLOT"] :

        path_user: Path = Path(cfg["SAVE_USER"])
        if not path_user.exists():
            path_user = Path.cwd() / "Report"

        path_user.mkdir(parents=True, exist_ok=True)    
        new = util.horodatage(filename)
        file_path: Path = path_user / new

        try:
            # évite les labels coupés
            plt.tight_layout()

            # Save the current matplotlib figure
            plt.savefig(file_path, dpi=300)
        except Exception as e:
            display.print(f"Erreur sauvegarde : {file_path}", colors['error'])

def bbox_overflow(
    cfg,
    values: Sequence[float],
    overflow_warning: float,
    overflow_error: float,
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
    save_plot("bbox_overflow.png", cfg)
    plt.show()

def histogram_taille_bbox(
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    cfg,
    ymax :int
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
    plt.hist(values, bins=ct.BINS)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    
    plt.ylim(top=ymax)

    plt.tight_layout()
    save_plot("histogram.png",  cfg)
    plt.show()

def histogram_anomalies(
    values: dict[str, int],
    y_label: str,
    cfg,
    fault: str | None = None,
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
    save_plot("histogram_multiple",  cfg)
    plt.show()

def histogram_classe(items, class_names, cfg, total):

    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)

    #items_sorted = items_sorted[:40]

    n = len(items_sorted)

    def build_data(sub_items):
        labels = []
        counts = []
        colors = []

        for cls, count in sub_items:
            pct = (count / total) * 100
            name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"

            labels.append(name)
            counts.append(count)

            if pct < cfg["RARE"]:
                colors.append("red")
            elif pct < cfg["DOMINANT"]:
                colors.append("orange")
            else:
                colors.append("green")

        return labels, counts, colors

    # --- seuils globaux ---
    rare_val = (cfg["RARE"] / 100) * total
    dom_val = (cfg["DOMINANT"] / 100) * total

    # --- CAS 1 ---
    if n <= 40:
        labels, counts, colors = build_data(items_sorted)
        y_pos = np.arange(len(labels))

        plt.figure(figsize=(16, max(6, 0.4 * len(labels))))
        plt.barh(y_pos, counts, color=colors)
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()

        plt.axvline(rare_val, color='red', linestyle='--', label=f'Rare ({cfg["RARE"]}%)')
        plt.axvline(dom_val, color='green', linestyle='--', label=f'Dominant ({cfg["DOMINANT"]}%)')

        plt.axvspan(0, rare_val, color='red', alpha=0.08)
        plt.axvspan(rare_val, dom_val, color='orange', alpha=0.08)
        plt.axvspan(dom_val, max(counts), color='green', alpha=0.08)

    # --- CAS 2 : 2 graphes ---
    else:
        mid = n // 2
        left_items = items_sorted[:mid]
        right_items = items_sorted[mid:]

        #fig, axes = plt.subplots(1, 2, figsize=(22, max(6, 0.3 * mid)))
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        for ax, sub_items, title in zip(
            axes,
            [left_items, right_items],
            ["Classes dominantes", "Classes rares"]
        ):
            labels, counts, colors = build_data(sub_items)
            y_pos = np.arange(len(labels))

            ax.barh(y_pos, counts, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(title)

            # seuils
            ax.axvline(rare_val, color='red', linestyle='--', label=f'Rare ({cfg["RARE"]}%)')
            ax.axvline(dom_val, color='green', linestyle='--', label=f'Dominant ({cfg["DOMINANT"]}%)')

            ax.axvspan(0, rare_val, color='red', alpha=0.08)
            ax.axvspan(rare_val, dom_val, color='orange', alpha=0.08)
            ax.axvspan(dom_val, max(counts), color='green', alpha=0.08)

        axes[0].set_xlabel("Nombre d'éléments")
        axes[1].set_xlabel("Nombre d'éléments")

    plt.legend(loc='lower right')

    plt.tight_layout()
    save_plot("histogram_X_classe", cfg)
    plt.show()    
