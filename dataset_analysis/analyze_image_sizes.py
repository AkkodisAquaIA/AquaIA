#!/usr/bin/env python3
"""
Analyse des dimensions d'un dataset d'images de classification.

Structure attendue :
DATASET_DIR/
├── Classe_1/
│   ├── image1.jpg
│   └── image2.jpg
├── Classe_2/
│   └── ...
└── ...

Le script analyse :
- largeur / hauteur
- côté court / côté long
- ratio largeur / hauteur
- nombre de pixels
- statistiques globales
- statistiques par classe
- impact de différentes résolutions d'entrée DINOv3
- histogrammes
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AG_dataset_genus_09-29")

OUTPUT_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/AquaIA/dataset_analysis/results_image_sizes")

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# Résolutions DINOv3 à comparer
TARGET_SIZES = [224, 336, 448, 512, 672]


# ============================================================
# DÉTECTION DU DATASET SOURCE
# ============================================================


def detect_source_dataset(path: Path) -> str:
    """
    Détecte le dataset source à partir du nom du fichier.

    IMPORTANT :
    FIN-Benthic2 est testé avant FIN-Benthic
    car "FIN-Benthic" est contenu dans "FIN-Benthic2".
    """

    name = path.name.lower()

    if "fin-benthic2" in name:
        return "FIN-Benthic2"

    if "fin-benthic" in name:
        return "FIN-Benthic"

    if "perla" in name:
        return "PERLA"

    return "INCONNU"


# ============================================================
# DÉTECTION DES IMAGES HORS NORME
# ============================================================


def detect_size_outliers(df: pd.DataFrame, iqr_factor: float = 1.5):
    """
    Détecte les images dont les dimensions sont atypiques
    par rapport aux autres images du même dataset source.

    Méthode IQR :
        borne basse = Q1 - iqr_factor * IQR
        borne haute = Q3 + iqr_factor * IQR

    avec :
        IQR = Q3 - Q1

    On analyse séparément :
    - le côté court
    - le côté long

    Une image est considérée comme hors norme si au moins
    une de ces deux dimensions sort des bornes.
    """

    all_outliers = []
    summary = []

    print("\n" + "=" * 70)
    print("DÉTECTION DES IMAGES HORS NORME")
    print("=" * 70)

    for dataset in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset].copy()

        # ----------------------------------------------------
        # Côté long
        # ----------------------------------------------------

        q1_long = subset["long_side"].quantile(0.25)
        q3_long = subset["long_side"].quantile(0.75)

        iqr_long = q3_long - q1_long

        lower_long = max(
            0,
            q1_long - iqr_factor * iqr_long,
        )

        upper_long = q3_long + iqr_factor * iqr_long

        # ----------------------------------------------------
        # Côté court
        # ----------------------------------------------------

        q1_short = subset["short_side"].quantile(0.25)
        q3_short = subset["short_side"].quantile(0.75)

        iqr_short = q3_short - q1_short

        lower_short = max(
            0,
            q1_short - iqr_factor * iqr_short,
        )

        upper_short = q3_short + iqr_factor * iqr_short

        # ----------------------------------------------------
        # Identification des outliers
        # ----------------------------------------------------

        subset["outlier_long"] = (subset["long_side"] < lower_long) | (subset["long_side"] > upper_long)

        subset["outlier_short"] = (subset["short_side"] < lower_short) | (subset["short_side"] > upper_short)

        subset["size_outlier"] = subset["outlier_long"] | subset["outlier_short"]

        outliers = subset[subset["size_outlier"]].copy()

        # ----------------------------------------------------
        # Type d'anomalie
        # ----------------------------------------------------

        def get_reason(row):
            reasons = []

            if row["long_side"] > upper_long:
                reasons.append("long_side_trop_grand")

            if row["long_side"] < lower_long:
                reasons.append("long_side_trop_petit")

            if row["short_side"] > upper_short:
                reasons.append("short_side_trop_grand")

            if row["short_side"] < lower_short:
                reasons.append("short_side_trop_petit")

            return ";".join(reasons)

        outliers["outlier_reason"] = outliers.apply(
            get_reason,
            axis=1,
        )

        # ----------------------------------------------------
        # Écart par rapport à la médiane
        # ----------------------------------------------------

        median_long = subset["long_side"].median()
        median_short = subset["short_side"].median()

        outliers["long_vs_median"] = outliers["long_side"] / median_long

        outliers["short_vs_median"] = outliers["short_side"] / median_short

        all_outliers.append(outliers)

        # ----------------------------------------------------
        # Résumé
        # ----------------------------------------------------

        summary.append(
            {
                "dataset": dataset,
                "n_images": len(subset),
                "n_outliers": len(outliers),
                "outliers_%": 100 * len(outliers) / len(subset),
                "long_Q1": q1_long,
                "long_median": median_long,
                "long_Q3": q3_long,
                "long_lower_limit": lower_long,
                "long_upper_limit": upper_long,
                "short_Q1": q1_short,
                "short_median": median_short,
                "short_Q3": q3_short,
                "short_lower_limit": lower_short,
                "short_upper_limit": upper_short,
            }
        )

        print(f"\n{dataset}")
        print("-" * 50)

        print(f"Côté long  : Q1={q1_long:.0f} | médiane={median_long:.0f} | Q3={q3_long:.0f}")

        print(f"Bornes     : {lower_long:.0f} → {upper_long:.0f} px")

        print(f"Côté court : Q1={q1_short:.0f} | médiane={median_short:.0f} | Q3={q3_short:.0f}")

        print(f"Bornes     : {lower_short:.0f} → {upper_short:.0f} px")

        print(f"Images hors norme : {len(outliers)} / {len(subset)} ({100 * len(outliers) / len(subset):.2f} %)")

        # Affichage des 10 plus grandes anomalies
        if len(outliers) > 0:
            print("\nImages les plus grandes :")

            largest = outliers.sort_values(
                "long_side",
                ascending=False,
            ).head(10)

            for _, row in largest.iterrows():
                print(f"  {row['width']}×{row['height']} | {row['class']} | {row['filename']}")

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    if all_outliers:
        outlier_df = pd.concat(
            all_outliers,
            ignore_index=True,
        )

    else:
        outlier_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary)

    return outlier_df, summary_df


# ============================================================
# ANALYSE DES IMAGES
# ============================================================


def collect_image_sizes(dataset_dir: Path) -> pd.DataFrame:
    records = []

    image_paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]

    print(f"Images trouvées : {len(image_paths)}")

    for i, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as img:
                width, height = img.size

            short_side = min(width, height)
            long_side = max(width, height)

            source_dataset = detect_source_dataset(path)

            records.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "class": path.parent.name,
                    "dataset": source_dataset,
                    "width": width,
                    "height": height,
                    "short_side": short_side,
                    "long_side": long_side,
                    "aspect_ratio": width / height,
                    "megapixels": (width * height) / 1_000_000,
                }
            )

        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARNING] Impossible de lire : {path} ({e})")

        if i % 5000 == 0:
            print(f"{i}/{len(image_paths)} images analysées...")

    return pd.DataFrame(records)


# ============================================================
# STATISTIQUES GLOBALES
# ============================================================


def print_stats(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STATISTIQUES GLOBALES")
    print("=" * 70)

    print(f"\nNombre d'images : {len(df)}")
    print(f"Nombre de classes : {df['class'].nunique()}")

    print("\nProvenance des images :")

    dataset_counts = df["dataset"].value_counts()

    for dataset, count in dataset_counts.items():
        percentage = 100 * count / len(df)

        print(f"{dataset:15s} : {count:6d} ({percentage:5.1f} %)")

    columns = [
        "width",
        "height",
        "short_side",
        "long_side",
        "aspect_ratio",
        "megapixels",
    ]

    for col in columns:
        values = df[col]

        print(f"\n--- {col} ---")

        print(f"Min      : {values.min():.2f}")
        print(f"P10      : {values.quantile(0.10):.2f}")
        print(f"P25      : {values.quantile(0.25):.2f}")
        print(f"Médiane  : {values.median():.2f}")
        print(f"Moyenne  : {values.mean():.2f}")
        print(f"P75      : {values.quantile(0.75):.2f}")
        print(f"P90      : {values.quantile(0.90):.2f}")
        print(f"P95      : {values.quantile(0.95):.2f}")
        print(f"Max      : {values.max():.2f}")


# ============================================================
# STATISTIQUES PAR DATASET
# ============================================================


def dataset_statistics(df: pd.DataFrame):
    stats = (
        df.groupby("dataset")
        .agg(
            n_images=("path", "count"),
            width_median=("width", "median"),
            height_median=("height", "median"),
            short_side_min=("short_side", "min"),
            short_side_p10=(
                "short_side",
                lambda x: x.quantile(0.10),
            ),
            short_side_p25=(
                "short_side",
                lambda x: x.quantile(0.25),
            ),
            short_side_median=("short_side", "median"),
            short_side_p75=(
                "short_side",
                lambda x: x.quantile(0.75),
            ),
            short_side_p90=(
                "short_side",
                lambda x: x.quantile(0.90),
            ),
            short_side_max=("short_side", "max"),
            long_side_min=("long_side", "min"),
            long_side_p10=(
                "long_side",
                lambda x: x.quantile(0.10),
            ),
            long_side_p25=(
                "long_side",
                lambda x: x.quantile(0.25),
            ),
            long_side_median=("long_side", "median"),
            long_side_p75=(
                "long_side",
                lambda x: x.quantile(0.75),
            ),
            long_side_p90=(
                "long_side",
                lambda x: x.quantile(0.90),
            ),
            long_side_max=("long_side", "max"),
            aspect_ratio_median=("aspect_ratio", "median"),
            megapixels_median=("megapixels", "median"),
        )
        .sort_values("long_side_median")
    )

    return stats


# ============================================================
# AFFICHAGE STATS PAR DATASET
# ============================================================


def print_dataset_stats(dataset_stats: pd.DataFrame):
    print("\n" + "=" * 70)
    print("DIMENSIONS PAR DATASET SOURCE")
    print("=" * 70)

    for dataset, row in dataset_stats.iterrows():
        print(f"\n{dataset}")
        print("-" * 50)

        print(f"Nombre d'images : {int(row['n_images'])}")

        print(
            f"Côté court : "
            f"P10={row['short_side_p10']:.0f} | "
            f"P25={row['short_side_p25']:.0f} | "
            f"médiane={row['short_side_median']:.0f} | "
            f"P75={row['short_side_p75']:.0f} | "
            f"P90={row['short_side_p90']:.0f}"
        )

        print(f"Côté long  : P10={row['long_side_p10']:.0f} | P25={row['long_side_p25']:.0f} | médiane={row['long_side_median']:.0f} | P75={row['long_side_p75']:.0f} | P90={row['long_side_p90']:.0f}")


# ============================================================
# COMPARAISON RÉSOLUTIONS DINOv3
# ============================================================


def analyse_target_sizes(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("COMPARAISON DES RÉSOLUTIONS DINOv3")
    print("=" * 70)

    results = []

    for dataset in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset]

        print("\n" + "=" * 50)
        print(f"{dataset} ({len(subset)} images)")
        print("=" * 50)

        for size in TARGET_SIZES:
            n_downscale = (subset["long_side"] > size).sum()

            n_upscale = (subset["long_side"] < size).sum()

            n_same = (subset["long_side"] == size).sum()

            pct_downscale = 100 * n_downscale / len(subset)

            pct_upscale = 100 * n_upscale / len(subset)

            pct_same = 100 * n_same / len(subset)

            scale_factor = size / subset["long_side"]

            results.append(
                {
                    "dataset": dataset,
                    "target_size": size,
                    "n_images": len(subset),
                    "downscaled_%": pct_downscale,
                    "upscaled_%": pct_upscale,
                    "unchanged_%": pct_same,
                    "median_scale_factor": scale_factor.median(),
                    "p10_scale_factor": scale_factor.quantile(0.10),
                    "p90_scale_factor": scale_factor.quantile(0.90),
                }
            )

            print(f"\n{size} × {size}")

            print(f"  Réduites   : {n_downscale:6d} ({pct_downscale:5.1f} %)")

            print(f"  Agrandies  : {n_upscale:6d} ({pct_upscale:5.1f} %)")

            print(f"  Identiques : {n_same:6d} ({pct_same:5.1f} %)")

            print(f"  Facteur médian : {scale_factor.median():.2f}×")

    return pd.DataFrame(results)


# ============================================================
# STATISTIQUES PAR CLASSE
# ============================================================


def class_statistics(df: pd.DataFrame):
    stats = (
        df.groupby("class")
        .agg(
            n_images=("path", "count"),
            n_datasets=("dataset", "nunique"),
            short_side_median=(
                "short_side",
                "median",
            ),
            long_side_median=(
                "long_side",
                "median",
            ),
            long_side_p25=(
                "long_side",
                lambda x: x.quantile(0.25),
            ),
            long_side_p75=(
                "long_side",
                lambda x: x.quantile(0.75),
            ),
            aspect_ratio_median=(
                "aspect_ratio",
                "median",
            ),
        )
        .sort_values("long_side_median")
    )

    return stats


# ============================================================
# STATISTIQUES CLASSE × DATASET
# ============================================================


def class_dataset_statistics(df: pd.DataFrame):
    stats = (
        df.groupby(["class", "dataset"])
        .agg(
            n_images=("path", "count"),
            short_side_median=(
                "short_side",
                "median",
            ),
            long_side_median=(
                "long_side",
                "median",
            ),
            long_side_p25=(
                "long_side",
                lambda x: x.quantile(0.25),
            ),
            long_side_p75=(
                "long_side",
                lambda x: x.quantile(0.75),
            ),
        )
        .reset_index()
    )

    return stats


# ============================================================
# MATRICE CLASSE × DATASET
# ============================================================


def create_class_dataset_matrix(df: pd.DataFrame):
    matrix = pd.crosstab(
        df["class"],
        df["dataset"],
    )

    matrix["TOTAL"] = matrix.sum(axis=1)

    # Nombre de datasets représentés pour chaque classe
    dataset_columns = [col for col in matrix.columns if col != "TOTAL"]

    matrix["N_DATASETS"] = (matrix[dataset_columns] > 0).sum(axis=1)

    return matrix


# ============================================================
# CLASSES MONO-DATASET
# ============================================================


def print_single_dataset_classes(matrix: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CLASSES PRÉSENTES DANS UN SEUL DATASET")
    print("=" * 70)

    single = matrix[matrix["N_DATASETS"] == 1]

    print(f"\n{len(single)} / {len(matrix)} classes sont présentes dans un seul dataset.")

    dataset_columns = [col for col in matrix.columns if col not in {"TOTAL", "N_DATASETS"}]

    for dataset in dataset_columns:
        classes = single[single[dataset] > 0]

        print(f"\n{dataset} : {len(classes)} classes")


# ============================================================
# GRAPHIQUES
# ============================================================


def create_plots(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Histogrammes séparés par dataset
    # --------------------------------------------------------

    for dataset in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset]

        # Côté long
        plt.figure(figsize=(10, 6))

        plt.hist(
            subset["long_side"],
            bins=100,
            alpha=0.8,
        )

        for size in TARGET_SIZES:
            plt.axvline(
                size,
                linestyle="--",
                label=f"{size}px",
            )

        plt.xlabel("Côté long (pixels)")
        plt.ylabel("Nombre d'images")

        plt.title(f"Distribution du côté long — {dataset}")

        plt.legend()
        plt.tight_layout()

        safe_name = dataset.replace("/", "_").replace(" ", "_")

        plt.savefig(
            output_dir / f"hist_long_side_{safe_name}.png",
            dpi=200,
        )

        plt.close()

        # Côté court
        plt.figure(figsize=(10, 6))

        plt.hist(
            subset["short_side"],
            bins=100,
            alpha=0.8,
        )

        for size in TARGET_SIZES:
            plt.axvline(
                size,
                linestyle="--",
                label=f"{size}px",
            )

        plt.xlabel("Côté court (pixels)")
        plt.ylabel("Nombre d'images")

        plt.title(f"Distribution du côté court — {dataset}")

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir / f"hist_short_side_{safe_name}.png",
            dpi=200,
        )

        plt.close()


# ============================================================
# MAIN
# ============================================================


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("ANALYSE DES DIMENSIONS DU DATASET")
    print("=" * 70)

    print(f"\nDataset : {DATASET_DIR}")

    df = collect_image_sizes(DATASET_DIR)

    if df.empty:
        raise RuntimeError("Aucune image trouvée dans le dataset.")

    # --------------------------------------------------------
    # Stats globales
    # --------------------------------------------------------

    print_stats(df)

    # --------------------------------------------------------
    # Stats par dataset
    # --------------------------------------------------------

    dataset_stats = dataset_statistics(df)

    print_dataset_stats(dataset_stats)

    # --------------------------------------------------------
    # Détection des images hors norme
    # --------------------------------------------------------

    outliers, outlier_summary = detect_size_outliers(
        df,
        iqr_factor=1.5,
    )
    # --------------------------------------------------------
    # Résolutions DINOv3 par dataset
    # --------------------------------------------------------

    resolution_results = analyse_target_sizes(df)

    # --------------------------------------------------------
    # Stats par classe
    # --------------------------------------------------------

    class_stats = class_statistics(df)

    # --------------------------------------------------------
    # Stats classe × dataset
    # --------------------------------------------------------

    class_dataset_stats = class_dataset_statistics(df)

    # --------------------------------------------------------
    # Matrice classe × dataset
    # --------------------------------------------------------

    matrix = create_class_dataset_matrix(df)

    print_single_dataset_classes(matrix)

    # --------------------------------------------------------
    # Sauvegardes
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_DIR / "image_dimensions.csv",
        index=False,
    )

    dataset_stats.to_csv(
        OUTPUT_DIR / "dimensions_par_dataset.csv",
    )

    resolution_results.to_csv(
        OUTPUT_DIR / "comparaison_resolutions_dinov3_par_dataset.csv",
        index=False,
    )

    class_stats.to_csv(
        OUTPUT_DIR / "dimensions_par_classe.csv",
    )

    class_dataset_stats.to_csv(
        OUTPUT_DIR / "dimensions_par_classe_et_dataset.csv",
        index=False,
    )

    matrix.to_csv(
        OUTPUT_DIR / "matrice_classe_dataset.csv",
    )

    outliers.to_csv(
        OUTPUT_DIR / "images_dimensions_hors_norme.csv",
        index=False,
    )

    outlier_summary.to_csv(
        OUTPUT_DIR / "resume_images_hors_norme.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Graphiques
    # --------------------------------------------------------

    create_plots(
        df,
        OUTPUT_DIR,
    )

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)

    print(f"\nRésultats disponibles dans : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
