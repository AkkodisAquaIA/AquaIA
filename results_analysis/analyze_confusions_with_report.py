import re
import numpy as np
import pandas as pd

Path_results = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/test_dinov3/unfreeze1/focal_05/20260602-115513/inference_outputs"
REPORT_FILE = Path_results + "/classification_report.txt"
CONFUSION_MATRIX_FILE = Path_results + "/confusion_matrix.npy"
OUTPUT_FILE = Path_results + "/confusion_analysis.csv"

def parse_classification_report(report_file):
    rows = []

    with open(report_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            match = re.match(
                r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$",
                line
            )

            if match:
                class_name = match.group(1)

                if class_name in ["accuracy", "macro avg", "weighted avg"]:
                    continue

                rows.append({
                    "class": class_name,
                    "precision": float(match.group(2)),
                    "recall": float(match.group(3)),
                    "f1_score": float(match.group(4)),
                    "total": int(match.group(5))
                })

    return pd.DataFrame(rows)


def get_top_confusions(cm, class_names, top_k=4):
    results = []

    for i, class_name in enumerate(class_names):
        row = cm[i].copy()

        total = int(row.sum())
        correct = int(row[i])
        misclassified = total - correct

        # Ignorer les bonnes prédictions
        row[i] = -1

        # Les top_k plus grosses confusions
        top_indices = np.argsort(row)[-top_k:][::-1]

        result = {
            "class": class_name,
            "nb_images_mal_predites": misclassified
        }

        for rank, j in enumerate(top_indices, start=1):
            result[f"confusion_{rank}_class"] = class_names[j]
            result[f"confusion_{rank}_count"] = int(cm[i, j])

        results.append(result)

    return pd.DataFrame(results)


def main():
    report_df = parse_classification_report(REPORT_FILE)
    cm = np.load(CONFUSION_MATRIX_FILE)

    class_names = report_df["class"].tolist()

    if cm.shape[0] != len(class_names):
        raise ValueError(
            f"Problème de dimensions : matrice {cm.shape}, "
            f"mais {len(class_names)} classes dans le rapport."
        )

    confusion_df = get_top_confusions(cm, class_names, top_k=4)

    final_df = report_df.merge(confusion_df, on="class")

    final_df = final_df[
        [
            "class",
            "precision",
            "recall",
            "f1_score",
            "total",
            "nb_images_mal_predites",

            "confusion_1_class",
            "confusion_1_count",

            "confusion_2_class",
            "confusion_2_count",

            "confusion_3_class",
            "confusion_3_count",

            "confusion_4_class",
            "confusion_4_count",
        ]
    ]

    # Trier par précision croissante
    final_df = final_df.sort_values(by="precision", ascending=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Analyse sauvegardée dans : {OUTPUT_FILE}")
    print(final_df.head(20))


if __name__ == "__main__":
    main()