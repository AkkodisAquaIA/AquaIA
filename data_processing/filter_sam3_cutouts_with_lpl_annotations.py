"""Filtre et classe les cutouts SAM3 à partir des annotations YOLO LPL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.crop_detection_to_classification import (  # noqa: E402
    build_conversion_table,
    load_class_ids,
    load_taxonomy_prefixes,
    read_yolo_labels,
)
from data_augmentation.detection_augmentation.copy_and_paste_augmentation import (  # noqa: E402
    PNG_COMPRESS_LEVEL,
    create_sam3_cutout_from_box,
)

SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam3-dir",
        type=Path,
        default=Path("datasets/LPL_dataset_cutouts_sam3_by_split"),
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("datasets/LPL_dataset_final/dataset_LPL_no_split/Annotations"),
    )
    parser.add_argument(
        "--split-images-dir",
        type=Path,
        default=Path("datasets/LPL_dataset_final/dataset/images"),
    )
    parser.add_argument(
        "--by-split-output",
        type=Path,
        default=Path("datasets/LPL_dataset_cutouts_sam3_filtered_by_split"),
    )
    parser.add_argument(
        "--by-taxon-output",
        type=Path,
        default=Path("datasets/LPL_dataset_cutouts_sam3_filtered_by_taxon"),
    )
    parser.add_argument("--min-detection-coverage", type=float, default=0.5)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument(
        "--fill-unmatched",
        action="store_true",
        help="Relance SAM3 avec chaque bounding box non retrouvée comme prompt.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def yolo_to_corners(box):
    class_id, center_x, center_y, width, height = box
    return class_id, (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def overlap_metrics(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "detection_coverage": intersection / first_area if first_area else 0.0,
        "annotation_coverage": intersection / second_area if second_area else 0.0,
    }


def match_detections(detections, annotations, min_detection_coverage, min_iou):
    """Associe au plus une détection SAM3 à chaque annotation LPL."""
    candidates = []
    for detection_index, detection in enumerate(detections):
        _, detection_box = yolo_to_corners(detection)
        for annotation_index, annotation in enumerate(annotations):
            _, annotation_box = yolo_to_corners(annotation)
            metrics = overlap_metrics(detection_box, annotation_box)
            if metrics["detection_coverage"] >= min_detection_coverage and metrics["iou"] >= min_iou:
                candidates.append(
                    (
                        metrics["iou"],
                        metrics["detection_coverage"],
                        metrics["annotation_coverage"],
                        detection_index,
                        annotation_index,
                    )
                )

    matches = {}
    used_annotations = set()
    for iou, detection_coverage, annotation_coverage, detection_index, annotation_index in sorted(
        candidates,
        reverse=True,
    ):
        if detection_index in matches or annotation_index in used_annotations:
            continue
        matches[detection_index] = {
            "annotation_index": annotation_index,
            "iou": iou,
            "detection_coverage": detection_coverage,
            "annotation_coverage": annotation_coverage,
        }
        used_annotations.add(annotation_index)
    return matches, used_annotations


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_output(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Le dossier existe déjà : {path}. Utiliser --overwrite pour le remplacer.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def source_stems(split_images_dir, split):
    return sorted(path.stem for path in (split_images_dir / split).iterdir() if path.is_file())


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    sam3_dir = args.sam3_dir.resolve()
    annotations_dir = args.annotations_dir.resolve()
    split_images_dir = args.split_images_dir.resolve()
    by_split_output = args.by_split_output.resolve()
    by_taxon_output = args.by_taxon_output.resolve()

    for threshold_name in ("min_detection_coverage", "min_iou"):
        threshold = getattr(args, threshold_name)
        if not 0 <= threshold <= 1:
            raise ValueError(f"{threshold_name} doit être compris entre 0 et 1.")

    prepare_output(by_split_output, args.overwrite)
    prepare_output(by_taxon_output, args.overwrite)

    class_ids, class_sources = load_class_ids(yaml_path=annotations_dir / "data.yaml")
    prefixes = load_taxonomy_prefixes(PROJECT_ROOT / "data_processing/resources/taxonomy_prefixes.json")
    conversion_rows, id_to_taxon = build_conversion_table(class_ids, class_sources, prefixes)

    manifest = []
    rejected = []
    report = {
        "sam3_dir": str(sam3_dir),
        "annotations_dir": str(annotations_dir),
        "min_detection_coverage": args.min_detection_coverage,
        "min_iou": args.min_iou,
        "fill_unmatched": args.fill_unmatched,
        "splits": {},
    }

    model = processor = device = None
    if args.fill_unmatched:
        import torch
        from transformers import Sam3Model, Sam3Processor

        if not torch.cuda.is_available():
            raise RuntimeError("--fill-unmatched requiert CUDA.")
        device = "cuda"
        print("Chargement unique de facebook/sam3 pour compléter les annotations manquantes...")
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()

    for split in SPLITS:
        totals = {
            "source_images": 0,
            "sam3_cutouts": 0,
            "kept_cutouts": 0,
            "rejected_cutouts": 0,
            "annotations": 0,
            "matched_annotations": 0,
            "unmatched_annotations": 0,
            "box_prompt_failures": 0,
            "unmatched_annotations_by_taxon": Counter(),
            "taxa": set(),
        }
        kept_labels_dir = by_split_output / split / "labels"
        kept_labels_dir.mkdir(parents=True, exist_ok=True)

        for stem in source_stems(split_images_dir, split):
            sam_label_path = sam3_dir / split / "labels" / f"{stem}.txt"
            annotation_path = annotations_dir / "labels" / "train" / f"{stem}.txt"
            cutouts = sorted((sam3_dir / split / "macroinvertebrates").glob(f"{stem}_*.png"))
            if not sam_label_path.is_file():
                raise FileNotFoundError(f"Labels SAM3 absents : {sam_label_path}")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Annotations LPL absentes : {annotation_path}")

            detections = read_yolo_labels(sam_label_path)
            annotations = read_yolo_labels(annotation_path)
            if len(cutouts) != len(detections):
                raise ValueError(
                    f"{stem}: {len(cutouts)} cutout(s), mais {len(detections)} label(s) SAM3."
                )

            matches, used_annotations = match_detections(
                detections,
                annotations,
                args.min_detection_coverage,
                args.min_iou,
            )
            kept_label_lines = []
            totals["source_images"] += 1
            totals["sam3_cutouts"] += len(cutouts)
            totals["annotations"] += len(annotations)

            for detection_index, (cutout_path, detection) in enumerate(zip(cutouts, detections)):
                match = matches.get(detection_index)
                if match is None:
                    rejected.append(
                        {
                            "split": split,
                            "source_image": stem,
                            "cutout": str(cutout_path),
                            "reason": "aucune_bounding_box_compatible",
                        }
                    )
                    totals["rejected_cutouts"] += 1
                    continue

                annotation_index = match["annotation_index"]
                class_id = annotations[annotation_index][0]
                taxon = id_to_taxon[class_id]
                split_destination = by_split_output / split / "macroinvertebrates" / cutout_path.name
                taxon_destination = by_taxon_output / taxon / split / cutout_path.name
                link_or_copy(cutout_path, split_destination)
                link_or_copy(cutout_path, taxon_destination)
                kept_label_lines.append(" ".join(map(str, detection)))
                totals["kept_cutouts"] += 1
                totals["taxa"].add(taxon)
                manifest.append(
                    {
                        "split": split,
                        "source_image": stem,
                        "cutout": cutout_path.name,
                        "class_id": class_id,
                        "taxon": taxon,
                        "annotation_index": annotation_index,
                        "iou": f'{match["iou"]:.6f}',
                        "detection_coverage": f'{match["detection_coverage"]:.6f}',
                        "annotation_coverage": f'{match["annotation_coverage"]:.6f}',
                        "match_method": "text_prompt_and_bbox_filter",
                    }
                )

            if args.fill_unmatched and len(used_annotations) < len(annotations):
                image_paths = list((split_images_dir / split).glob(f"{stem}.*"))
                if len(image_paths) != 1:
                    raise ValueError(f"Image source ambiguë ou absente pour {split}/{stem}: {image_paths}")
                from PIL import Image
                import numpy as np

                with Image.open(image_paths[0]) as image:
                    original_image = np.asarray(image.convert("RGB"))

                for annotation_index, annotation in enumerate(annotations):
                    if annotation_index in used_annotations:
                        continue
                    cutout = create_sam3_cutout_from_box(
                        original_image,
                        annotation,
                        model,
                        processor,
                        device,
                    )
                    if cutout is None:
                        totals["box_prompt_failures"] += 1
                        continue

                    class_id = annotation[0]
                    taxon = id_to_taxon[class_id]
                    cutout_name = f"{stem}_bbox_{annotation_index + 1:03d}.png"
                    split_destination = by_split_output / split / "macroinvertebrates" / cutout_name
                    taxon_destination = by_taxon_output / taxon / split / cutout_name
                    split_destination.parent.mkdir(parents=True, exist_ok=True)
                    cutout.save(split_destination, compress_level=PNG_COMPRESS_LEVEL)
                    link_or_copy(split_destination, taxon_destination)
                    kept_label_lines.append("0 " + " ".join(map(str, annotation[1:])))
                    totals["kept_cutouts"] += 1
                    totals["taxa"].add(taxon)
                    used_annotations.add(annotation_index)
                    manifest.append(
                        {
                            "split": split,
                            "source_image": stem,
                            "cutout": cutout_name,
                            "class_id": class_id,
                            "taxon": taxon,
                            "annotation_index": annotation_index,
                            "iou": "",
                            "detection_coverage": "",
                            "annotation_coverage": "",
                            "match_method": "annotation_box_prompt",
                        }
                    )

            totals["matched_annotations"] += len(used_annotations)
            for annotation_index, annotation in enumerate(annotations):
                if annotation_index not in used_annotations:
                    totals["unmatched_annotations_by_taxon"][id_to_taxon[annotation[0]]] += 1

            (kept_labels_dir / f"{stem}.txt").write_text(
                "\n".join(kept_label_lines) + ("\n" if kept_label_lines else ""),
                encoding="utf-8",
            )

        totals["unmatched_annotations"] = totals["annotations"] - totals["matched_annotations"]
        totals["unmatched_annotations_by_taxon"] = dict(
            sorted(totals["unmatched_annotations_by_taxon"].items())
        )
        totals["taxa"] = sorted(totals["taxa"])
        report["splits"][split] = totals

    annotated_taxa = {
        id_to_taxon[class_id]
        for split in SPLITS
        for stem in source_stems(split_images_dir, split)
        for class_id, *_ in read_yolo_labels(annotations_dir / "labels" / "train" / f"{stem}.txt")
    }
    represented_taxa = {row["taxon"] for row in manifest}
    report["summary"] = {
        "annotated_taxa": len(annotated_taxa),
        "represented_taxa": len(represented_taxa),
        "taxa_without_sam3_cutout": sorted(annotated_taxa - represented_taxa),
    }

    manifest_fields = [
        "split",
        "source_image",
        "cutout",
        "class_id",
        "taxon",
        "annotation_index",
        "iou",
        "detection_coverage",
        "annotation_coverage",
        "match_method",
    ]
    write_csv(by_split_output / "manifest.csv", manifest, manifest_fields)
    write_csv(by_split_output / "rejected_cutouts.csv", rejected, ["split", "source_image", "cutout", "reason"])
    write_csv(by_split_output / "table_conversion_classes.csv", conversion_rows, list(conversion_rows[0]))
    report_path = by_split_output / "filter_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Dataset filtré par split : {by_split_output}")
    print(f"Dataset filtré par taxon : {by_taxon_output}")
    print(f"Rapport : {report_path}")


if __name__ == "__main__":
    main()
