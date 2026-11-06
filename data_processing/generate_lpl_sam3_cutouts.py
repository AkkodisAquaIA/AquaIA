"""Relance SAM3 sur tous les splits LPL en conservant leur séparation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/LPL_dataset_final/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/LPL_dataset_cutouts_sam3_by_split"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--prompt", default="insect")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--detection-max-size", type=int, default=3000)
    parser.add_argument("--context-ratio", type=float, default=3.0)
    parser.add_argument("--refine-threshold", type=float, default=0.3)
    parser.add_argument("--refine-max-size", type=int, default=1008)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stems_with_cutouts(split_output: Path) -> set[str]:
    stems = set()
    for cutout in (split_output / "macroinvertebrates").glob("*.png"):
        stems.add(cutout.stem.rsplit("_", 1)[0])
    return stems


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def cutout_counts_by_stem(split_output: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cutout in (split_output / "macroinvertebrates").glob("*.png"):
        source_stem = cutout.stem.rsplit("_", 1)[0]
        counts[source_stem] = counts.get(source_stem, 0) + 1
    return counts


def main():
    from data_augmentation.detection_augmentation.copy_and_paste_augmentation import create_cutouts_sam3

    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Le dossier existe déjà : {output_dir}. Utiliser --overwrite pour le remplacer.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("SAM3 complet doit être lancé avec CUDA pour éviter un traitement CPU très long.")
    print(f"Chargement unique de {args.model_id} sur {device}...")
    processor = Sam3Processor.from_pretrained(args.model_id)
    model = Sam3Model.from_pretrained(args.model_id).to(device).eval()

    report = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "detection_max_size": args.detection_max_size,
        "context_ratio": args.context_ratio,
        "refine_threshold": args.refine_threshold,
        "refine_max_size": args.refine_max_size,
        "splits": {},
    }
    for split in args.splits:
        source_dir = dataset_root / "images" / split
        split_output = output_dir / split
        print(f"\n=== SAM3 split {split}: {source_dir} ===")
        create_cutouts_sam3(
            source_dir=source_dir,
            output_dir=split_output,
            model_id=args.model_id,
            prompt=args.prompt,
            threshold=args.threshold,
            detection_max_size=args.detection_max_size,
            context_ratio=args.context_ratio,
            refine_threshold=args.refine_threshold,
            refine_max_size=args.refine_max_size,
            full_resolution=True,
            class_name="macroinvertebrates",
            create_labels=True,
            model=model,
            processor=processor,
            device=device,
        )
        image_stems = {path.stem for path in source_dir.iterdir() if path.is_file()}
        represented = stems_with_cutouts(split_output)
        generated_counts = cutout_counts_by_stem(split_output)
        expected_counts = {
            stem: count_nonempty_lines(dataset_root / "labels" / split / f"{stem}.txt")
            for stem in sorted(image_stems)
        }
        count_mismatches = {
            stem: {"annotations": expected_counts[stem], "sam3_cutouts": generated_counts.get(stem, 0)}
            for stem in sorted(image_stems)
            if expected_counts[stem] != generated_counts.get(stem, 0)
        }
        report["splits"][split] = {
            "source_images": len(image_stems),
            "represented_source_images": len(represented),
            "cutouts": len(list((split_output / "macroinvertebrates").glob("*.png"))),
            "annotated_objects": sum(expected_counts.values()),
            "images_without_cutout": sorted(image_stems - represented),
            "images_with_count_mismatch": count_mismatches,
        }
        (output_dir / "sam3_generation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Rapport : {output_dir / 'sam3_generation_report.json'}")


if __name__ == "__main__":
    main()
