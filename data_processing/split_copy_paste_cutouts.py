"""Range les cutouts copy-paste selon le split de leur image LPL source.

Les noms produits par SAM3 suivent ``<image_source>_<instance>.png``. Ce script
recoupe donc chaque PNG avec ``dataset/images/{train,val,test}``, crée un dossier
par split et écrit un rapport d'audit. Les fichiers sont créés comme hard links
quand le système le permet : ils sont directement lisibles et exportables, sans
dupliquer les centaines de Mo de pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutout-dir", type=Path, default=Path("datasets/LPL_dataset_cutouts_single"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/LPL_dataset_final/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/LPL_dataset_cutouts_by_split"))
    parser.add_argument("--overwrite", action="store_true", help="Remplace un dossier de sortie déjà existant.")
    return parser.parse_args()


def image_stems_by_split(dataset_root: Path) -> dict[str, set[str]]:
    stems = {}
    owners = {}
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {image_dir}")
        stems[split] = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        for stem in stems[split]:
            if stem in owners:
                raise ValueError(f"L'image source {stem!r} appartient à plusieurs splits.")
            owners[stem] = split
    return stems


def source_stem(cutout: Path, known_stems: set[str]) -> str:
    matches = [stem for stem in known_stems if cutout.stem.startswith(f"{stem}_")]
    if len(matches) != 1:
        raise ValueError(f"Impossible d'identifier une source unique pour {cutout}")
    return matches[0]


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_split_cutouts(cutout_dir: Path, dataset_root: Path, output_dir: Path, overwrite: bool = False):
    cutout_dir = cutout_dir.resolve()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if not cutout_dir.is_dir():
        raise FileNotFoundError(f"Dossier de cutouts introuvable : {cutout_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Le dossier existe déjà : {output_dir}. Utiliser --overwrite pour le remplacer.")
        shutil.rmtree(output_dir)

    stems = image_stems_by_split(dataset_root)
    owner_by_stem = {stem: split for split, values in stems.items() for stem in values}
    cutouts = sorted(cutout_dir.rglob("*.png"))
    if not cutouts:
        raise FileNotFoundError(f"Aucun cutout PNG dans {cutout_dir}")

    represented_sources = defaultdict(set)
    cutout_counts = defaultdict(int)
    link_modes = defaultdict(int)
    try:
        for cutout in cutouts:
            stem = source_stem(cutout, set(owner_by_stem))
            split = owner_by_stem[stem]
            relative = cutout.relative_to(cutout_dir)
            mode = link_or_copy(cutout, output_dir / split / relative)
            represented_sources[split].add(stem)
            cutout_counts[split] += 1
            link_modes[mode] += 1

        source_labels = cutout_dir / "labels"
        for split in SPLITS:
            for stem in sorted(represented_sources[split]):
                label = source_labels / f"{stem}.txt"
                if label.is_file():
                    link_or_copy(label, output_dir / split / "labels" / label.name)

        report = {
            "source_cutout_dir": str(cutout_dir),
            "dataset_root": str(dataset_root),
            "policy": {
                "training_may_use": "train only",
                "validation": "real images/val only",
                "test_cutouts": "isolated and forbidden during training",
            },
            "splits": {
                split: {
                    "dataset_source_images": len(stems[split]),
                    "represented_source_images": len(represented_sources[split]),
                    "cutouts": cutout_counts[split],
                    "missing_source_images": sorted(stems[split] - represented_sources[split]),
                }
                for split in SPLITS
            },
            "storage": dict(link_modes),
        }
        (output_dir / "split_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def main():
    args = parse_args()
    report = prepare_split_cutouts(args.cutout_dir, args.dataset_root, args.output_dir, args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Cutouts séparés dans : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
