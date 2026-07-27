# Author : GPT 5.6 Sol High
# Samples approximately 1% of the images and labels from each dataset split.
# Keeps image/label pairs matched through their shared file stem.
# Ensures that 'train', 'val', and 'test' each contain all 80 COCO classes.
# Preserves the original 'images/<split>' and 'labels/<split>' structure.
# Validates pair counts and class coverage after copying.

import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(
    r"C:\Users\zhijian.zhou\OneDrive - Akkodis\Travail\10_AquaIA"
    r"\01_Git\AquaIA\datasets\coco_custom_match"
)
DEFAULT_DESTINATION = Path(
    r"C:\Users\zhijian.zhou\OneDrive - Akkodis\Travail\10_AquaIA"
    r"\01_Git\AquaIA\datasets\coco_cus_mat_hun"
)
SPLITS = ("train", "val", "test")
NUM_CLASSES = 80
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def numeric_sort_key(path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def read_classes(label_path):
    classes = set()
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError as error:
                raise ValueError(f"Invalid class ID in {label_path}, line {line_number}: {parts[0]}") from error
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"Class ID {class_id} in {label_path} is outside 0-{NUM_CLASSES - 1}")
            classes.add(class_id)
    return classes


def build_image_index(image_dir):
    image_index = {}
    for image_path in image_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem in image_index:
            raise ValueError(f"Duplicate image stem in {image_dir}: {image_path.stem}")
        image_index[image_path.stem] = image_path
    return image_index


def records_from_cache(cache_path, label_dir, image_index):
    cache = np.load(cache_path, allow_pickle=True).item()
    records = []
    for cached_label in cache["labels"]:
        stem = Path(cached_label["im_file"]).stem
        label_path = label_dir / f"{stem}.txt"
        image_path = image_index.get(stem)
        if image_path is None or not label_path.is_file():
            continue
        classes = {int(class_id) for class_id in np.asarray(cached_label["cls"]).reshape(-1).tolist()}
        invalid_classes = classes - set(range(NUM_CLASSES))
        if invalid_classes:
            raise ValueError(f"Classes outside 0-{NUM_CLASSES - 1} in {label_path}: {sorted(invalid_classes)}")
        records.append((image_path, label_path, classes))
    return records


def records_from_txt(label_dir, image_index):
    label_paths = sorted(label_dir.glob("*.txt"), key=numeric_sort_key)
    with ThreadPoolExecutor(max_workers=16) as executor:
        class_sets = list(executor.map(read_classes, label_paths))

    records = []
    for label_path, classes in zip(label_paths, class_sets):
        image_path = image_index.get(label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No image found for label: {label_path}")
        records.append((image_path, label_path, classes))
    return records


def load_records(source_root, split):
    image_dir = source_root / "images" / split
    label_dir = source_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing images/labels directory for split: {split}")

    image_index = build_image_index(image_dir)
    cache_path = source_root / "labels" / f"{split}.cache"
    if cache_path.is_file():
        print(f"[{split}] Loading label metadata from {cache_path.name}...")
        records = records_from_cache(cache_path, label_dir, image_index)
    else:
        print(f"[{split}] No cache found; reading TXT labels...")
        records = records_from_txt(label_dir, image_index)

    if len(records) != len(image_index):
        raise ValueError(f"[{split}] Image/label count mismatch: {len(image_index)} images, {len(records)} matched labels")
    return records


def select_records(records, fraction, seed, split):
    target_count = max(1, round(len(records) * fraction))
    all_classes = set().union(*(record[2] for record in records))
    missing_from_source = set(range(NUM_CLASSES)) - all_classes
    if missing_from_source:
        raise ValueError(f"[{split}] Source split does not contain all {NUM_CLASSES} classes; missing: {sorted(missing_from_source)}")

    rng = random.Random(f"{seed}:{split}")
    candidate_indices = list(range(len(records)))
    rng.shuffle(candidate_indices)

    selected_indices = []
    selected_set = set()
    uncovered = set(range(NUM_CLASSES))
    while uncovered:
        best_index = max(
            (index for index in candidate_indices if index not in selected_set),
            key=lambda index: len(records[index][2] & uncovered),
        )
        newly_covered = records[best_index][2] & uncovered
        if not newly_covered:
            raise RuntimeError(f"[{split}] Could not cover classes: {sorted(uncovered)}")
        selected_indices.append(best_index)
        selected_set.add(best_index)
        uncovered -= newly_covered

    if len(selected_indices) > target_count:
        raise ValueError(f"[{split}] Covering all {NUM_CLASSES} classes requires at least {len(selected_indices)} images, which exceeds the 1% target of {target_count} images")

    remaining_indices = [index for index in candidate_indices if index not in selected_set]
    selected_indices.extend(remaining_indices[: target_count - len(selected_indices)])
    selected = [records[index] for index in selected_indices]
    return sorted(selected, key=lambda record: numeric_sort_key(record[0]))


def copy_record(record, destination_root, split):
    image_path, label_path, _ = record
    shutil.copy2(image_path, destination_root / "images" / split / image_path.name)
    shutil.copy2(label_path, destination_root / "labels" / split / label_path.name)


def copy_split(selected, destination_root, split):
    (destination_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (destination_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda record: copy_record(record, destination_root, split),
                selected,
            )
        )


def validate_split(destination_root, split, expected_count):
    image_dir = destination_root / "images" / split
    label_dir = destination_root / "labels" / split
    image_stems = {path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    label_paths = sorted(label_dir.glob("*.txt"), key=numeric_sort_key)
    label_stems = {path.stem for path in label_paths}

    if image_stems != label_stems:
        raise ValueError(
            f"[{split}] Output image/label pairs do not match. Images without labels: {sorted(image_stems - label_stems)[:10]}; labels without images: {sorted(label_stems - image_stems)[:10]}"
        )
    if len(image_stems) != expected_count:
        raise ValueError(f"[{split}] Expected {expected_count} pairs, found {len(image_stems)}")

    covered_classes = set()
    for label_path in label_paths:
        covered_classes.update(read_classes(label_path))
    missing_classes = set(range(NUM_CLASSES)) - covered_classes
    if missing_classes:
        raise ValueError(f"[{split}] Output is missing classes: {sorted(missing_classes)}")
    print(f"[{split}] VERIFIED: {len(image_stems)} image/label pairs, all {NUM_CLASSES} classes covered")


def parse_args():
    parser = argparse.ArgumentParser(description=("Sample approximately 1% of each YOLO-format COCO split while ensuring that every split contains all 80 classes."))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source.resolve()
    destination_root = args.destination.resolve()

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in the interval (0, 1]")
    if source_root == destination_root:
        raise ValueError("Source and destination must be different directories")
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination_root}. Remove it or choose another destination.")

    print(f"Source:      {source_root}")
    print(f"Destination: {destination_root}")
    print(f"Fraction:    {args.fraction:.2%}")
    print(f"Seed:        {args.seed}")

    for split in SPLITS:
        records = load_records(source_root, split)
        selected = select_records(
            records=records,
            fraction=args.fraction,
            seed=args.seed,
            split=split,
        )
        actual_fraction = len(selected) / len(records)
        print(f"[{split}] Selected {len(selected)}/{len(records)} ({actual_fraction:.3%}); copying...")
        copy_split(selected, destination_root, split)
        validate_split(destination_root, split, len(selected))

    print("Sampling and validation completed successfully.")


if __name__ == "__main__":
    main()
