# This script splits the 2017 Train into train and test sets
# Because the original 2017 Test is not publicly available
# With the original 2017 Val, we have train/val/test splits

# Brut data downloaded from https://cocodataset.org/#download
# 2017 Train images [118K/18GB]
# 2017 Val images [5K/1GB]
# 2017 Train/Val annotations [241MB]

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.converter import convert_coco

# Root dir, contains "train2017", "val2017" and "annotations" subdir
ROOT_DIR = "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/AquaIA/datasets/coco_raw"

# Train/Test split ratio
SPLIT_RATIO = 0.8


def create_dir_structure(base_path):
    """Create YOLO standard dir structure"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)


def copy_files(file_names, src_img_dir, src_lbl_dir, dest_base, split):
    """Copy images and corresponding labels to the target dir"""
    print(f"Building {split} dataset...")
    for fname in tqdm(file_names):
        img_name = fname + '.jpg'
        lbl_name = fname + '.txt'

        src_img = os.path.join(src_img_dir, img_name)
        src_lbl = os.path.join(src_lbl_dir, lbl_name)

        dest_img = os.path.join(dest_base, 'images', split, img_name)
        dest_lbl = os.path.join(dest_base, 'labels', split, lbl_name)

        if os.path.exists(src_img):
            shutil.copy2(src_img, dest_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dest_lbl)


def main():
    root_path = Path(ROOT_DIR).resolve()
    train_img_dir = root_path / "train2017"
    val_img_dir = root_path / "val2017"
    annotations_dir = root_path / "annotations"

    # Final dataset dir
    final_output_dir = root_path.parent / "coco_custom"

    print(f"Read input dataset dir: {root_path}")
    print(f"Output dataset dir: {final_output_dir}")
    print(f"Train/test ratio: {SPLIT_RATIO}\n")

    # If old split exists, clean and recreate
    if final_output_dir.exists():
        print(f"Detected existing {final_output_dir}, cleaning...")
        shutil.rmtree(final_output_dir)

    # Convet .json to .txt, no segments no keypoints only 80 detection classes
    convert_coco(labels_dir=str(annotations_dir), use_segments=False, use_keypoints=False, cls91to80=True)

    # Dynamic search: start from root_path, go up 3 levels, search globally for "coco_converted/labels"
    lbl_search_root = root_path.parents[2]
    found_paths = list(lbl_search_root.glob("**/coco_converted/labels"))
    if not found_paths:
        raise FileNotFoundError(f"Under {lbl_search_root} and its subdirectories not found 'coco_converted/labels' folder.")
    # Use the first found path
    generated_lbl_dir = found_paths[0]

    # Initialize target dir structure
    create_dir_structure(str(final_output_dir))

    # Val set
    val_lbl_dir = generated_lbl_dir / 'val2017'
    val_fnames = [os.path.splitext(f)[0] for f in os.listdir(str(val_img_dir)) if f.endswith('.jpg')]
    copy_files(val_fnames, str(val_img_dir), str(val_lbl_dir), str(final_output_dir), 'val')

    # Train test sets
    train_lbl_dir = generated_lbl_dir / 'train2017'
    train_fnames = [os.path.splitext(f)[0] for f in os.listdir(str(train_img_dir)) if f.endswith('.jpg')]

    random.seed(42)
    random.shuffle(train_fnames)

    split_idx = int(len(train_fnames) * SPLIT_RATIO)
    actual_train_fnames = train_fnames[:split_idx]
    actual_test_fnames = train_fnames[split_idx:]

    copy_files(actual_train_fnames, str(train_img_dir), str(train_lbl_dir), str(final_output_dir), 'train')
    copy_files(actual_test_fnames, str(train_img_dir), str(train_lbl_dir), str(final_output_dir), 'test')

    # Delete generated files when conversion .json -> .txt
    tmp_lbl_root = generated_lbl_dir.parent
    if tmp_lbl_root.exists():
        print(f"\nCleaning temporary folder: {tmp_lbl_root}")
        shutil.rmtree(tmp_lbl_root)

    print(f"\nCustom COCO dataset created at '{final_output_dir}'")
    print(f"Train: {len(actual_train_fnames)} images")
    print(f"Val: {len(val_fnames)} images")
    print(f"Test: {len(actual_test_fnames)} images")

if __name__ == '__main__':
    main()
