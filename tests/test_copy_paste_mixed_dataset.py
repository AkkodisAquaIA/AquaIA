import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_augmentation.detection_augmentation.copy_paste_dataset import (
    MixedRealCopyPasteYOLODataset,
    YOLOCopyPasteDataset,
)
from data_processing.split_copy_paste_cutouts import prepare_split_cutouts


class _FakeDataset:
    collate_fn = None

    def __init__(self, prefix, length):
        self.prefix = prefix
        self.length = length
        self.epochs = []
        self.labels = [{"bboxes": np.zeros((0, 4)), "cls": np.zeros((0, 1))}]
        self.im_files = [f"{prefix}-{index}" for index in range(length)]
        self.mosaic = False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"source": self.prefix, "index": index}

    def get_image_and_label(self, index):
        return {"source": self.prefix, "index": index}

    def update_labels_info(self, label):
        return {**label, "labels_updated": True}

    def transforms(self, label):
        return {**label, "yolo_augmented": True}

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def _save_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(path)


def _save_background(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (240, 240, 240)).save(path)


def test_prepare_split_cutouts_keeps_test_out_of_train(tmp_path):
    dataset_root = tmp_path / "dataset"
    for split, stem in (("train", "train_image"), ("val", "val_image"), ("test", "test_image")):
        image_path = dataset_root / "images" / split / f"{stem}.TIF"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()

    cutout_dir = tmp_path / "cutouts"
    for stem in ("train_image", "val_image", "test_image"):
        _save_image(cutout_dir / "macroinvertebrates" / f"{stem}_001.png")

    output_dir = tmp_path / "split_cutouts"
    report = prepare_split_cutouts(cutout_dir, dataset_root, output_dir)

    assert (output_dir / "train/macroinvertebrates/train_image_001.png").is_file()
    assert not (output_dir / "train/macroinvertebrates/test_image_001.png").exists()
    assert (output_dir / "test/macroinvertebrates/test_image_001.png").is_file()
    assert report["splits"]["train"]["cutouts"] == 1
    assert json.loads((output_dir / "split_audit.json").read_text())["policy"]["training_may_use"] == "train only"


def test_mixed_dataset_contains_real_and_synthetic_and_forwards_epoch():
    real = _FakeDataset("real", 3)
    synthetic = _FakeDataset("synthetic", 2)
    mixed = MixedRealCopyPasteYOLODataset(real, synthetic)

    assert len(mixed) == 5
    assert mixed.real_count == 3
    assert mixed.synthetic_count == 2
    assert mixed[0]["source"] == "real"
    assert mixed[3]["source"] == "synthetic"
    assert mixed[3]["labels_updated"] is True
    assert mixed[3]["yolo_augmented"] is True

    mixed.set_epoch(7)
    assert synthetic.epochs == [7]


def test_mixed_dataset_can_limit_yolo_augmentations_to_real_images():
    real = _FakeDataset("real", 3)
    synthetic = _FakeDataset("synthetic", 2)
    mixed = MixedRealCopyPasteYOLODataset(
        real,
        synthetic,
        apply_yolo_augmentations_to_synthetic=False,
    )

    assert mixed[0]["source"] == "real"
    synthetic_sample = mixed[3]
    assert synthetic_sample["source"] == "synthetic"
    assert "labels_updated" not in synthetic_sample
    assert "yolo_augmented" not in synthetic_sample


def test_yolo_copy_paste_sample_has_same_keys_as_real_yolo_sample(tmp_path):
    cutout_dir = tmp_path / "cutouts"
    _save_image(cutout_dir / "macroinvertebrates" / "insect.png")
    background_dir = tmp_path / "backgrounds"
    for index in range(8):
        _save_background(background_dir / f"background_{index}.png")

    dataset = YOLOCopyPasteDataset(
        cutout_dir=cutout_dir,
        background_dir=background_dir,
        length=1,
        image_size=64,
        min_macroinvertebrates=1,
        max_macroinvertebrates=1,
        padding=0,
        cache_backgrounds=False,
    )

    sample = dataset[0]

    assert set(sample) == {
        "img",
        "cls",
        "bboxes",
        "batch_idx",
        "im_file",
        "ori_shape",
        "resized_shape",
    }
