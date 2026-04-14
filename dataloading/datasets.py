import os
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from detection.utils.config_utils import load_class_names

class BaseDetectionDataset(Dataset):
    """
    Base class shared by NPY / PIL / RAM datasets.

    Handles:
    - label loading
    - statistics (mean/std)
    - normalization
    """

    def __init__(self, dataset_root: str, stats_file: str = "stats.npy", device: str = "cpu"):
        self.dataset_root = Path(dataset_root)
        self.stats_file = stats_file
        self.device = torch.device(device)
        self.num_classes = 0
        self.load_stats()
        self.class_names = load_class_names(dataset_root)
        self.image_files = self.list_image_files()
        self.load_targets()

    def load_stats(self) -> None:
        stats_path = self.dataset_root / self.stats_file

        if stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True).item()
            self.stats = {
                "mean": torch.tensor(stats["mean"], dtype=torch.float32, device=self.device).view(-1, 1, 1),
                "std": torch.clamp(
                    torch.tensor(stats["std"], dtype=torch.float32, device=self.device).view(-1, 1, 1),
                    min=1e-6,
                ),
            }
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(self.device)
        mean = self.stats["mean"].to(dtype=img.dtype)
        std = self.stats["std"].to(dtype=img.dtype)
        return (img - mean) / std

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def list_image_files(self) -> List[str]:
        image_dir = self.dataset_root / "images"
        return sorted(glob.glob(str(image_dir / "*.*")))

    @staticmethod
    def _numeric_sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def get_sorted_label_files(self) -> List[str]:
        label_dir = self.dataset_root / "labels"
        label_files = [str(path) for path in label_dir.glob("*.txt")]
        return sorted(label_files, key=lambda path: self._numeric_sort_key(Path(path)))

    def parse_label_line(self, line: str):
        class_id, x_center, y_center, width, height = line.split()
        return int(class_id), [float(x_center), float(y_center), float(width), float(height)]

    def read_target(self, label_path: str):
        labels = []
        boxes = []
        if label_path is not None and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id, bbox = self.parse_label_line(line)
                    labels.append(class_id)
                    boxes.append(bbox)
        return {
            "labels": torch.tensor(labels, dtype=torch.int64, device=self.device),
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=self.device).reshape(-1, 4),
        }

    def load_targets(self) -> None:
        label_files = self.get_sorted_label_files()
        if not label_files:
            raise FileNotFoundError(f"No label files found under {self.dataset_root / 'labels'}")
        self.targets = [self.read_target(path) for path in label_files]
        if len(self.targets) != len(self.image_files):
            raise ValueError(
                f"Label count mismatch for {self.dataset_root}: found {len(self.targets)} label files for {len(self.image_files)} images."
            )
        self.num_classes = 1 + max((int(t["labels"].max().item()) for t in self.targets if len(t["labels"]) > 0), default=0)

class NpyDetectionDataset(BaseDetectionDataset):
    def __init__(self, dataset_root: str, stats_file: str = "stats.npy", device: str = "cpu",):
        super().__init__(dataset_root=dataset_root, stats_file=stats_file, device=device)
        img_file = self.dataset_root / "npy_images.npy"
        self.imgs = np.load(img_file).astype(np.float32)
        if len(self.imgs) != len(self.image_files):
            raise ValueError(
                f"Sample count mismatch for {self.dataset_root}: found {len(self.image_files)} images but {len(self.imgs)} NPY samples."
            )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.imgs[idx]).float()
        img = self.normalize_img(img)
        tgt = self.targets[idx]
        img_file = self.image_files[idx] if idx < len(self.image_files) else str(idx)
        return img, tgt, img_file


class PilDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_root: str,
        img_size: Tuple[int, int],
        stats_file: str = "stats.npy",
        device: str = "cpu",
    ):
        super().__init__(dataset_root=dataset_root, stats_file=stats_file, load_targets=False, device=device)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_files[idx]).convert("RGB").resize(
            self.img_size
        )
        img = np.array(img, dtype=np.float32) / 255.0
        img = self.to_tensor(img)
        img = self.normalize_img(img)
        return img

class RAMDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_root: str,
        img_size: Tuple[int, int],
        stats_file: str = "stats.npy",
        device: str = "cpu",
    ):
        super().__init__(dataset_root=dataset_root, stats_file=stats_file, load_targets=False, device=device)
        self.img_size = img_size

        imgs = []
        for f in self.image_files:
            img = Image.open(f).convert("RGB").resize(self.img_size)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)

        self.imgs = np.stack(imgs, axis=0)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = self.to_tensor(self.imgs[idx])
        img = self.normalize_img(img)
        return img


def detection_collate_fn(batch):
    images, targets, image_files = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets), list(image_files)

def sample_indices(dataset_size, num_samples, seed):
    rng = random.Random(seed)
    sample_size = min(num_samples, dataset_size)
    return sorted(rng.sample(range(dataset_size), sample_size))


def sample_dataset(dataset, num_samples, seed):
    sampled_indices = sample_indices(len(dataset), num_samples, seed)
    samples = [dataset[index] for index in sampled_indices]
    images = torch.stack([sample[0] for sample in samples], dim=0)
    img_files = [sample[2] for sample in samples]
    return images, img_files
