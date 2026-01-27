import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

@dataclass
class DatasetConfig:
    dataset_name: str
    root_folder: str = "datasets"
    img_size: Tuple[int, int] = (304, 304)
    stats_file: str = "stats.npy"


class BaseDetectionDataset(Dataset):
    """
    Base class shared by NPY / PIL / RAM datasets.

    Handles:
    - label loading
    - statistics (mean/std)
    - normalization
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.labels: List[torch.Tensor] = []

    def load_labels(self, image_files: List[str]) -> None:
        label_dir = os.path.join(
            self.config.root_folder,
            self.config.dataset_name,
            "labels"
        )

        self.labels = []

        for img_path in image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, base + ".txt")

            if os.path.exists(label_path):
                lbl = torch.from_numpy(
                    np.loadtxt(label_path, dtype=np.float32)
                )
                if lbl.ndim == 1:
                    lbl = lbl.unsqueeze(0)
            else:
                lbl = torch.zeros((0, 5), dtype=torch.float32)

            self.labels.append(lbl)

    def load_or_compute_stats(self, imgs: np.ndarray) -> None:
        stats_path = os.path.join(
            self.config.root_folder,
            self.config.dataset_name,
            self.config.stats_file
        )

        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            mean = imgs.mean(axis=(0, 1, 2))
            std = imgs.std(axis=(0, 1, 2))
            std = np.clip(std, 1e-6, None)

            self.stats = {"mean": mean, "std": std}
            np.save(stats_path, self.stats)

            print(f"{self.config.stats_file} created automatically")

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.stats["mean"],
            dtype=img.dtype
        ).view(-1, 1, 1)

        std = torch.tensor(
            self.stats["std"],
            dtype=img.dtype
        ).view(-1, 1, 1)

        std = torch.clamp(std, min=1e-6)
        return (img - mean) / std

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).float().permute(2, 0, 1)

class NpyDetectionDataset(BaseDetectionDataset):
    def __init__(self, dataset_name: str, root_folder: str, stats_file: str):
        config = DatasetConfig(
            dataset_name=dataset_name,
            root_folder=root_folder,
            stats_file=stats_file
        )
        super().__init__(config)

        img_file = os.path.join(
            root_folder,
            dataset_name,
            "npy_images.npy"
        )

        self.imgs = np.load(img_file)
        self.load_labels([f"img_{i}" for i in range(len(self.imgs))])
        self.load_or_compute_stats(self.imgs)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = self.to_tensor(self.imgs[idx])
        img = self.normalize_img(img)
        return img, self.labels[idx]


class PilDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_name: str,
        root_folder: str,
        img_size: Tuple[int, int],
        stats_file: str
    ):
        config = DatasetConfig(
            dataset_name=dataset_name,
            root_folder=root_folder,
            img_size=img_size,
            stats_file=stats_file
        )
        super().__init__(config)

        image_dir = os.path.join(
            root_folder,
            dataset_name,
            "images"
        )

        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.load_labels(self.image_files)

        imgs = []
        for f in self.image_files:
            img = Image.open(f).convert("RGB").resize(img_size)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)

        imgs = np.stack(imgs, axis=0)
        self.load_or_compute_stats(imgs)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_files[idx]).convert("RGB").resize(
            self.config.img_size
        )
        img = np.array(img, dtype=np.float32) / 255.0
        img = self.to_tensor(img)
        img = self.normalize_img(img)
        return img, self.labels[idx]

class RAMDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_name: str,
        root_folder: str,
        img_size: Tuple[int, int],
        stats_file: str
    ):
        config = DatasetConfig(
            dataset_name=dataset_name,
            root_folder=root_folder,
            img_size=img_size,
            stats_file=stats_file
        )
        super().__init__(config)

        image_dir = os.path.join(
            root_folder,
            dataset_name,
            "images"
        )

        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.load_labels(self.image_files)

        imgs = []
        for f in self.image_files:
            img = Image.open(f).convert("RGB").resize(img_size)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)

        self.imgs = np.stack(imgs, axis=0)
        self.load_or_compute_stats(self.imgs)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = self.to_tensor(self.imgs[idx])
        img = self.normalize_img(img)
        return img, self.labels[idx]


