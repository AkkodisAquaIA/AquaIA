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
        image_dir = os.path.join(
            config.root_folder,
            config.dataset_name,
            "images"
        )
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.labels  =  []

    def load_labels(self) -> None:
        # Load labels from label folder
        # Should return a list of dict("labels", "boxes")
        pass

    def load_stats(self) -> None:
        stats_path = os.path.join(
            self.config.root_folder,
            self.config.dataset_name,
            self.config.stats_file
        )

        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

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
    def __init__(self, dataset_name: str, root_folder: str, stats_file: str = "stats.npy"):
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
        self.load_stats()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.imgs[idx])
        img = self.normalize_img(img)
        return img


class PilDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_name: str,
        root_folder: str,
        img_size: Tuple[int, int],
        stats_file: str = "stats.npy"
    ):
        config = DatasetConfig(
            dataset_name=dataset_name,
            root_folder=root_folder,
            img_size=img_size,
            stats_file=stats_file
        )
        super().__init__(config)
        self.load_stats()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_files[idx]).convert("RGB").resize(
            self.config.img_size
        )
        img = np.array(img, dtype=np.float32) / 255.0
        img = self.to_tensor(img)
        img = self.normalize_img(img)
        return img

class RAMDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_name: str,
        root_folder: str,
        img_size: Tuple[int, int],
        stats_file: str = "stats.npy"
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

        imgs = []
        for f in self.image_files:
            img = Image.open(f).convert("RGB").resize(img_size)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)

        self.imgs = np.stack(imgs, axis=0)
        self.load_stats()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = self.to_tensor(self.imgs[idx])
        img = self.normalize_img(img)
        return img


