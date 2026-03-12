import os
import glob
from dataclasses import dataclass
from pathlib import Path
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
        self.image_files: List[str] = []
        self.image_ids: List[str] = []
        self.targets: List[dict] = []
        self.num_classes = 0

    def load_stats(self) -> None:
        stats_path = os.path.join(self.config.root_folder, self.config.dataset_name, self.config.stats_file)

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

    @property
    def dataset_root(self) -> str:
        return os.path.join(self.config.root_folder, self.config.dataset_name)

    def list_image_files(self) -> List[str]:
        image_dir = os.path.join(self.dataset_root, "images")
        return sorted(glob.glob(os.path.join(image_dir, "*.*")))

    def infer_image_ids(self) -> List[str]:
        return [Path(path).stem for path in self.image_files]

    @staticmethod
    def _numeric_sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def get_sorted_label_files(self) -> List[str]:
        label_dir = Path(self.dataset_root) / "labels"
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
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }

    def load_targets(self) -> None:
        label_files = self.get_sorted_label_files()
        self.targets = [self.read_target(path) for path in label_files]
        if not self.targets:
            self.targets = [{"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty((0, 4), dtype=torch.float32)} for _ in range(len(self))]
        self.num_classes = 1 + max((int(t["labels"].max().item()) for t in self.targets if len(t["labels"]) > 0), default=0)

class NpyDetectionDataset(BaseDetectionDataset):
    def __init__(
        self,
        dataset_name: str,
        root_folder: str,
        stats_file: str = "stats.npy",
        load_targets: bool = True,
        return_image_id: bool = False,
    ):
        config = DatasetConfig(
            dataset_name=dataset_name,
            root_folder=root_folder,
            stats_file=stats_file
        )
        super().__init__(config)

        img_file = os.path.join(root_folder, dataset_name, "npy_images.npy")

        self.imgs = np.load(img_file).astype(np.float32)
        self.image_files = self.list_image_files()
        self.image_ids = self.infer_image_ids()
        self.return_image_id = return_image_id
        self.load_targets_flag = load_targets
        self.load_stats()
        if load_targets:
            self.load_targets()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.imgs[idx]).float()
        img = self.normalize_img(img)
        output = [img]
        if self.load_targets_flag:
            output.append(self.targets[idx])
        if self.return_image_id:
            output.append(self.image_ids[idx] if idx < len(self.image_ids) else str(idx))
        if len(output) == 1:
            return output[0]
        return tuple(output)


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
        self.image_files = self.list_image_files()
        self.image_ids = self.infer_image_ids()
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
        self.image_files = self.list_image_files()
        self.image_ids = self.infer_image_ids()

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


def dataset_config_from_path(dataset_path: str, stats_file: str = "stats.npy") -> DatasetConfig:
    dataset_path = Path(dataset_path)
    return DatasetConfig(
        dataset_name=dataset_path.name,
        root_folder=str(dataset_path.parent),
        stats_file=stats_file,
    )


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
