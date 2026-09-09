import os
from pathlib import Path
from dataloading.det_augmentation import DetectionAugmentation, build_ultralytics_labels
from PIL import Image
import numpy as np
import torch
import random
from detection.utils.config_utils import load_class_names


# TODO : AutoAugment automatically searches for the best augmentation policies


class BaseDetectionDataset:
    """
    Load and process detection image. One image traitement.
    sample["image"]: RGB CHW resized image for visualization.
    sample["input"]: RGB CHW resized normalized image.

    It handles dataset metadata and target loading:
    - loads normalization statistics from stats_<img_size>.npy and scales them from [0, 1] to [0, 255] pixel units
    - loads class names and number of classes
    - builds a sorted list of label files
    - parses YOLO-format label files into class labels and bounding boxes
    - stores targets as torch tensors
    - some functions for augmentation 
    """

    def __init__(
        self,
        dataset_root: str,
        img_size: int = 640,
        stats_file: str | None = None,
        data_split: str = "train",
        augment: bool = False,
        augmentation_config=None,
        img_format: str = "jpg",
    ):
        self.dataset_root = Path(dataset_root)
        self.data_split = data_split
        self.img_dir = self.dataset_root / "images" / self.data_split
        self.img_size = int(img_size)
        self.stats_file = stats_file or f"stats_{self.img_size}.npy"
        self.img_format = img_format
        self.augment = bool(augment and self.data_split == "train")
        self.load_stats()
        self.class_names, self.num_classes = load_class_names(dataset_root)
        self.load_targets()
        # Ultralytics Mosaic samples the full dataset when cache is set to "ram"
        # Images remain loaded on demand; this flag only selects its index-sampling path
        self.cache = "ram"
        self.augmentation = (
            DetectionAugmentation(
                dataset=self,
                img_size=self.img_size,
                config=augmentation_config or {},
            )
            if self.augment
            else None
        )

    def load_stats(self) -> None:
        """Load dataset mean/std statistics and scale them from [0, 1] to [0, 255]
        to normalize RGB image tensors whose pixel values remain in [0, 255]."""
        stats_path = self.dataset_root / self.stats_file
        if stats_path.exists():
            # allow_pickle=True allows loading Python objects like dict, .npy file may be a dict
            # .item() transforms to a dict
            stats = np.load(stats_path, allow_pickle=True).item()
            self.stats = {
                "mean": stats["mean"] * np.float32(255.0),
                "std": np.clip(stats["std"], min=1e-6) * np.float32(255.0),
            }
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        # Transform mean and std to torch tensors and reshape to [C, 1, 1] for broadcasting
        self.mean = torch.from_numpy(self.stats["mean"]).to(dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.from_numpy(self.stats["std"]).to(dtype=torch.float32).view(-1, 1, 1)

    def load_targets(self) -> None:
        """Load label files, parse targets, store as torch tensors.
        A dict per sample with keys "labels" and "boxes". A list of dicts for all samples."""
        target_dir = self.dataset_root / "labels" / self.data_split

        def numeric_sort_key(path: Path):
            """Generate a sort key for file paths,
            purely numeric filenames are sorted by their numerical value,
            non-numeric filenames follow numeric ones and are sorted alphabetically."""
            # Get file name without extension
            stem = path.stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        # Samples are sorted by numeric_sort_key
        self.target_files = sorted(target_dir.glob("*.txt"), key=numeric_sort_key)
        if not self.target_files:
            raise FileNotFoundError(f"No label files found under {self.dataset_root / 'labels' / self.data_split}")

        # Samples are sorted by numeric_sort_key
        self.targets = []
        for label_path in self.target_files:
            # Read one label file and parse targets (class labels and bbox coords)
            labels = []
            boxes = []
            if label_path is not None and os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        class_id, x_center, y_center, width, height = line.split()
                        labels.append(int(class_id))
                        boxes.append([float(x_center), float(y_center), float(width), float(height)])
            self.targets.append(
                {
                    "labels": torch.tensor(labels, dtype=torch.int64),
                    "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                }
            )

    def __len__(self) -> int:
        return len(self.target_files)

    def get_image_and_label(self, idx: int):
        """Load one indexed raw image and target in the format expected by Ultralytics transforms,
        which is a BGR HWC image with normalized xywh bbox coords, for augmentation."""
        # Build image path corresponding to a target index
        img_id = self.target_files[idx].stem
        img_path = self.img_dir / f"{img_id}.{self.img_format}"
        # RGB
        with Image.open(img_path).convert("RGB") as image:
            # RGB HWC
            rgb_img = np.asarray(image, dtype=np.uint8)
        # Ultralytics detection transforms expect a contiguous BGR HWC image
        bgr_img = np.ascontiguousarray(rgb_img[:, :, ::-1])
        target = self.targets[idx]
        # Build Ultralytics Instances while keeping the source target tensors unchanged
        return build_ultralytics_labels(
            img=bgr_img,
            classes=target["labels"].numpy(),
            boxes=target["boxes"].numpy(),
            img_path=img_path,
        )

    def __getitem__(self, idx: int):
        """Build one indexed sample dict. Load and optionally augment one indexed sample,
        return image, input, target, image path."""
        # Build image path corresponding to a target index
        img_id = self.target_files[idx].stem
        img_path = self.img_dir / f"{img_id}.{self.img_format}"

        # Augmentation
        if self.augmentation is not None:
            # augmentation updates image and bbox, then Format returns RGB CHW resized image
            # labels = {"img":..., "cls":..., "bboxes":...}
            labels = self.augmentation(self.get_image_and_label(idx))
            img = labels["img"]
            # Convert Ultralytics output back to the target dict used by the training loop
            target = {
                "labels": labels["cls"].reshape(-1).to(dtype=torch.int64),
                "boxes": labels["bboxes"].reshape(-1, 4).to(dtype=torch.float32),
            }
        # Without augmentation
        else:
            # RGB
            with Image.open(img_path).convert("RGB") as image:
                # RGB resized
                image = image.resize((self.img_size, self.img_size))
                # copy guarantees writable contiguous memory, HWC to CHW, RGB CHW resized image
                img = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
            # Clone labels and boxes to avoid modifying cached source targets
            # A dict per sample with keys "labels" and "boxes"
            target = {key: value.clone() for key, value in self.targets[idx].items()}

        img = img.float()
        # Mean/std normalize image tensor
        norm_img = (img - self.mean) / self.std
        return {
            # RGB CHW resized image for visualization
            "image": img,
            # RGB CHW resized normalized image
            "input": norm_img,
            # target: {"labels":..., "boxes":...}
            "target": target,
            "img_path": str(img_path),
        }

    def close_mosaic(self):
        """Disable multi-image augmentation while keeping single-image transforms active."""
        if self.augmentation is not None:
            self.augmentation.close_mosaic()

    def disable_augmentation(self):
        """Disable all augmentation, mainly before sampling images after training."""
        self.augment = False
        self.augmentation = None


class JpgDetectionDataset(BaseDetectionDataset):
    """JPG dataset using the base class's default image format and loading pipeline."""


def detection_collate_fn(batch):
    """Stack single sample into batch."""
    # Nb targets per image may vary, later use target_counts to re-split concatenated labels and boxes.
    target_counts = [len(item["target"]["labels"]) for item in batch]
    return {
        # [B, 3, H, W]
        "inputs": torch.stack([item["input"] for item in batch], dim=0),
        # Concatenation for better GPU transfer performance
        "targets": {
            # Concatenate the class labels of all images in the batch into a 1D tensor, [Nb targets of the batch]
            "labels": torch.cat([item["target"]["labels"] for item in batch], dim=0),
            # Concatenate the bbox of all images in the batch into a 2D tensor, [Nb targets of the batch, 4]
            "boxes": torch.cat([item["target"]["boxes"] for item in batch], dim=0),
            # List of nb targets per image, [B]
            "counts": target_counts,
        },
        # [paths]
        "img_paths": [item["img_path"] for item in batch],
    }


def parse_batch(batch, device=None):
    """Extract model inputs and targets from a dataloader batch
    and convert targets to a per-image list of dictionaries on the specified device."""
    inputs = batch["inputs"]
    targets = batch["targets"]
    # detection_collate_fn() returns targets as a dict with keys "labels", "boxes", "counts"
    labels = targets["labels"]
    boxes = targets["boxes"]
    if device is not None:
        # Move labels (all in one tensor) and boxes (all in one tensor) to device
        labels = labels.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
    # Split labels and boxes into per-image lists based on counts
    labels_per_image = labels.split(targets["counts"])
    boxes_per_image = boxes.split(targets["counts"])
    # Reconstruct targets as a list of dicts, one per image, with keys "labels" and "boxes"
    targets = [{"labels": image_labels, "boxes": image_boxes} for image_labels, image_boxes in zip(labels_per_image, boxes_per_image)]
    # inputs = torch.Tensor( shape=[B, 3, H, W], dtype=torch.float32, )
    # targets = [ { "labels": torch.Tensor( shape=[N_i], dtype=torch.int64, ),
    #                "boxes": torch.Tensor( shape=[N_i, 4], dtype=torch.float32, ), }, ... ]
    return inputs, targets


def sample_dataset(dataset, num_samples, seed, device):
    """Randomly sample from dataset, return a batch of model input (on device),
    visualization tensors, image paths.
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}."""
    # Use an independent seeded generator so sampling is reproducible without changing global random state
    rng = random.Random(seed)
    # Never request more samples than the dataset contains
    sample_size = min(num_samples, len(dataset))
    # Randomly select unique indices, then sort them to preserve dataset order in the output batch
    sampled_indices = sorted(rng.sample(range(len(dataset)), sample_size))
    # Load the selected samples from the dataset
    samples = [dataset[index] for index in sampled_indices]
    inputs = torch.stack([sample["input"] for sample in samples], dim=0).to(device)
    imgs = torch.stack([sample["image"] for sample in samples], dim=0)
    img_paths = [sample["img_path"] for sample in samples]
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}
    return samples


"""
Output information: 
JpgDetectionDataset.__getitem__ -> {"image", "input", "target": {"labels", "boxes"}, "img_path"}

detection_collate_fn -> {"inputs", "targets": {"labels", "boxes", "counts"}, "img_paths"}
"""
