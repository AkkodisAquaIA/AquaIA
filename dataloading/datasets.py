import os
from pathlib import Path
from typing import List
from dataloading.det_augmentation import DetectionAugmentation, build_ultralytics_labels
from detection.utils.import_utils import (
    fn,
    pipeline_def,
    types,
    DALIRaggedIterator,
    LastBatchPolicy,
)
from PIL import Image

import numpy as np
import torch
import random
from detection.utils.config_utils import load_class_names


# TODO : AutoAugment automatically searches for the best augmentation policies


# Transform a regular Python function into a definition function for DALI data processing pipeline
@pipeline_def
def create_detection_pipeline(dataset_src, stats, img_size=640, device="gpu", cpu_preprocessed=False):
    """DALI data preprocessing pipeline, used to read, optionally decode and resize, and normalize JPG image
    then output for training. One image treatment pipeline.
    Without CPU augmentation: DALI decode, resize, mean/std normalize, DALI CHW tensor (inputs).
    With CPU augmentation: image already decoded/resized, DALI mean/std normalize, DALI CHW tensor (inputs).
    Returns inputs, labels and boxes on GPU, idx on CPU.
    """
    # external_source() gets data from outside of DALI pipeline
    # here dataset_src = self.dataset.__call__ = JpgDALIDataset.__call__
    # JpgDALIDataset.__call__() returns a tuple of (img, labels, boxes, idx)
    # img is encoded JPG bytes without CPU augmentation, otherwise a decoded/augmented HWC image
    # labels = class labels, boxes = bbox coords, idx = index of image in dataset
    image, labels, boxes, idx = fn.external_source(
        source=dataset_src,
        # external source returns 4 outputs
        num_outputs=4,
        # external source returns one sample at a time not a batch
        batch=False,
        # external source can be called in parallel by multiple threads
        parallel=True,
        dtype=[types.UINT8, types.INT64, types.FLOAT, types.INT64],
        # image has 3 dim when decoded, otherwise encoded JPG bytes have 1 dim
        ndim=[3 if cpu_preprocessed else 1, 1, 2, 1],
    )
    if cpu_preprocessed:
        # CPU augmentation already decodes, resizes and returns an RGB HWC image
        if device == "gpu":
            image = image.gpu()
    else:
        # mixed = read/prepare on CPU, decode on GPU
        decoding_device = "mixed" if device == "gpu" else device
        # TODO : add cache/padding to the decoding part to avoid memory re-allocation
        # Decode image to RGB, HWC on GPU
        image = fn.decoders.image(image, device=decoding_device, output_type=types.RGB)
        image = fn.resize(
            image,
            resize_x=img_size,
            resize_y=img_size,
            device=device,
        )
    # Transform to float, mean/std normalize, from HWC to CHW, on GPU
    inputs = fn.crop_mirror_normalize(
        image,
        device=device,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=stats["mean"],
        std=stats["std"],
    )
    # Move labels and boxes to GPU
    if device == "gpu":
        labels = labels.gpu()
        boxes = boxes.gpu()
    # inputs is DALI tensor
    return inputs, labels, boxes, idx


class BaseDetectionDataset:
    """
    Base class shared by non-DALI and DALI dataset implementations.
    This class handles dataset metadata and target loading:
    - loads normalization statistics from stats.npy and scales them from [0, 1] to [0, 255] pixel units
    - loads class names and number of classes
    - builds a sorted list of label files
    - parses YOLO-format label files into class labels and bounding boxes
    - stores targets as torch tensors rather than DALI tensors
    And some useful functions for image and target loading, augmentation, and normalization.
    """

    def __init__(
        self,
        dataset_root: str,
        data_split: str = "train",
        stats_file: str = "stats.npy",
        device: str = "cpu",
        img_size: int = 640,
        img_format: str = "jpg",
        augment: bool = False,
        augmentation_config=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.data_split = data_split
        self.stats_file = stats_file
        self.img_dir = self.dataset_root / "images" / self.data_split
        self.img_size = int(img_size)
        self.img_format = img_format
        self.augment = bool(augment and self.data_split == "train")
        self.load_stats()
        self.class_names, self.num_classes = load_class_names(dataset_root)
        self.device = device
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

    def __len__(self):
        # self.target_files apprears after load_targets() is called
        return len(self.target_files)

    def load_stats(self) -> None:
        """Load dataset mean/std statistics and scale them from [0, 1] to [0, 255]
        for normalization of decoded JPG pixels."""
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

    @staticmethod
    # Static method don't need class param
    def _numeric_sort_key(path: Path):
        """Generate a sort key for file paths,
        purely numeric filenames are sorted by their numerical value,
        non-numeric filenames follow numeric ones and are sorted alphabetically."""
        # Get file name without extension
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def get_sorted_target_files(self) -> List[str]:
        target_dir = self.dataset_root / "labels" / self.data_split
        target_files = [path for path in target_dir.glob("*.txt")]
        return sorted(target_files, key=self._numeric_sort_key)

    def parse_target_line(self, line: str):
        class_id, x_center, y_center, width, height = line.split()
        return int(class_id), [float(x_center), float(y_center), float(width), float(height)]

    def read_target(self, label_path: str):
        """Read one label file and parse targets (class labels and bbox coords)."""
        labels = []
        boxes = []
        if label_path is not None and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id, bbox = self.parse_target_line(line)
                    labels.append(class_id)
                    boxes.append(bbox)
        # TODO : clean up, dict struct is not longer necessary
        return {
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }

    def load_targets(self) -> None:
        """Load label files, parse targets, store as torch tensors.
        A dict per sample with keys "labels" and "boxes". A list of dicts for all samples."""
        self.target_files = self.get_sorted_target_files()
        if not self.target_files:
            raise FileNotFoundError(f"No label files found under {self.dataset_root / 'labels' / self.data_split}")
        self.targets = [self.read_target(path) for path in self.target_files]

    def get_image_path(self, idx: int) -> Path:
        """Build image path corresponding to a target index."""
        img_id = self.target_files[idx].stem
        return self.img_dir / f"{img_id}.{self.img_format}"

    def get_image_and_label(self, idx: int):
        """Load one raw image and target in the format expected by Ultralytics transforms.
        BGR HWC image, normalized xywh bbox coords."""
        img_path = self.get_image_path(idx)
        with Image.open(img_path).convert("RGB") as image:
            rgb_img = np.asarray(image, dtype=np.uint8)
        # Ultralytics detection transforms expect a contiguous BGR HWC image
        bgr_img = np.ascontiguousarray(rgb_img[:, :, ::-1])
        target = self.targets[idx]
        # Build Ultralytics Instances while keeping the source target tensors unchanged
        return build_ultralytics_labels(
            img=bgr_img,
            labels=target["labels"].numpy(),
            boxes=target["boxes"].numpy(),
            img_path=img_path,
        )

    def copy_target(self, idx: int) -> dict:
        """Clone and return target (label + bbox) at an index to avoid modifying source target.
        A dict per sample with keys "labels" and "boxes"."""
        return {key: value.clone() for key, value in self.targets[idx].items()}

    def load_sample(self, idx: int):
        """Load and optionally augment one sample. Returns RGB CHW image, target dict and image path."""
        img_path = self.get_image_path(idx)
        if self.augmentation is not None:
            # augmentation updates image and bbox, then Format returns RGB CHW image
            # labels = {"img":..., "cls":..., "bboxes":...}
            labels = self.augmentation(self.get_image_and_label(idx))
            img = labels["img"]
            # Convert Ultralytics output back to the target dict used by the training loop
            target = {
                "labels": labels["cls"].reshape(-1).to(dtype=torch.int64),
                "boxes": labels["bboxes"].reshape(-1, 4).to(dtype=torch.float32),
            }
        else:
            # Without augmentation
            with Image.open(img_path).convert("RGB") as image:
                image = image.resize((self.img_size, self.img_size))
                # copy guarantees writable contiguous memory, HWC to CHW
                img = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
            # Clone labels and boxes to avoid modifying cached source targets
            target = self.copy_target(idx)
        # target: {"label":..., "boxes":...}
        return img, target, img_path

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        """Mean/std normalize image tensor. Only for non-DALI situations."""
        # Transform mean and std to torch tensors and reshape to [C, 1, 1] for broadcasting
        mean = torch.from_numpy(self.stats["mean"]).to(dtype=img.dtype).view(-1, 1, 1)
        std = torch.from_numpy(self.stats["std"]).to(dtype=img.dtype).view(-1, 1, 1)
        return (img - mean) / std

    def build_sample(self, idx: int):
        """Build the sample dictionary shared by DALI __getitem__ and non-DALI datasets."""
        img, target, img_path = self.load_sample(idx)
        # sample["image"] is float CHW for visualization; sample["input"] is mean/std normalized
        img = img.float()
        norm_img = self.normalize_img(img)
        return {
            "image": img,
            "input": norm_img,
            "target": target,
            "target_idx": idx,
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


class JpgDALIDataset(BaseDetectionDataset):
    """Mainly used for DALI's external_source, which means the following __call__
    will be repeatedly called by DALI to read one image.
    Train (__call__) without augmentation: JPG bytes; decode, resize and normalize will be done in DALI pipeline.
    Train (__call__) with augmentation: PIL decode and CPU augmentation first; normalization will be done in DALI pipeline.
    Direct access (__getitem__): PIL decode, resize, CHW PyTorch tensor (sample["image"]),
    then mean/std normalize (sample["input"])."""
    # TODO : only JPEG, need to think about TIFF handling

    def __init__(
        self,
        dataset_root: str,
        data_split: str = "train",
        batch_size: int = 16,
        img_size: int = 640,
        img_format: str = "jpg",
        stats_file: str = "stats.npy",
        device: str = "cpu",
        augment: bool = False,
        augmentation_config=None,
    ):
        super().__init__(
            dataset_root=dataset_root,
            data_split=data_split,
            stats_file=stats_file,
            device=device,
            img_size=img_size,
            img_format=img_format,
            augment=augment,
            augmentation_config=augmentation_config,
        )
        # After called super().__init__(), we have self.img_dir, self.stats, self.class_names,
        # self.num_classes, self.target_files, self.targets, self.img_size and optional self.augmentation
        self.batch_size = batch_size
        if img_format not in ["jpg", "jpeg"]:
            raise NotImplementedError(f"Unsupported image format: {img_format}. Only jpg is currently supported.")

        self.n = len(self.target_files)
        # Create indices for all samples
        self.indices = list(range(self.n))
        # Compute the number of full batches
        self.full_iterations = self.n // batch_size
        # Shuffling related stuff
        self.perm = self.indices  # permutation of indices
        # last_seen_epoch is used to track the epoch index for shuffling
        # All samples in the same epoch have same self.perm
        self.last_seen_epoch = (
            # so that we don't have to recompute the `self.perm` for every sample
            None
        )

    def __call__(self, sample_info):
        """Called by DALI's external_source to get a sample (image, labels, boxes, index).
        Image is encoded JPG bytes without augmentation, otherwise decoded/augmented HWC uint8."""
        # Get sample's position in actual epoch from sample_info given by DALI
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.data_split == "train":
            # Shuffling at the start of each epoch
            if self.last_seen_epoch != sample_info.epoch_idx:
                self.last_seen_epoch = sample_info.epoch_idx
                # Create a random number generator
                self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx)
                # Shuffle the indices for this epoch
                self.perm = self.perm.permutation(self.indices)
        # Find the true dataset index based on the current sample's position in the epoch
        idx = self.perm[sample_idx]
        if self.augment:
            # Use a reproducible but different augmentation seed for every sample and epoch
            augmentation_seed = 42 + sample_info.epoch_idx * self.n + sample_idx
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed % (2**32))
            # CPU augmentation returns RGB CHW uint8; DALI external_source expects HWC numpy
            img, target, _ = self.load_sample(idx)
            img = np.ascontiguousarray(img.permute(1, 2, 0).numpy())
        else:
            img_path = self.get_image_path(idx)
            # Preserve GPU/mixed JPEG decoding when CPU augmentation is disabled.
            # Read original binary image bytes from disk and convert to numpy array of uint8
            img = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
            target = self.targets[idx]
        return (
            img,
            target["labels"].numpy(),
            target["boxes"].numpy(),
            np.array([idx], dtype=np.int64),
        )

    def __getitem__(self, key):
        """Load one sample outside the DALI iterator for visualization / testing."""
        # Slow but useful for sampling a few images for visualization / testing
        # Find the true dataset index
        idx = self.indices[key]
        # Shared CPU helper returns image, normalized input, target, index and image path
        return self.build_sample(idx)


class JpgDetectionDataset(BaseDetectionDataset):
    """This class is used for non-DALI situations, where images are loaded and processed using PIL and NumPy.
    One image. JPG bytes, PIL decode, optional augmentation, resize, CHW pytorch tensor (sample["image"]),
    mean/std normalize (sample["input"]). No CPU/CUDA/GPU transfer in this class."""
    def __init__(
        self,
        dataset_root: str,
        img_size: int = 640,
        stats_file: str = "stats.npy",
        device: str = "cpu",
        data_split: str = "train",
        augment: bool = False,
        augmentation_config=None,
    ):
        super().__init__(
            dataset_root=dataset_root,
            stats_file=stats_file,
            data_split=data_split,
            device=device,
            img_size=img_size,
            augment=augment,
            augmentation_config=augmentation_config,
        )

    def __len__(self) -> int:
        return len(self.target_files)

    def __getitem__(self, idx: int):
        """Load one sample and return image, input, target, target index and image path."""
        return self.build_sample(idx)


class DALIDetectionDataLoader:
    """Wrap JpgDALIDataset into a DALI dataloader for training loop as `for batch in loader`"""
    def __init__(
        self,
        dataset,
        device="gpu",  # can be dropped and inferred from dataset, but keeping it explicit for now
        # nb threads used for DALI pipeline execution
        num_threads=3,
        # nb Python workers used by DALI when calling a Python external_source
        py_num_workers=3,
        py_start_method="spawn",
    ):
        self.dataset = dataset
        self.device = device
        self.pipeline = create_detection_pipeline(
            dataset_src=self.dataset.__call__,
            stats=self.dataset.stats,
            device=self.device,
            img_size=self.dataset.img_size,
            # Tell DALI whether external_source returns encoded bytes or an augmented HWC image
            cpu_preprocessed=self.dataset.augment,
            batch_size=self.dataset.batch_size,
            num_threads=num_threads,
            py_num_workers=py_num_workers,
            py_start_method=py_start_method,
        )
        self.pipeline.build()
        self.loader = DALIRaggedIterator(
            pipelines=[self.pipeline],
            # Define outputs, batch["inputs"] etc
            output_map=["inputs", "labels", "boxes", "targets_idx"],
            # Declare output types for each output, DALI tensor or list of DALI tensors
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                DALIRaggedIterator.SPARSE_LIST_TAG,
                DALIRaggedIterator.SPARSE_LIST_TAG,
                DALIRaggedIterator.DENSE_TAG,
            ],
            # Nb samples per epoch
            size=self.dataset.full_iterations * self.dataset.batch_size,
            # Drop last batch if smaller than batch_size
            last_batch_policy=LastBatchPolicy.DROP,
            # Reset iterator
            auto_reset=True,
        )

    def __len__(self):
        # Return full batch nb
        return self.dataset.full_iterations

    def __iter__(self):
        # Define iteration behavior
        for batch in self.loader:
            if isinstance(batch, list):
                batch = batch[0]
            # .pop() removes the key from the dict and returns its value
            labels_batch = batch.pop("labels")
            boxes_batch = batch.pop("boxes")
            # zip() pairs each labels and boxes from the batch together
            batch["targets"] = [{"labels": labels, "boxes": boxes} for labels, boxes in zip(labels_batch, boxes_batch)]
            # Yield means that this function is a generator, it will return a batch and pause until the next call to __next__()
            # { "inputs": ..., "targets_idx": ...,
            # "targets": [ {"labels": ..., "boxes": ...}, {"labels": ..., "boxes": ...}, ... ], }
            yield batch


def parse_batch(batch, device=None):
    """Extract model inputs and targets from a dataloader batch
    and convert targets to a per-image list of dictionaries on the specified device.
    If non DALI, moves labels and boxes to the specified device."""
    if isinstance(batch, list):
        batch = batch[0]
    inputs = batch["inputs"]
    targets = batch["targets"]

    # Non DALI, detection_collate_fn() returns targets as a dict with keys "labels", "boxes", "counts"
    if isinstance(targets, dict):
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
    # DALI
    elif device is not None:
        # Move each value in target dict to device if it's a tensor, otherwise keep it as is
        targets = [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]
    # inputs = torch.Tensor( shape=[B, 3, H, W], dtype=torch.float32, )
    # targets = [ { "labels": torch.Tensor( shape=[N_i], dtype=torch.int64, ),
    #                "boxes": torch.Tensor( shape=[N_i, 4], dtype=torch.float32, ), }, ... ]
    return inputs, targets


def sample_indices(dataset_size, num_samples, seed):
    """Randomly sample some sample indices from the dataset and return them sorted."""
    rng = random.Random(seed)
    sample_size = min(num_samples, dataset_size)
    return sorted(rng.sample(range(dataset_size), sample_size))


def detection_collate_fn(batch):
    """Stack single sample into batch. Only for non-DALI situations.
    No CPU/CUDA/GPU transfer."""
    # Nb targets per image may vary, later use target_counts to re-split concatenated labels and boxes.
    target_counts = [len(item["target"]["labels"]) for item in batch]
    collated_batch = {
        # [B, 3, H, W]
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "inputs": torch.stack([item["input"] for item in batch], dim=0),
        "targets": {
            # Concatenate the class labels of all images in the batch into a 1D tensor, [Nb targets of the batch]
            "labels": torch.cat([item["target"]["labels"] for item in batch], dim=0),
            # Concatenate the bbox of all images in the batch into a 2D tensor, [Nb targets of the batch, 4]
            "boxes": torch.cat([item["target"]["boxes"] for item in batch], dim=0),
            # List of nb targets per image, [B]
            "counts": target_counts,
        },
        # [indices]
        "targets_idx": [item["target_idx"] for item in batch],
        # [paths]
        "img_paths": [item["img_path"] for item in batch],
    }
    return collated_batch


def sample_dataset(dataset, num_samples, seed, device):
    """Randomly sample from dataset, return model input batch, visualization batch, image paths.
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}."""
    sampled_indices = sample_indices(len(dataset), num_samples, seed)
    # Get samples
    samples = [dataset[index] for index in sampled_indices]
    inputs = torch.stack([sample["input"] for sample in samples], dim=0).to(device)
    imgs = torch.stack([sample["image"] for sample in samples], dim=0)
    img_paths = [sample["img_path"] for sample in samples]
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}
    return samples


"""
==================
Output information
==================

DALI training path without CPU augmentation:
JpgDALIDataset.__call__ -> encoded_img, labels, boxes, idx
create_detection_pipeline -> DALI decode, resize and normalize -> inputs, labels, boxes, idx
DALIDetectionDataLoader -> inputs, targets (labels, boxes), targets_idx in batch

DALI training path with CPU augmentation:
JpgDALIDataset.__call__ -> decoded/augmented HWC image, labels, boxes, idx
create_detection_pipeline -> DALI normalize and HWC to CHW -> inputs, labels, boxes, idx
DALIDetectionDataLoader -> inputs, targets (labels, boxes), targets_idx in batch

DALI __getitem__ path:
JpgDALIDataset.__getitem__ -> image, input, target (labels, boxes), target_idx, img_path

non-DALI path:
JpgDetectionDataset.__getitem__ -> image, input, target (labels, boxes), target_idx, img_path
detection_collate_fn -> images, inputs, targets (labels, boxes, counts), targets_idx, img_paths in batch
"""
