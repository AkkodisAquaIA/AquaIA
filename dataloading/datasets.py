import os
from pathlib import Path
from typing import List
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIRaggedIterator, LastBatchPolicy
import nvidia.dali.experimental.dynamic as ndd

import numpy as np
import torch
import random
from detection.utils.config_utils import load_class_names

# TODO : AutoAugment: Learning Augmentation Strategies from Data


@pipeline_def
def create_detection_pipeline(dataset_src, stats, img_size=640, device="gpu"):
	encoded, labels, bboxes = fn.external_source(
		source=dataset_src,
		num_outputs=3,
		batch=False,
		parallel=True,
		dtype=[types.UINT8, types.INT64, types.FLOAT],
	)
	decoding_device = "mixed" if device == "gpu" else device
	# TODO : add cache/padding to the decoding part to avoid memory re-allocation
	images = fn.decoders.image(encoded, device=decoding_device, output_type=types.RGB)
	images = fn.resize(
		images,
		resize_x=img_size,
		resize_y=img_size,
		device=device,
	)
	inputs = fn.crop_mirror_normalize(
		images,
		device=device,
		dtype=types.FLOAT,
		output_layout="CHW",
		mean=stats["mean"],
		std=stats["std"],
	)
	return inputs, labels.gpu(), bboxes.gpu()


class BaseDetectionDataset:
	"""
	Base class shared by NPY / PIL / RAM datasets.

	Handles:
	- label loading
	- statistics (mean/std)
	- normalization
	"""

	def __init__(
		self,
		dataset_root: str,
		data_split: str = "train",
		stats_file: str = "stats.npy",
	):
		self.dataset_root = Path(dataset_root)
		self.stats_file = stats_file
		self.data_split = data_split
		self.load_stats()
		self.class_names, self.num_classes = load_class_names(dataset_root)
		self.load_targets()
		# if not (self.dataset_root / self.data_split).exists():
		#     raise FileNotFoundError(f"Data split directory not found: {self.dataset_root / self.data_split}")

	def __len__(self):
		return len(self.target_files)

	def load_stats(self) -> None:
		# /!\ Expect stats to be computed in normalized pixels in [0, 1] range
		stats_path = self.dataset_root / self.stats_file

		if stats_path.exists():
			stats = np.load(stats_path, allow_pickle=True).item()
			self.stats = {
				"mean": stats["mean"] * np.float32(255.0),
				"std": np.clip(
					stats["std"],
					min=1e-6,
				)
				* np.float32(255.0),
			}
		else:
			# raise FileNotFoundError(f"Stats file not found: {stats_path}")
			self.stats = {
				"mean": np.zeros(3, dtype=np.float32) * np.float32(255.0),
				"std": np.clip(
					np.ones(3, dtype=np.float32),
					min=1e-6,
				)
				* np.float32(255.0),
			}

	def to_tensor(self, img: np.ndarray) -> torch.Tensor:
		return torch.from_numpy(img).float().permute(2, 0, 1)

	@staticmethod
	def _numeric_sort_key(path: Path):
		stem = path.stem
		return (0, int(stem)) if stem.isdigit() else (1, stem)

	def get_sorted_target_files(self) -> List[str]:
		target_dir = self.dataset_root / "labels" / self.data_split
		target_files = [path for path in target_dir.glob("*.txt")]
		return sorted(target_files, key=lambda path: self._numeric_sort_key(path))

	def parse_target_line(self, line: str):
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
					class_id, bbox = self.parse_target_line(line)
					labels.append(class_id)
					boxes.append(bbox)
		# TODO : clean up, dict struct is not longer necessary
		return {
			"labels": np.array(labels, dtype=np.int64),
			"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
		}

	def load_targets(self) -> None:
		self.target_files = self.get_sorted_target_files()
		if not self.target_files:
			raise FileNotFoundError(f"No label files found under {self.dataset_root / 'labels'}")
		self.targets = [self.read_target(path) for path in self.target_files]

	def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
		mean = torch.from_numpy(self.stats["mean"]).to(dtype=img.dtype).view(-1, 1, 1)
		std = torch.from_numpy(self.stats["std"]).to(dtype=img.dtype).view(-1, 1, 1)
		return (img - mean) / std


class YOLOFormatDataset(BaseDetectionDataset):
	# TODO : only JPEG, need to think about TIFF handling

	def __init__(
		self,
		dataset_root: str,
		data_split: str = "train",
		batch_size: int = 16,
		img_size: int = 640,
		img_format: str = "jpg",
		stats_file: str = "stats.npy",
	):
		super().__init__(
			dataset_root=dataset_root,
			data_split=data_split,
			stats_file=stats_file,
		)
		self.img_size = img_size
		self.batch_size = batch_size
		self.img_format = img_format
		if img_format not in ["jpg", "jpeg"]:
			raise NotImplementedError(f"Unsupported image format: {img_format}. Only jpg is currently supported.")

		self.img_dir = self.dataset_root / "images" / self.data_split
		self.n = len(self.target_files)
		self.indices = list(range(self.n))
		self.full_iterations = self.n // batch_size
		# Shuffling related stuff
		self.perm = self.indices  # permutation of indices
		self.last_seen_epoch = (
			# so that we don't have to recompute the `self.perm` for every sample
			None
		)

	@staticmethod
	def _dali_tensor_to_torch(tensor):
		return torch.from_dlpack(tensor.evaluate().data)

	def __call__(self, sample_info):
		# print(sample_info.iteration, sample_info.idx_in_epoch, sample_info.epoch_idx)
		sample_idx = sample_info.idx_in_epoch
		if sample_info.iteration >= self.full_iterations:
			# Indicate end of the epoch
			raise StopIteration
		if self.data_split == "train":
			# Shuffling at the start of each epoch
			if self.last_seen_epoch != sample_info.epoch_idx:
				self.last_seen_epoch = sample_info.epoch_idx
				self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx)
				self.perm = self.perm.permutation(self.indices)
		idx = self.perm[sample_idx]
		label = self.targets[idx]["labels"]
		bboxes = self.targets[idx]["boxes"]
		img_id = self.target_files[idx].stem
		img_path = self.img_dir / f"{img_id}.{self.img_format}"
		# Encoded image bytes. DALI will decode this.
		encoded_img = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
		return encoded_img, label, bboxes

	def __getitem__(self, key):
		# Slow but useful for sampling a few images for visualization / testing
		# Mirrors the DALI pipeline path using DALI dynamic operators.
		idx = self.indices[key]
		label = self.targets[idx]["labels"]
		bboxes = self.targets[idx]["boxes"]
		img_id = self.target_files[idx].stem
		img_path = self.img_dir / f"{img_id}.{self.img_format}"

		encoded_img = np.frombuffer(img_path.read_bytes(), dtype=np.uint8).copy()
		device = "gpu" if torch.cuda.is_available() else "cpu"
		decoding_device = "mixed" if device == "gpu" else device
		mean = self.stats["mean"].astype(np.float32).tolist()
		std = self.stats["std"].astype(np.float32).tolist()

		img = ndd.decoders.image(encoded_img, device=decoding_device, output_type=types.RGB)
		img = ndd.resize(
			img,
			resize_x=float(self.img_size),
			resize_y=float(self.img_size),
			device=device,
		)
		norm_img = ndd.crop_mirror_normalize(
			img,
			device=device,
			dtype=types.FLOAT,
			output_layout="CHW",
			mean=mean,
			std=std,
		)
		img = self._dali_tensor_to_torch(img).cpu().permute(2, 0, 1)
		norm_img = self._dali_tensor_to_torch(norm_img)
		torch_device = "cuda" if device == "gpu" else "cpu"
		sample = {
			"image": img,
			"input": norm_img,
			"label": torch.from_numpy(label).to(torch_device),
			"bboxes": torch.from_numpy(bboxes).to(torch_device),
			"img_path": str(img_path),
		}
		return sample


class DALIDetectionDataLoader:
	def __init__(
		self,
		dataset,
		device="gpu",
		num_threads=3,
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
			batch_size=self.dataset.batch_size,
			num_threads=num_threads,
			py_num_workers=py_num_workers,
			py_start_method=py_start_method,
		)
		self.pipeline.build()
		self.loader = DALIRaggedIterator(
			pipelines=[self.pipeline],
			output_map=["inputs", "labels", "bboxes"],
			output_types=[
				DALIRaggedIterator.DENSE_TAG,
				DALIRaggedIterator.SPARSE_LIST_TAG,
				DALIRaggedIterator.SPARSE_LIST_TAG,
			],
			size=self.dataset.full_iterations * self.dataset.batch_size,
			last_batch_policy=LastBatchPolicy.DROP,
			auto_reset=True,
		)

	def __len__(self):
		return self.dataset.full_iterations

	def __iter__(self):
		return iter(self.loader)


def parse_batch(batch):
	if isinstance(batch, list):
		batch = batch[0]
	inputs = batch["inputs"]
	# TODO : ugly but currently required. Need to modify downstream code to avoid this conversion
	targets = [{"labels": labels, "boxes": boxes} for labels, boxes in zip(batch["labels"], batch["bboxes"])]
	return inputs, targets, batch.get("img_paths", None)


def sample_indices(dataset_size, num_samples, seed):
	rng = random.Random(seed)
	sample_size = min(num_samples, dataset_size)
	return sorted(rng.sample(range(dataset_size), sample_size))


def collate_yolo(batch):
	collated = {
		"images": torch.stack([item["image"] for item in batch], dim=0),
		"inputs": torch.stack([item["input"] for item in batch], dim=0),
		"labels": [item["label"] for item in batch],
		"bboxes": [item["bboxes"] for item in batch],
		"img_paths": [item["img_path"] for item in batch],
	}
	return collated


def sample_dataset(dataset, num_samples, seed, device):
	sampled_indices = sample_indices(len(dataset), num_samples, seed)
	samples = [dataset[index] for index in sampled_indices]
	inputs = torch.stack([sample["input"] for sample in samples], dim=0).to(device)
	imgs = torch.stack([sample["image"] for sample in samples], dim=0)
	img_paths = [sample["img_path"] for sample in samples]
	samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}
	return samples
