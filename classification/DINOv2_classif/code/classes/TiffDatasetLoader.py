import numpy as np
import torch
import torchvision
from torchvision.datasets import VisionDataset
from PIL import Image
from pathlib import Path
import torch.nn.functional as nnF
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class TiffDatasetLoader(VisionDataset):
	def __init__(
		self,
		indices,
		img_data,
		img_labels,
		data_stats=None,
		img_res=224,
		inference_mode=False,
		p=0.5,
		weights: bool = True,
		augmentation_params=False,
		rare_classes=None,  # 👈 now a dict: {class_name: p_scale}
	):
		super().__init__(root=None, transforms=None)
		self.indices = indices
		self.img_data = img_data
		self.img_labels = img_labels
		self.data_stats = data_stats or {}
		self.img_res = img_res
		self.inference_mode = inference_mode
		self.p = p
		self.weights = weights
		self.class_p_scale = rare_classes or {}  # expects: {'Hydropsyche siltalai': 1.8, ...}
		self.class_names = list(img_data.keys())
		self.num_classes = len(self.class_names)
		self.class_weights = self._compute_class_weights()
		self.augmentation_params = augmentation_params
		self.albumentation_transform = None  # Lazy init per sample

	def __len__(self):
		return len(self.indices)

	def _compute_class_weights(self) -> torch.Tensor:
		all_labels = []
		for cls in self.class_names:
			all_labels.extend(self.img_labels[cls])
		y = np.array(all_labels, dtype=int)
		classes = np.arange(self.num_classes)
		weights = compute_class_weight("balanced", classes=classes, y=y)
		return torch.from_numpy(weights.astype(np.float32))

	def _get_albumentations_transform(self, mean, std, p_scale):
		"""
		Builds an augmentation pipeline with intensity scaled by `p_scale`.
		For rare classes (p_scale > 1.0), additional stronger transforms are included.
		"""
		base_p = 0.7
		p = min(1.0, max(0.0, round(base_p * p_scale, 4)))

		transforms = [
			# Basic geometry
			A.HorizontalFlip(p=0.5 * p),
			A.VerticalFlip(p=0.5 * p),
			A.Transpose(p=0.3 * p),
			A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.4 * p),
			A.Affine(
				scale=(0.9, 1.1),
				translate_percent={"x": 0.05, "y": 0.05},
				rotate=(-15, 15),
				shear={"x": (-10, 10), "y": (-10, 10)},
				interpolation=cv2.INTER_LINEAR,
				border_mode=cv2.BORDER_REFLECT_101,
				fit_output=False,
				p=0.4 * p,
			),
			# Photometric
			# A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.4 * p),
			A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4 * p),
			A.RandomGamma(gamma_limit=(80, 120), p=0.3 * p),
			# A.ChannelShuffle(p=0.2 * p),
			# Distortion
			A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2 * p),
		]

		# Stronger transforms for rare classes
		if p_scale > 1.0:
			transforms += [
				A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0, fill_mask=None, p=p),
				A.ElasticTransform(alpha=50, sigma=5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, approximate=False, p=p),
				A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=p),
			]

		# Final resize and normalize
		transforms += [A.Resize(self.img_res, self.img_res, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()]

		return A.Compose(transforms)

	def __getitem__(self, idx):
		class_idx, sample_idx = self.indices[idx]
		class_name = self.class_names[class_idx]

		mean, std = self.data_stats.get(class_name, self.data_stats.get("default"))

		img_path = Path(self.img_data[class_name][sample_idx])
		img = np.array(Image.open(img_path).convert("RGB"))

		if img.ndim == 2:
			img = np.stack([img] * 3, axis=-1)
		elif img.shape[0] == 3:
			img = img.transpose(1, 2, 0)

		if self.augmentation_params:
			# Dynamically read all augmentation values
			for k, v in self.augmentation_params.items():
				setattr(self, f"aug_{k}", v)

			p_scale = self.class_p_scale.get(class_name, 0.8)
			transform = self._get_albumentations_transform(mean=mean, std=std, p_scale=p_scale)
			img_resized = transform(image=img)["image"]
		else:
			img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
			img_resized = nnF.interpolate(img_tensor.unsqueeze(0), size=(self.img_res, self.img_res), mode="bilinear", align_corners=False).squeeze(0)
			img_resized = torchvision.transforms.functional.normalize(img_resized, mean=mean, std=std)

		label = self.img_labels[class_name][sample_idx]
		return img_resized, label
