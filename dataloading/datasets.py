import glob
import os
import numpy as np
from torchvision.datasets import VisionDataset
import torch

class NpyDetectionDataset(VisionDataset):

	def __init__(
			self, 
			dataset_name, 
			img_size=(304,304), 
			root_folder="datasets",  
			transform=None, 
			target_transform=None
		):
		self.transform = transform
		self.target_transform = target_transform
		self.dataset_name = dataset_name
		self.img_size = img_size
		self.root_folder = root_folder
		self.img_files = os.path.join(self.root_folder, dataset_name,'npy_images.npy')
		# /!\ naive sorted may be too fragile to ensure alignment between images and labels
		self.label_files = sorted(glob.glob(os.path.join(self.root_folder, dataset_name, 'labels', "*.txt")))
		self.imgs = np.lib.format.open_memmap(self.img_files, dtype=np.float32, mode='r')
		# Load labels as a list of numpy arrays into RAM
		self.labels = [torch.from_numpy(np.loadtxt(label_file, dtype=np.float32))  for label_file in self.label_files]
		self.stats = np.load(os.path.join(self.root_folder, dataset_name, "stats.npy"), allow_pickle=True)

	def _normalize_image(self, img):
		mean, std = self.stats['mean'], self.stats['std']
		return (img - mean) / std

	def __getitem__(self, idx):
		img = self.imgs[idx]
		normalized_img = self._normalize_image(img)
		label = self.labels[idx]
		if self.transform:
			normalized_img = self.transform(normalized_img)
		if self.target_transform:
			label = self.target_transform(label)
		return normalized_img, label
	
class PILDetectionDataset(VisionDataset):

	# Read images from disk on the fly using PIL during forward pass
	# Resize to img_size (304,304) and normalize using stats.npy
	def __init__(self):
		pass

class RAMDetectionDataset(VisionDataset):
	# Read images from disk using PIL during __init__ and index into array during forward pass
	# Resize to img_size (304,304) and normalize using stats.npy

	def __init__(self):
		pass