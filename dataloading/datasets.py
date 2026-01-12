import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


# =====================
# NPY Dataset
# =====================
class NpyDetectionDataset(Dataset):
    def __init__(self, dataset_name, root_folder="datasets",
                 transform=None, target_transform=None, stats_file="stats_npy.npy"):
        self.dataset_name = dataset_name
        self.root_folder = root_folder
        self.transform = transform
        self.target_transform = target_transform

        # Paths to files
        self.img_file = os.path.join(root_folder, dataset_name, "npy_images.npy")

        # Load images from NPY file
        self.imgs = np.load(self.img_file)

        # Load label files
        self.label_files = sorted(glob.glob(os.path.join(root_folder, dataset_name, "labels", "*.txt")))

        # Verify image shape
        self.imgs = np.load(self.img_file)  # shape N,H,W,3
        if self.imgs.ndim != 4 or self.imgs.shape[-1] != 3:
            raise ValueError(f"The last axis must be 3 (RGB), found {self.imgs.shape[-1]}")

        # Load labels
        if self.label_files:
            self.labels = [torch.from_numpy(np.loadtxt(f, dtype=np.float32)) for f in self.label_files]
        else:
            print(" No labels found, creating empty labels")
            self.labels = [torch.zeros((0,5), dtype=torch.float32) for _ in range(len(self.imgs))]

        # Load or compute statistics (mean and std)
        stats_path = os.path.join(root_folder, dataset_name, stats_file)
        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            mean = self.imgs.mean(axis=(0,1,2))
            std  = self.imgs.std(axis=(0,1,2))
            self.stats = {"mean": mean, "std": std}
            np.save(stats_path, self.stats)
            print(f"{stats_file} not found → calculated automatically")

    def __len__(self):
        return len(self.imgs)

    def _normalize_image(self, img):
        # img: H,W,C → convert to tensor C,H,W
        img = torch.from_numpy(img.copy()).permute(2,0,1)
        mean = torch.tensor(self.stats['mean'], dtype=img.dtype).view(-1,1,1)
        std  = torch.tensor(self.stats['std'], dtype=img.dtype).view(-1,1,1)
        return (img - mean)/std

    def __getitem__(self, idx):
        # Retrieve image and label
        img = self.imgs[idx]
        img = self._normalize_image(img)
        label = self.labels[idx]

        # Apply optional transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


# =====================
# PIL Dataset
# =====================
class PilDetectionDataset(Dataset):
    def __init__(self, dataset_name, root_folder="datasets",
                 img_size=(304,304), stats_file="stats_pil.npy"):
        self.dataset_name = dataset_name
        self.root_folder = root_folder
        self.img_size = img_size

        # Image paths
        self.image_dir = os.path.join(root_folder, dataset_name, "images")
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))

        # Label paths
        self.label_files = sorted(glob.glob(
            os.path.join(root_folder, dataset_name, "labels", "*.txt")
        ))

        # Load labels
        self.labels = []
        label_dir = os.path.join(root_folder, dataset_name, "labels")

        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, base + ".txt")

            if os.path.exists(label_path):
                lbl = torch.from_numpy(np.loadtxt(label_path, dtype=np.float32))
                if lbl.ndim == 1:
                    lbl = lbl.unsqueeze(0)
            else:
                lbl = torch.zeros((0, 5), dtype=torch.float32)

            self.labels.append(lbl)

        # Compute or load statistics from resized PIL images
        stats_path = os.path.join(root_folder, dataset_name, stats_file)
        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            imgs = []
            for f in self.image_files:
                img = Image.open(f).convert("RGB").resize(img_size)
                imgs.append(np.array(img, dtype=np.float32) / 255.0)

            imgs = np.stack(imgs, axis=0)  # N,H,W,C
            mean = imgs.mean(axis=(0,1,2))
            std  = imgs.std(axis=(0,1,2))

            self.stats = {"mean": mean, "std": std}
            np.save(stats_path, self.stats)
            print("stats_pil.npy created")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and convert to tensor
        img = Image.open(self.image_files[idx]).convert("RGB").resize(self.img_size)
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)  # C,H,W

        # Normalize using precomputed statistics
        mean = torch.tensor(self.stats["mean"], dtype=img.dtype).view(-1,1,1)
        std  = torch.tensor(self.stats["std"], dtype=img.dtype).view(-1,1,1)
        img = (img - mean) / std

        label = self.labels[idx]
        return img, label


#------------------------------------------------------------------------------
# RAM Dataset
#------------------------------------------------------------------------------
class RAMDetectionDataset(VisionDataset):
    # Load images into RAM during initialization
    # Resize to img_size and normalize using statistics

    def __init__(self, dataset_name, root_folder="datasets", img_size=(304,304),
                 stats_file="stats_ram.npy"):
        super().__init__(root=root_folder)

        self.dataset_name = dataset_name
        self.root_folder = root_folder
        self.img_size = img_size

        # Image paths
        self.image_dir = os.path.join(root_folder, dataset_name, "images")
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))
        self.label_dir = os.path.join(root_folder, dataset_name, "labels")

        # Load labels robustly
        self.labels = []
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, base + ".txt")

            if os.path.exists(label_path):
                lbl = torch.from_numpy(np.loadtxt(label_path, dtype=np.float32))
                if lbl.ndim == 1:
                    lbl = lbl.unsqueeze(0)
            else:
                lbl = torch.zeros((0,5), dtype=torch.float32)
            self.labels.append(lbl)

        # Load and resize all images into RAM
        self.imgs = []
        for f in self.image_files:
            img = Image.open(f).convert("RGB").resize(img_size)
            img = np.array(img, dtype=np.float32)/255.0
            self.imgs.append(img)
        self.imgs = np.stack(self.imgs, axis=0)  # N,H,W,C

        # Compute or load statistics
        stats_path = os.path.join(root_folder, dataset_name, stats_file)
        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            mean = self.imgs.mean(axis=(0,1,2))
            std  = self.imgs.std(axis=(0,1,2))
            self.stats = {"mean": mean, "std": std}
            np.save(stats_path, self.stats)
            print(f"{stats_file} created automatically")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Convert image from H,W,C to C,H,W and normalize
        img = torch.from_numpy(self.imgs[idx]).permute(2,0,1)
        mean = torch.tensor(self.stats["mean"], dtype=img.dtype).view(-1,1,1)
        std  = torch.tensor(self.stats["std"], dtype=img.dtype).view(-1,1,1)
        img = (img - mean)/std

        label = self.labels[idx]
        return img, label
    