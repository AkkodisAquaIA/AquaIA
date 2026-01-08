import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchvision.datasets import VisionDataset

def create_npy_images(dataset_name, root_folder, img_size=(304,304)):
    image_dir = os.path.join(root_folder, dataset_name, "images")
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))

    if not image_files:
        raise RuntimeError("Aucune image trouvée dans images/")

    all_images = []
    for f in image_files:
        img = Image.open(f).convert("RGB").resize(img_size)
        img = np.array(img, dtype=np.float32)
        all_images.append(img)

    imgs = np.stack(all_images, axis=0)  # N,H,W,3
    out_path = os.path.join(root_folder, dataset_name, "npy_images.npy")
    np.save(out_path, imgs)

    print(f" npy_images.npy créé : {imgs.shape}")

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

        # fichiers
        self.img_file = os.path.join(root_folder, dataset_name, "npy_images.npy")

        if not os.path.exists(self.img_file):
            print(" npy_images.npy absent → création automatique")
            create_npy_images(dataset_name, root_folder)

        self.imgs = np.load(self.img_file)

        self.label_files = sorted(glob.glob(os.path.join(root_folder, dataset_name, "labels", "*.txt")))

        # lecture images
        self.imgs = np.load(self.img_file)  # shape N,H,W,3
        if self.imgs.ndim != 4 or self.imgs.shape[-1] != 3:
            raise ValueError(f"Le dernier axe doit être 3 (RGB), trouvé {self.imgs.shape[-1]}")

        # lecture labels
        if self.label_files:
            self.labels = [torch.from_numpy(np.loadtxt(f, dtype=np.float32)) for f in self.label_files]
        else:
            print(" Aucun label trouvé, création automatique de labels vides")
            self.labels = [torch.zeros((0,5), dtype=torch.float32) for _ in range(len(self.imgs))]

        # stats
        stats_path = os.path.join(root_folder, dataset_name, stats_file)
        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            mean = self.imgs.mean(axis=(0,1,2))
            std  = self.imgs.std(axis=(0,1,2))
            self.stats = {"mean": mean, "std": std}
            np.save(stats_path, self.stats)
            print(f"{stats_file} absent → calculé automatiquement")

    def __len__(self):
        return len(self.imgs)

    def _normalize_image(self, img):
        # img: H,W,C → convertir en tensor C,H,W
        img = torch.from_numpy(img.copy()).permute(2,0,1)
        mean = torch.tensor(self.stats['mean'], dtype=img.dtype).view(-1,1,1)
        std  = torch.tensor(self.stats['std'], dtype=img.dtype).view(-1,1,1)
        return (img - mean)/std

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = self._normalize_image(img)
        label = self.labels[idx]

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

        self.image_dir = os.path.join(root_folder, dataset_name, "images")
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))
        self.label_files = sorted(glob.glob(
            os.path.join(root_folder, dataset_name, "labels", "*.txt")
        ))

        # labels
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

        # stats (calculées sur images PIL redimensionnées)
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
            print("stats_pil.npy créé")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # image
        img = Image.open(self.image_files[idx]).convert("RGB").resize(self.img_size)
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)  # C,H,W

        # normalisation identique à NPY
        mean = torch.tensor(self.stats["mean"], dtype=img.dtype).view(-1,1,1)
        std  = torch.tensor(self.stats["std"], dtype=img.dtype).view(-1,1,1)
        img = (img - mean) / std

        label = self.labels[idx]
        return img, label


#------------------------------------------------------------------------------

class RAMDetectionDataset(VisionDataset):
	# Read images from disk using PIL during __init__ and index into array during forward pass
	# Resize to img_size (304,304) and normalize using stats.npy

    def __init__(self, dataset_name, root_folder="datasets", img_size=(304,304),
                 stats_file="stats_ram.npy"):
        super().__init__(root=root_folder)

        self.dataset_name = dataset_name
        self.root_folder = root_folder
        self.img_size = img_size

        # fichiers
        self.image_dir = os.path.join(root_folder, dataset_name, "images")
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))
        self.label_dir = os.path.join(root_folder, dataset_name, "labels")

        # labels robustes
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

        # lecture et resize des images en RAM
        self.imgs = []
        for f in self.image_files:
            img = Image.open(f).convert("RGB").resize(img_size)
            img = np.array(img, dtype=np.float32)/255.0
            self.imgs.append(img)
        self.imgs = np.stack(self.imgs, axis=0)  # N,H,W,C

        # stats
        stats_path = os.path.join(root_folder, dataset_name, stats_file)
        if os.path.exists(stats_path):
            self.stats = np.load(stats_path, allow_pickle=True).item()
        else:
            mean = self.imgs.mean(axis=(0,1,2))
            std  = self.imgs.std(axis=(0,1,2))
            self.stats = {"mean": mean, "std": std}
            np.save(stats_path, self.stats)
            print(f"{stats_file} créé automatiquement")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # img: H,W,C → C,H,W
        img = torch.from_numpy(self.imgs[idx]).permute(2,0,1)
        mean = torch.tensor(self.stats["mean"], dtype=img.dtype).view(-1,1,1)
        std  = torch.tensor(self.stats["std"], dtype=img.dtype).view(-1,1,1)
        img = (img - mean)/std

        label = self.labels[idx]
        return img, label