import random
from pathlib import Path

import numpy as np
import torch

from training.dataset_dino import load_stats


class InferenceNpyDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.images = np.load(self.root / "npy_images.npy").astype(np.float32)
        self.mean, self.std = load_stats(self.root / "stats.npy")
        self.image_ids = self._load_image_ids()

    def _load_image_ids(self):
        image_dir = self.root / "images"
        if image_dir.exists():
            image_paths = sorted(image_dir.iterdir(), key=lambda path: path.stem)
            image_ids = [path.stem for path in image_paths if path.is_file()]
            if len(image_ids) == len(self.images):
                return image_ids
        return [str(index) for index in range(len(self.images))]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()
        img = (img - self.mean) / self.std
        return img, self.image_ids[idx]


def sample_dataset(dataset_root, num_samples, seed):
    dataset = InferenceNpyDataset(root=dataset_root)
    rng = random.Random(seed)
    sample_size = min(num_samples, len(dataset))
    sampled_indices = sorted(rng.sample(range(len(dataset)), sample_size))

    samples = [dataset[index] for index in sampled_indices]
    images = torch.stack([sample[0] for sample in samples], dim=0)
    sampled_image_ids = [sample[1] for sample in samples]
    return images, sampled_image_ids
