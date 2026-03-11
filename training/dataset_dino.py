import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _get_sorted_label_files(root_dir):
	def _numeric_sort_key(path):
		stem = Path(path).stem
		return (0, int(stem)) if stem.isdigit() else (1, stem)

	label_dir = os.path.join(root_dir, "labels")
	label_files = [str(path) for path in Path(label_dir).glob("*.txt")]
	return sorted(label_files, key=_numeric_sort_key)


def parse_label_line(line):
    class_id, x_center, y_center, width, height = line.split()
    return int(class_id), [float(x_center), float(y_center), float(width), float(height)]


def load_stats(stats_path):
    stats_obj = np.load(stats_path, allow_pickle=True)
    stats = stats_obj.item() if hasattr(stats_obj, "item") else stats_obj
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(3, 1, 1)
    return mean, torch.clip(std, 1e-6, None)


class Coco128NpyDataset(Dataset):
    def __init__(self, root="datasets/coco128", device="cuda"):
        self.root = root
        self.device = device
        self.images = np.load(os.path.join(root, "npy_images.npy")).astype(np.float32)
        self.mean, self.std = load_stats(os.path.join(root, "stats.npy"))
        label_files = _get_sorted_label_files(root)
        self.targets = [self._read_target(path, device) for path in label_files]
        self.num_classes = 1 + max((int(t["labels"].max().item()) for t in self.targets if len(t["labels"]) > 0), default=0)

    def _read_target(self, label_path, device):
        labels = []
        boxes = []
        if label_path is not None and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id, bbox = parse_label_line(line)
                    labels.append(class_id)
                    boxes.append(bbox)
        return {
            "labels": torch.tensor(labels, dtype=torch.int64, device=device),
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        }

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()
        img = (img - self.mean) / self.std
        return img, self.targets[idx]


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
