import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets import (
    NpyDetectionDataset,
    PilDetectionDataset,
    RAMDetectionDataset
)

@dataclass
class BenchmarkConfig:
    num_epochs: int = 5
    batch_size: int = 32
    num_workers: int = 0


def benchmark_dataset(
    dataset,
    name: str,
    cfg: BenchmarkConfig
) -> Dict[str, Any]:
    """
    Measure iteration time and RAM usage for a dataset.
    """

    process = psutil.Process(os.getpid())
    ram_dataset = process.memory_info().rss / (1024 ** 2)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    start_time = time.time()

    for _ in range(cfg.num_epochs):
        for images, labels in loader:
            _ = images * 2  # simulate workload

    total_time = time.time() - start_time

    return {
        "name": name,
        "total_time": total_time,
        "avg_time": total_time / cfg.num_epochs,
        "ram_dataset": ram_dataset
    }

if __name__ == "__main__":

    cfg = BenchmarkConfig()

    cwd = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA"
    root_folder = os.path.join(cwd, "Data")
    dataset_name = "coco128"

    print("Loading datasets...")

    datasets = [
        NpyDetectionDataset(dataset_name, root_folder, "stats_npy.npy"),
        PilDetectionDataset(dataset_name, root_folder, (304, 304), "stats_pil.npy"),
        RAMDetectionDataset(dataset_name, root_folder, (304, 304), "stats_ram.npy")
    ]

    results: List[Dict[str, Any]] = []

    for ds in datasets:
        print(f"\n==== Benchmark {ds.__class__.__name__} ====")
        r = benchmark_dataset(ds, ds.__class__.__name__, cfg)
        results.append(r)

        print(
            f"{r['name']}: "
            f"Total={r['total_time']:.2f}s | "
            f"Avg/Epoch={r['avg_time']:.2f}s | "
            f"RAM={r['ram_dataset']:.1f} MB"
        )

