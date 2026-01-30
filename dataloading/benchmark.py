import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import argparse

import psutil
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


def read_param() -> Tuple[Path, Path]:
    """
    Parse command-line arguments and validate the data directory.

    Returns:
        Tuple[Path, Path]: A tuple containing:
            - the base working directory
            - the data directory (working_dir / folder)
    """
    parser = argparse.ArgumentParser(
        description="Program using a base path and a data folder"
    )

    parser.add_argument(
        "-p",
        "--work_dir",
        required=True,
        help="Base path (e.g. C:/mes_data)"
    )

    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        help="Folder containing the data (inside the base path)"
    )

    args = parser.parse_args()

    base_path: Path = Path(args.work_dir)
    folder_name: str = args.folder

    # Build the full data path
    data_path: Path = base_path / folder_name

    # Check that the base path exists
    if not base_path.exists():
        print(f"Error: base path does not exist -> {base_path}")
        sys.exit(1)

    # Check that the data directory exists and is a directory
    if not data_path.exists():
        print(f"Error: data directory does not exist -> {data_path}")
        sys.exit(1)

    if not data_path.is_dir():
        print(f"Error: data path is not a directory -> {data_path}")
        sys.exit(1)

    return base_path, data_path

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

    # Lecture des chemins depuis les arguments de la ligne de commande
    working_path, folder = read_param()
    
    # root_folder = os.path.join(working_path, folder)
    # dataset_name = folder

    print("Loading datasets...")

    datasets = [
        NpyDetectionDataset(folder, working_path, "stats_npy.npy"),
        PilDetectionDataset(folder, working_path, (304, 304), "stats_pil.npy"),
        RAMDetectionDataset(folder, working_path, (304, 304), "stats_ram.npy")
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

