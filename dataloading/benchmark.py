import time
import torch
from torch.utils.data import DataLoader
from datasets import NpyDetectionDataset, PilDetectionDataset, RAMDetectionDataset

NUM_EPOCHS = 5
BATCH_SIZE = 32
dataset_name = "coco128"
root_folder = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Data"

def benchmark_dataloader(name, dataloader, num_epochs=NUM_EPOCHS):
    start_time = time.time()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            _ = images * 2  # Dummy operation
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_epochs
    print(f"{name}: Total={total_time:.2f}s | Avg/Epoch={avg_time:.2f}s")
    return total_time, avg_time

if __name__ == "__main__":
    results = []

    # --------------------------
    # NPY Dataset
    # --------------------------
    print("==== Benchmark NPY Dataset ====")
    dataset_npy = NpyDetectionDataset(
        dataset_name,
        root_folder=root_folder,
        stats_file="stats_npy.npy"
    )
    dataloader_npy = DataLoader(dataset_npy, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total, avg = benchmark_dataloader("NPY", dataloader_npy)
    results.append(("NPY", total, avg))

    # --------------------------
    # PIL Dataset
    # --------------------------
    print("==== Benchmark PIL Dataset ====")
    dataset_pil = PilDetectionDataset(
        dataset_name,
        root_folder=root_folder,
        img_size=(304,304),
        stats_file="stats_pil.npy"
    )
    dataloader_pil = DataLoader(dataset_pil, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total, avg = benchmark_dataloader("PIL", dataloader_pil)
    results.append(("PIL", total, avg))

    # --------------------------
    # RAM Dataset
    # --------------------------
    print("==== Benchmark RAM Dataset ====")
    dataset_ram = RAMDetectionDataset(
        dataset_name,
        root_folder=root_folder,
        img_size=(304,304),
        stats_file="stats_ram.npy"
    )
    dataloader_ram = DataLoader(dataset_ram, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total, avg = benchmark_dataloader("RAM", dataloader_ram)
    results.append(("RAM", total, avg))

    # --------------------------
    # Tableau comparatif
    # --------------------------
    print("\n==== Tableau comparatif ====")
    print(f"{'Dataset':<10} | {'Total(s)':<10} | {'Avg/Epoch(s)':<12}")
    print("-"*38)
    for name, total, avg in results:
        print(f"{name:<10} | {total:<10.2f} | {avg:<12.2f}")
