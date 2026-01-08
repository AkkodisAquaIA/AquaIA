import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import NpyDetectionDataset, PilDetectionDataset, RAMDetectionDataset
import psutil
import matplotlib.pyplot as plt
from PIL import Image

NUM_EPOCHS = 5
BATCH_SIZE = 32
dataset_name = "coco128"
root_folder = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Data"

process = psutil.Process(os.getpid())

# -----------------------------------
# Benchmark étape par étape
# -----------------------------------
def benchmark_steps(dataset, name, num_epochs=NUM_EPOCHS):
    ram_dataset = process.memory_info().rss / (1024**2)

    read_times, resize_times, norm_times = [], [], []
    start_total = time.time()

    for epoch in range(num_epochs):
        for idx in range(len(dataset)):
            img, label = dataset[idx]

            if isinstance(dataset, PilDetectionDataset):
                # Lecture disque
                t0 = time.time()
                pil_img = Image.open(dataset.image_files[idx]).convert("RGB")
                read_times.append(time.time() - t0)

                # Resize
                t0 = time.time()
                pil_img = pil_img.resize(dataset.img_size)
                resize_times.append(time.time() - t0)

                # Normalisation
                t0 = time.time()
                img_tensor = torch.from_numpy(np.array(pil_img, dtype=np.float32)/255.0).permute(2,0,1)
                mean = torch.tensor(dataset.stats["mean"], dtype=img_tensor.dtype).view(-1,1,1)
                std  = torch.tensor(dataset.stats["std"], dtype=img_tensor.dtype).view(-1,1,1)
                img_tensor = (img_tensor - mean)/std
                norm_times.append(time.time() - t0)

            elif isinstance(dataset, RAMDetectionDataset):
                # Resize fait dans __init__, normalisation dans __getitem__
                t0 = time.time()
                _ = img
                read_times.append(0.0)
                resize_times.append(0.0)
                norm_times.append(time.time() - t0)

            elif isinstance(dataset, NpyDetectionDataset):
                t0 = time.time()
                _ = img
                read_times.append(0.0)
                resize_times.append(0.0)
                norm_times.append(time.time() - t0)

    total_time = time.time() - start_total
    avg_time = total_time / num_epochs

    return {
        "name": name,
        "total_time": total_time,
        "avg_time": avg_time,
        "ram_dataset": ram_dataset,
        "read_time": np.sum(read_times),
        "resize_time": np.sum(resize_times),
        "norm_time": np.sum(norm_times)
    }

# -----------------------------------
# Création datasets
# -----------------------------------
datasets = [
    NpyDetectionDataset(dataset_name, root_folder=root_folder, stats_file="stats_npy.npy"),
    PilDetectionDataset(dataset_name, root_folder=root_folder, img_size=(304,304), stats_file="stats_pil.npy"),
    RAMDetectionDataset(dataset_name, root_folder=root_folder, img_size=(304,304), stats_file="stats_ram.npy")
]

# -----------------------------------
# Benchmark
# -----------------------------------
results = []
for ds in datasets:
    print(f"\n==== Benchmark {ds.__class__.__name__} ====")
    r = benchmark_steps(ds, ds.__class__.__name__)
    results.append(r)
    print(f"{r['name']}: Total={r['total_time']:.2f}s | Avg/Epoch={r['avg_time']:.2f}s | RAM={r['ram_dataset']:.1f} MB")
    if r["read_time"] > 0:
        print(f"  Read: {r['read_time']:.2f}s | Resize: {r['resize_time']:.2f}s | Norm: {r['norm_time']:.2f}s")

# -----------------------------------
# Tableau comparatif
# -----------------------------------
print("\n==== Tableau comparatif ====")
print(f"{'Dataset':<12} | {'Total(s)':<10} | {'Avg/Epoch(s)':<12} | {'RAM(MB)':<10}")
print("-"*50)
for r in results:
    print(f"{r['name']:<12} | {r['total_time']:<10.2f} | {r['avg_time']:<12.2f} | {r['ram_dataset']:<10.1f}")

# -----------------------------------
# Premier graphique : Total Time vs RAM
# -----------------------------------
labels = [r['name'] for r in results]
total_times = [r['total_time'] for r in results]
ram_values = [r['ram_dataset'] for r in results]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,5))

ax1.bar(x - width/2, total_times, width, label='Total Time (s)', color='skyblue')
ax1.set_ylabel('Time (s)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title('Benchmark NPY / RAM / PIL - Temps vs RAM')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.bar(x + width/2, ram_values, width, label='RAM (MB)', color='salmon')
ax2.set_ylabel('RAM (MB)')
ax2.legend(loc='upper right')

plt.show()

# -----------------------------------
# Deuxième graphique : temps empilé par étape
# -----------------------------------
read_times = [r['read_time'] for r in results]
resize_times = [r['resize_time'] for r in results]
norm_times = [r['norm_time'] for r in results]

plt.figure(figsize=(8,5))
plt.bar(x, read_times, width, label='Read', color='lightgreen')
plt.bar(x, resize_times, width, bottom=np.array(read_times), label='Resize', color='orange')
plt.bar(x, norm_times, width, bottom=np.array(read_times)+np.array(resize_times), label='Norm', color='skyblue')

plt.xticks(x, labels)
plt.ylabel('Time (s)')
plt.title('Temps par étape par dataset (empilé)')
plt.legend()
plt.show()
