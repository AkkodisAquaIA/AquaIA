import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import NpyDetectionDataset, RAMDetectionDataset
import psutil
from PIL import Image
import matplotlib.pyplot as plt

# ======================================
# PIL → NPY Dataset
# ======================================
class PilToNpyDataset(Dataset):
    def __init__(self, root_folder, dataset_name, img_size=(304,304), npy_file="pillow_images.npy", stats_file="stats_pil.npy"):
        self.root_folder = root_folder
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.npy_path = os.path.join(root_folder, dataset_name, npy_file)
        self.label_dir = os.path.join(root_folder, dataset_name, "labels")
        self.label_files = sorted([os.path.join(self.label_dir,f) for f in os.listdir(self.label_dir) if f.endswith(".txt")])

        # Labels
        if self.label_files:
            self.labels = [torch.from_numpy(np.loadtxt(f,dtype=np.float32)) for f in self.label_files]
        else:
            self.labels = [torch.zeros((0,5), dtype=torch.float32) for _ in range(len(os.listdir(os.path.join(root_folder,dataset_name,"images"))))]

        # Vérifie si le .npy existe
        if os.path.exists(self.npy_path):
            print(f"Chargement {self.npy_path} depuis disque...")
            self.imgs = np.load(self.npy_path)
        else:
            print(f"{self.npy_path} absent → génération depuis PIL...")
            image_dir = os.path.join(root_folder, dataset_name, "images")
            image_files = sorted([os.path.join(image_dir,f) for f in os.listdir(image_dir)])
            all_imgs = []
            for f in image_files:
                img = Image.open(f).convert("RGB").resize(img_size)
                all_imgs.append(np.array(img,dtype=np.float32)/255.0)
            self.imgs = np.stack(all_imgs)  # N,H,W,C
            np.save(self.npy_path, self.imgs)
            print(f"{self.npy_path} créé.")

        # Stats
        stats_path = os.path.join(root_folder, dataset_name, stats_file)
        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True).item()
            self.mean, self.std = stats['mean'], stats['std']
        else:
            self.mean = self.imgs.mean(axis=(0,1,2))
            self.std = self.imgs.std(axis=(0,1,2))
            np.save(stats_path, {'mean': self.mean, 'std': self.std})
            print(f"{stats_file} créé.")

        # Préparer images en torch tensor
        self.images = []
        for img in self.imgs:
            img_tensor = torch.from_numpy(img).permute(2,0,1)
            mean_tensor = torch.tensor(self.mean, dtype=img_tensor.dtype).view(-1,1,1)
            std_tensor  = torch.tensor(self.std, dtype=img_tensor.dtype).view(-1,1,1)
            img_tensor = (img_tensor - mean_tensor)/std_tensor
            self.images.append(img_tensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ======================================
# Fonction benchmark principale
# ======================================
def benchmark_dataset(dataset, name, batch_size=32, num_workers=0, num_epochs=5):
    process = psutil.Process(os.getpid())
    ram_dataset = process.memory_info().rss / (1024**2)  # MB

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    start_time = time.time()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            _ = images * 2
    total_time = time.time() - start_time
    avg_time = total_time / num_epochs

    return {
        "name": name,
        "total_time": total_time,
        "avg_time": avg_time,
        "ram_dataset": ram_dataset
    }

# ======================================
# Fonction pour mesurer le temps par étape
# ======================================
def measure_steps(dataset):
    read_times, resize_times, norm_times = [], [], []

    for idx in range(len(dataset)):
        start = time.time()
        # Lecture
        if isinstance(dataset, PilToNpyDataset):
            # déjà chargé → lecture minimale
            img = dataset.imgs[idx]
            t_read = time.time() - start
            read_times.append(t_read)

            # Resize
            start_resize = time.time()
            img_resized = torch.from_numpy(img).permute(2,0,1)
            t_resize = time.time() - start_resize
            resize_times.append(t_resize)

            # Normalisation
            start_norm = time.time()
            mean_tensor = torch.tensor(dataset.mean, dtype=img_resized.dtype).view(-1,1,1)
            std_tensor  = torch.tensor(dataset.std, dtype=img_resized.dtype).view(-1,1,1)
            img_norm = (img_resized - mean_tensor)/std_tensor
            t_norm = time.time() - start_norm
            norm_times.append(t_norm)
        else:
            # Npy/RAM déjà torch → lecture + resize minimal
            start_resize = time.time()
            img = dataset[idx][0]
            t_resize = time.time() - start_resize
            resize_times.append(t_resize)
            read_times.append(0.0)
            start_norm = time.time()
            _ = img
            t_norm = time.time() - start_norm
            norm_times.append(t_norm)

    read_ms   = np.mean(read_times)*1000
    resize_ms = np.mean(resize_times)*1000
    norm_ms   = np.mean(norm_times)*1000
    return read_ms, resize_ms, norm_ms

# ======================================
# Bloc principal
# ======================================
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    dataset_name = "coco128"
    root_folder = "c:/Users/Pierre.FANCELLI/Documents/___Dev/Aqua-IA/Data"

    datasets = [
        NpyDetectionDataset(dataset_name, root_folder=root_folder, stats_file="stats_npy.npy"),
        PilToNpyDataset(root_folder, dataset_name, img_size=(304,304), npy_file="pillow_images.npy", stats_file="stats_pil.npy"),
        RAMDetectionDataset(dataset_name, root_folder=root_folder, img_size=(304,304), stats_file="stats_ram.npy")
    ]

    results = []
    step_times = []
    for ds in datasets:
        print(f"\n==== Benchmark {ds.__class__.__name__} ====")
        r = benchmark_dataset(ds, ds.__class__.__name__, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, num_epochs=NUM_EPOCHS)
        results.append(r)
        print(f"{r['name']}: Total={r['total_time']:.2f}s | Avg/Epoch={r['avg_time']:.2f}s | RAM={r['ram_dataset']:.1f} MB")

        # Mesure par étapes
        read_ms, resize_ms, norm_ms = measure_steps(ds)
        step_times.append((ds.__class__.__name__, read_ms, resize_ms, norm_ms))

    # ======================================
    # Tableau comparatif
    # ======================================
    print("\n==== Tableau comparatif ====")
    header_fmt = "{:<22} | {:>10} | {:>12} | {:>10}"
    row_fmt    = "{:<22} | {:>10.2f} | {:>12.2f} | {:>10.1f}"
    print(header_fmt.format("Dataset", "Total(s)", "Avg/Epoch(s)", "RAM(MB)"))
    print("-"*60)
    for r in results:
        print(row_fmt.format(r['name'], r['total_time'], r['avg_time'], r['ram_dataset']))

    # ======================================
    # Tableau temps par image (ms)
    # ======================================
    print("\n==== Temps moyen par image (ms) ====")
    header_fmt = "{:<22} | {:>10} | {:>10} | {:>10}"
    row_fmt    = "{:<22} | {:>10.2f} | {:>10.2f} | {:>10.2f}"
    print(header_fmt.format("Dataset", "Read(ms)", "Resize(ms)", "Norm(ms)"))
    print("-"*60)
    for name, r_ms, resize_ms, norm_ms in step_times:
        print(row_fmt.format(name, r_ms, resize_ms, norm_ms))

# ======================================
# Graphique 1 : Total Time vs RAM
# ======================================
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
ax1.set_title('Benchmark NPY / RAM / PIL→NPY - Temps vs RAM')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.bar(x + width/2, ram_values, width, label='RAM (MB)', color='salmon')
ax2.set_ylabel('RAM (MB)')
ax2.legend(loc='upper right')
fig.savefig("benchmark_total_vs_ram.png", bbox_inches='tight', dpi=150)
plt.show()

# ======================================
# Graphique 2 : Total Time + RAM annotée
# ======================================
plt.figure(figsize=(8,5))
plt.bar(x, total_times, width, label='Total Time', color='skyblue')
for i, ram in enumerate(ram_values):
    plt.text(i, total_times[i]+0.05, f'{ram:.0f} MB', ha='center', va='bottom', color='red')
plt.xticks(x, labels)
plt.ylabel('Time (s)')
plt.title('Temps total par dataset + RAM')
plt.legend()
plt.savefig("benchmark_total.png", bbox_inches='tight', dpi=150)
plt.show()

# ======================================
# Graphique 3 : Temps moyen par image (Read/Resize/Norm)
# ======================================
read_vals = [x[1] for x in step_times]
resize_vals = [x[2] for x in step_times]
norm_vals = [x[3] for x in step_times]

plt.figure(figsize=(8,5))
plt.bar(x - width, read_vals, width, label='Read(ms)', color='skyblue')
plt.bar(x, resize_vals, width, label='Resize(ms)', color='lightgreen')
plt.bar(x + width, norm_vals, width, label='Norm(ms)', color='salmon')
plt.xticks(x, labels)
plt.ylabel('Time per image (ms)')
plt.title('Temps moyen par image par étape')
plt.legend()
plt.tight_layout()
plt.savefig("benchmark_steps.png", dpi=150)
plt.show()

