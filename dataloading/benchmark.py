import torch
from .datasets import NpyDetectionDataset
from torch.utils.data import DataLoader
import time

NUM_EPOCHS = 5
dataset_npy = NpyDetectionDataset('coco128')
# ... other datasets type

dataloader_npy = DataLoader(dataset_npy, batch_size=32, shuffle=True, num_workers=4)

def benchmark_dataloader(dataloader, num_epochs=NUM_EPOCHS):
	start_time = time.time()
	for epoch in range(num_epochs):
		for images, labels in dataloader:
			# Simulate a training step
			_ = images * 2  # Dummy operation
	end_time = time.time()
	total_time = end_time - start_time
	print(f"Total time for {num_epochs} epochs: {total_time:.2f} seconds")
	print(f"Average time per epoch: {total_time / num_epochs:.2f} seconds")
