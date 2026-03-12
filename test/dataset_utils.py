import random
import torch

from dataloading.datasets import NpyDetectionDataset, dataset_config_from_path


def sample_dataset(dataset_root, num_samples, seed):
    dataset_config = dataset_config_from_path(dataset_root)
    dataset = NpyDetectionDataset(
        dataset_name=dataset_config.dataset_name,
        root_folder=dataset_config.root_folder,
        stats_file=dataset_config.stats_file,
        load_targets=False,
        return_image_id=True,
    )
    rng = random.Random(seed)
    sample_size = min(num_samples, len(dataset))
    sampled_indices = sorted(rng.sample(range(len(dataset)), sample_size))

    samples = [dataset[index] for index in sampled_indices]
    images = torch.stack([sample[0] for sample in samples], dim=0)
    sampled_image_ids = [sample[1] for sample in samples]
    return images, sampled_image_ids
