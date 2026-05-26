from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _get_sorted_jpg_files(root_dir):
	def _numeric_sort_key(path):
		stem = Path(path).stem
		return (0, int(stem)) if stem.isdigit() else (1, stem)

	img_dir = os.path.join(root_dir, "images")
	jpg_files = [str(path) for path in Path(img_dir).glob("*.jpg")]
	return sorted(jpg_files, key=_numeric_sort_key)


@pipeline_def
def _transform_jpg(files, size=640):
	jpgs = fn.readers.file(files=files, name="Reader")[0]
	images = fn.decoders.image(jpgs, device="mixed", output_type=types.RGB)
	images = fn.resize(images, resize_x=size, resize_y=size, min_filter=types.DALIInterpType.INTERP_TRIANGULAR)
	images = fn.transpose(images, device="gpu", perm=[2, 0, 1])

	return images


def convert_to_npy(root_dir, batch_size=128, num_threads=4, device_id=0, compute_stats=False, image_size=640):
	sorted_files = _get_sorted_jpg_files(root_dir)
	if not sorted_files:
		raise FileNotFoundError(f"No jpg files found under {os.path.join(root_dir, 'images')}")
	pipe = _transform_jpg(files=sorted_files, size=image_size, batch_size=batch_size, num_threads=num_threads, device_id=device_id)
	pipe.build()
	n = pipe.epoch_size("Reader")
	print(f"Found {n} images")
	num_iters = (n + batch_size - 1) // batch_size

	batches = []
	for _ in range(num_iters):
		batch = pipe.run()[0]  # TensorListGPU
		batches.append(batch.as_cpu().as_array())  # (B, 3, image_size, image_size)

	np_images = np.concatenate(batches, axis=0)[:n].astype(np.float32) / 255.0

	image_dir = os.path.join(root_dir, "npy_images.npy")
	np.save(image_dir, np_images)

	if compute_stats:
		mean = np.mean(np_images, axis=(0, 2, 3))
		std = np.std(np_images, axis=(0, 2, 3))
		stats = {"mean": mean, "std": std}
		np.save(os.path.join(root_dir, "stats.npy"), stats)
	return np_images


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Preprocess jpg images to .npy format using DALI")
	parser.add_argument("--data-dir", type=str, default="datasets", help="Directory containing datasets")
	parser.add_argument("--dataset", type=str, default="coco_small", help="Name of the dataset")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size for DALI pipeline")
	parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for DALI pipeline")
	parser.add_argument("--image-size", type=int, default=640, help="Output image size")
	args = parser.parse_args()

	root_dir = os.path.join(BASE_DIR, args.data_dir, args.dataset)
	image_dir = os.path.join(root_dir, "npy_images.npy")

	if not os.path.exists(image_dir):
		convert_to_npy(root_dir=root_dir, batch_size=args.batch_size, num_threads=args.num_threads, compute_stats=True, image_size=args.image_size)