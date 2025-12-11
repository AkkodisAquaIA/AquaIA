from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import numpy as np

@pipeline_def
def transform_jpeg(root_dir, size=304):
    jpegs = fn.readers.file(file_root=root_dir, file_filters="*.jpg")[0]
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.transpose(images, device="gpu", perm=[2,0,1])
    images = fn.resize(
        images,
        resize_x=size,
        resize_y=size,
        min_filter=types.DALIInterpType.INTERP_TRIANGULAR
    )

    return images

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess jpeg images to .npy format using DALI")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Directory containing datasets")
    parser.add_argument("--dataset", type=str, default="coco128", help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for DALI pipeline")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for DALI pipeline")
    args = parser.parse_args()

    root_dir = os.path.join(args.data_dir, args.dataset)
    image_dir = os.path.join(root_dir, "npy_images")

    pipe = transform_jpeg(root_dir=root_dir,batch_size=args.batch_size, num_threads=args.num_threads, device_id=0)
    pipe.build()
    images = pipe.run()[0]
    np_images = images.as_cpu().as_array().astype(np.float32)/255.0  # Normalize to [0, 1], float32
    np.save(image_dir, np_images)

    # Compute mean and std
    mean = np.mean(np_images, axis=(0, 2, 3))
    std = np.std(np_images, axis=(0, 2, 3))
    stats = {'mean': mean, 'std': std}
    np.save(os.path.join(root_dir, "stats.npy"), stats)
