from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics.utils.plotting import Annotator
from dataloading.datasets import sample_dataset
import torch

def annotate_images_with_predictions(images, predictions, class_names, output_dir, image_files):
    images = images.detach().cpu().float()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0).numpy()
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img_uint8 = (img * 255.0).clip(0, 255).astype("uint8")
        h, w = img_uint8.shape[:2]

        annotator = Annotator(img_uint8.copy(), line_width=2)
        image_predictions = predictions[i]
        boxes = image_predictions["boxes"].detach().cpu().float()
        scale = boxes.new_tensor([w, h, w, h])
        source_size = float(max(boxes.max().item(), 0.0)) if boxes.numel() else 0.0
        if source_size <= 1.0:
            kept_boxes = boxes * scale
        else:
            kept_boxes = boxes
        kept_scores = image_predictions["scores"].detach().cpu().float()
        kept_labels = image_predictions["labels"].detach().cpu().long()

        for box, score, label in zip(kept_boxes, kept_scores, kept_labels):
            x1, y1, x2, y2 = box.tolist()
            label_idx = int(label)
            label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
            annotator.box_label([x1, y1, x2, y2], label=f"{label_name} {float(score):.2f}")

        output_path = output_dir / f"{Path(image_files[i]).stem}.png"
        plt.imsave(output_path, annotator.result())


@torch.no_grad()
def save_sample_predictions(model, subset, output_dir, predict_fn, num_samples=20, conf=0.3, seed=0, device="cuda"):
	samples = sample_dataset(dataset=subset, num_samples=num_samples, seed=seed, device=device)
	print(f"Sampled {len(samples["img_paths"])} images from {subset.dataset_root}")
	model.eval()
	predictions = predict_fn(
		model=model, 
		samples=samples, 
		device=device, 
		conf_thres=conf
	)

	annotate_images_with_predictions(
		images=samples["images"],
		predictions=predictions,
		class_names=subset.class_names,
		output_dir=output_dir,
		image_files=samples["img_paths"],
	)


def plot_metrics(run_dir, output_dir=None, metrics_filename="metrics.npy"):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / metrics_filename
    if not metrics_path.exists():
        return None

    metrics_history = np.load(metrics_path, allow_pickle=True)
    if len(metrics_history) == 0:
        return None

    metrics_history = [entry for entry in list(metrics_history) if entry["epoch"] > 1]
    if not metrics_history:
        return None
    epochs = [entry["epoch"] for entry in metrics_history]
    metric_keys = []
    for key, value in metrics_history[0].items():
        if key == "epoch":
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            metric_keys.append(key)

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in metric_keys:
        values = np.asarray([entry[key] for entry in metrics_history], dtype=np.float32)
        max_value = float(np.max(np.abs(values))) if values.size else 0.0
        if max_value == 0.0 or max_value <= 1.0:
            normalized_values = values
        else:
            normalized_values = values / max_value
        ax.plot(epochs, normalized_values, label=key)

    ax.set_title("Training Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
