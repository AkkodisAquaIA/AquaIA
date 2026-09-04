from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics.utils.plotting import Annotator
from dataloading.datasets import sample_dataset
import torch


# Z: internal names --> display names
METRIC_DISPLAY_NAMES = {
    "loss": "Loss",
    "loss_ce": "Classification Loss",
    "loss_bbox": "Box Loss",
    "loss_giou": "GIoU Loss",
    "class_error": "Class Error",
    "cardinality_error": "Cardinality Error",
}

# Z: display order for metrics
METRIC_ORDER = (
    "loss",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "class_error",
    "cardinality_error",
)


def _as_float(value):
    """Z: Convert a value to float if it is a tensor or a numeric type, otherwise return None."""
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    return None


def _flatten_metrics(entry):
    """Z: Flatten a metrics entry into a single dictionary with keys like "train/loss", "val/loss", etc.
    entry = { "train": { "loss": 1.23, "loss_ce": 0.45, "loss_bbox": 0.32, "loss_giou": 0.18, ... },
              "val": { "loss": 1.56, "loss_ce": 0.52, "loss_bbox": 0.41, "loss_giou": 0.22, ... },
              "epoch": 2, }.
    flattened = { "train/loss": 1.23, "train/loss_ce": 0.45, ...}."""
    flattened = {}
    for key, value in entry.items():
        if key == "epoch":
            continue

        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                metric_value = _as_float(nested_value)
                if metric_value is not None:
                    flattened[f"{key}/{nested_key}"] = metric_value
            continue

        # Z: if { "map_50": 0.72, "map_50_95": 0.41, "epoch": 3, }
        metric_value = _as_float(value)
        if metric_value is not None:
            flattened[key] = metric_value

    # Z: flattened = { "train/loss": 1.23, "train/loss_ce": 0.45, ...}
    # Z: or flattened = { "map_50": 0.72, "map_50_95": 0.41, ...}
    return flattened


def _group_metrics_by_name(flattened_history):
    """Z: group by metric name, recording which splits each metric appears in.
    grouped = { "loss": {"train", "val"}, "loss_ce": {"train", "val"}, "map_50": {"map_50"}, ... }. """
    grouped = {}
    # Z: flattened_history = [ { "train/loss": 1.23, "train/loss_ce": 0.45,...}, { "val/loss": 1.56, "val/loss_ce": 0.52,...}, ... ]
    for entry in flattened_history:
        # Z: "train/loss", "train/loss_ce"
        for key in entry:
            if "/" in key:
                # Z: "train", "loss"
                split, metric_name = key.split("/", 1)
            else:
                # Z: "map_50", "map_50"
                split, metric_name = key, key
            # Z: if "grouped" has key [metric_name], add [split] to its set
            # Z: otherwise create a new empty set for [metric_name], then add [split]
            # Z: set handles duplicates, if "train" is already in the set for "loss", it won't be added again
            grouped.setdefault(metric_name, set()).add(split)
    # Z: grouped = { "loss": {"train", "val"}, "loss_ce": {"train", "val"}, "map_50": {"map_50"}, ... }
    return grouped


def _ordered_metric_names(metric_names):
    """Z: order metric names based on METRIC_ORDER, and then sort the rest alphabetically."""
    ordered = [metric_name for metric_name in METRIC_ORDER if metric_name in metric_names]
    ordered.extend(sorted(metric_name for metric_name in metric_names if metric_name not in METRIC_ORDER))
    return ordered


def annotate_images_with_predictions(images, predictions, class_names, output_dir, image_files):
    """Z: Draw model-predicted boxes onto image and save. For training and inference. """
    images = images.detach().cpu().float()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Z: Loop through each image in the batch
    for i in range(images.shape[0]):
        # Z: Convert to numpy HWC
        img = images[i].permute(1, 2, 0).numpy()
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            # Z: Normalize to [0, 1] range
            img = (img - img_min) / (img_max - img_min)
        # Z: Convert to uint8 [0, 255] range
        img_uint8 = (img * 255.0).clip(0, 255).astype("uint8")
        h, w = img_uint8.shape[:2]

        # Z: create annotator, draw on copied image
        annotator = Annotator(img_uint8.copy(), line_width=2)
        image_predictions = predictions[i]
        boxes = image_predictions["boxes"].detach().cpu().float()
        # Z: create a scale tensor with same dtype and device as boxes
        scale = boxes.new_tensor([w, h, w, h])
        # Z: get max box coordinate if predicted boxes exist, otherwise 0
        source_size = float(max(boxes.max().item(), 0.0)) if boxes.numel() else 0.0
        # Z: if predicted boxes are [0, 1] normalized
        if source_size <= 1.0:
            # Z: scale back to original image size
            kept_boxes = boxes * scale
        else:
            kept_boxes = boxes
        kept_scores = image_predictions["scores"].detach().cpu().float()
        kept_labels = image_predictions["labels"].detach().cpu().long()

        for box, score, label in zip(kept_boxes, kept_scores, kept_labels):
            x1, y1, x2, y2 = box.tolist()
            label_idx = int(label)
            # Z: valid label name or just label index if out of range
            label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
            # Z: Draw box and label on the image
            annotator.box_label([x1, y1, x2, y2], label=f"{label_name} {float(score):.2f}")

        output_path = output_dir / f"{Path(image_files[i]).stem}.png"
        plt.imsave(output_path, annotator.result())


@torch.no_grad()
def save_sample_predictions(model, subset, output_dir, predict_fn, num_samples=20, conf=0.3, seed=0, device="cuda"):
    """Z: Sample a subset of images from dataset, run model predictions, and save annotated images.
    For training and inference. """
    samples = sample_dataset(dataset=subset, num_samples=num_samples, seed=seed, device=device)
    print(f"Sampled {len(samples['img_paths'])} images from {subset.dataset_root}")
    model.eval()
    predictions = predict_fn(model=model, samples=samples, device=device, conf_thres=conf)

    annotate_images_with_predictions(
        images=samples["images"],
        predictions=predictions,
        class_names=subset.class_names,
        output_dir=output_dir,
        image_files=samples["img_paths"],
    )


def plot_metrics(run_dir, output_dir=None, metrics_filename="metrics.npy"):
    """Z: read one training run's metrics.npy file, plot the metrics curves and save, return figure path.
    For training."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / metrics_filename
    if not metrics_path.exists():
        return None

    metrics_history = np.load(metrics_path, allow_pickle=True)
    if len(metrics_history) == 0:
        return None

    # Z: metrics_history :
    # [ {"train": {"loss": 1.23, "loss_ce": 0.45,...},
    #      "val": {"loss": 1.56, "loss_ce": 0.52,...},...
    #    "epoch": 1},
    #   {"train": {"loss": 1.10, "loss_ce": 0.40,...},
    #      "val": {"loss": 1.42, "loss_ce": 0.48,...},...
    #    "epoch": 2} ]
    metrics_history = [entry for entry in list(metrics_history) if entry["epoch"] > 1]
    if not metrics_history:
        return None
    epochs = [entry["epoch"] for entry in metrics_history]
    flattened_history = [_flatten_metrics(entry) for entry in metrics_history]
    grouped_metrics = _group_metrics_by_name(flattened_history)
    metric_names = _ordered_metric_names(grouped_metrics)
    if not metric_names:
        return None

    num_cols = 2 if len(metric_names) > 1 else 1
    num_rows = int(np.ceil(len(metric_names) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows), squeeze=False)
    # Z: flatten axes array for easy iteration
    axes = axes.ravel()

    # Z: 1 axe -> 1 metric_name
    for ax, metric_name in zip(axes, metric_names):
        # Z: keep train and val first if they exist for this metric_name in grouped_metrics
        splits = [split for split in ("train", "val") if split in grouped_metrics[metric_name]]
        # Z: append sorted splits other than train and val for this metric_name in grouped_metrics
        splits.extend(sorted(split for split in grouped_metrics[metric_name] if split not in {"train", "val"}))
        # Z: record whether a valid curve has been drawn in the current subplot
        plotted = False
        for split in splits:
            # Z: if split = metric_name, key is a top-level metric like "map_50"; otherwise key is like "train/loss"
            key = metric_name if split == metric_name else f"{split}/{metric_name}"
            # Z: for each epoch get the value corresponding to the current key from the flattened metrics
            values = np.asarray([entry.get(key, np.nan) for entry in flattened_history], dtype=np.float32)
            # Z: if all values are NaN, this curve has no valid data, skip
            if np.all(np.isnan(values)):
                continue
            ax.plot(epochs, values, marker="o", linewidth=1.8, label=split)
            plotted = True

        ax.set_title(METRIC_DISPLAY_NAMES.get(metric_name, metric_name))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()

    # Z: for any remaining axes that don't have a corresponding metric turn off the axis
    for ax in axes[len(metric_names) :]:
        ax.axis("off")

    fig.suptitle("Training Metrics", y=0.995)
    fig.tight_layout()

    output_path = output_dir / "metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
