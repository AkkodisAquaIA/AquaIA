from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics.utils.plotting import Annotator

from Detection.utils.box_ops import box_cxcywh_to_xyxy


def annotate_images_with_predictions(images, outputs, class_names, conf_thres, output_dir, image_files):
    images = images.detach().cpu().float()
    pred_boxes = outputs["pred_boxes"].detach().cpu()
    pred_logits = outputs["pred_logits"].detach().cpu()
    class_logits = pred_logits[..., : len(class_names)]
    scores, labels = class_logits.sigmoid().max(dim=-1)

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
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i]).clamp(0, 1)
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h

        keep = scores[i] >= conf_thres
        kept_boxes = boxes_xyxy[keep]
        kept_scores = scores[i][keep]
        kept_labels = labels[i][keep]

        for box, score, label in zip(kept_boxes, kept_scores, kept_labels):
            x1, y1, x2, y2 = box.tolist()
            label_idx = int(label)
            label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
            annotator.box_label([x1, y1, x2, y2], label=f"{label_name} {float(score):.2f}")

        output_path = output_dir / f"{Path(image_files[i]).stem}.png"
        plt.imsave(output_path, annotator.result())


def annotate_yolo_predictions(results, class_names, conf_thres, output_dir, image_files):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result, image_file in zip(results, image_files):
        annotator = Annotator(result.orig_img.copy(), line_width=2)
        boxes = result.boxes
        if boxes is not None:
            for box, score, label in zip(boxes.xyxy, boxes.conf, boxes.cls):
                if float(score) < conf_thres:
                    continue
                label_idx = int(label)
                label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
                annotator.box_label(box.tolist(), label=f"{label_name} {float(score):.2f}")

        output_path = output_dir / f"{Path(image_file).stem}.png"
        Image.fromarray(annotator.result()).save(output_path)


def plot_metrics(run_dir, metrics_filename="metrics.npy"):
    run_dir = Path(run_dir)
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
        normalized_values = values if max_value == 0.0 else values / max_value
        ax.plot(epochs, normalized_values, label=key)

    ax.set_title("Training Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = run_dir / "metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
