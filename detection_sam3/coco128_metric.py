import sys
from pathlib import Path
import numpy as np
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywh2xyxy
import torch

def get_latest_result_dir(base_dir: Path) -> Path | None:
    """Return the newest sam3_result_det_YYYYMMDDHHmm* folder directory under base_dir."""
    candidates = [
        p for p in base_dir.glob("sam3_result_det_*")
        if p.is_dir() and p.name.replace("sam3_result_det_", "").isdigit()]
    if not candidates:
        print("No sam3_result_det_YYYYMMDDHHmm* directories found. Exiting.")
        sys.exit(1)
    latest = max(candidates, key=lambda p: p.name.replace("sam3_result_det_", ""))
    return latest

def load_label_txt(path: Path, with_conf: bool, conf_threshold: float | None = None):
    """Reads txt label file (GT or detections). Returns clean box coordinates with or without confidence.
    If with_conf is True and conf_threshold is provided, rows with conf < conf_threshold are dropped.
    """
    dim = 6 if with_conf else 5

    # If file does not exist, return empty array
    if not path.exists():
        return np.zeros((0, dim), dtype=np.float32)

    # If file is empty, return empty array
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return np.zeros((0, dim), dtype=np.float32)

    rows = []
    # For each line
    for line in txt.splitlines():
        # Split by whitespace
        parts = line.split()
        # If not enough parts, skip the line
        if len(parts) < dim:
            continue
        rows.append([float(x) for x in parts[:dim]])

    # If file not empty but no valid rows, return empty array
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)

    # Keep only detections above the specified (if provided) confidence threshold 
    row_thr = np.array(rows, dtype=np.float32)
    if with_conf and conf_threshold is not None:
        row_thr = row_thr[row_thr[:, 5] >= conf_threshold]

    # If after thresholding no rows remain, return empty array
    if row_thr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)

    return row_thr

def xywh_norm_to_xyxy_norm(xywhn: np.ndarray) -> np.ndarray:
    """
    Convert normalized [cx,cy,w,h] to normalized [x1,y1,x2,y2], clipped to [0,1].
    Args:
        xywhn: (N,4) normalized [cx,cy,w,h] in [0,1]
    Returns:
        (N,4) normalized xyxy (up left and down right corners) in [0,1]
    Note:
        IoU is invariant to uniform scaling, so calculating IoU with normalized coordinates
    is equivalent to pixel coordinates (no need to read image width/height).
    """
    # If xywhn empty, return empty array
    if xywhn.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Convert to xyxy and clip to [0,1]
    xyxy = xywh2xyxy(xywhn.copy())
    xyxy = np.clip(xyxy, 0.0, 1.0)
    return xyxy.astype(np.float32)

def match_predictions(pred_xyxy, pred_cls, gt_xyxy, gt_cls, iouv):
    """
    Match predictions to GT boxes at different IoU thresholds to get TP matrix.
    Args:
        pred_xyxy: pred box coordinates (Nb pred,4)
        pred_cls: pred classes (Nb pred,)
        gt_xyxy: GT box coordinates (Nb GT,4)
        gt_cls: GT classes (Nb GT,)
        iouv: IoU thresholds list (K,)
    Returns:
        tp: boolean matrix (Nb pred,K), tp[A,B]=True means Ath pred box is TP at Bth threshold value.
    """

    # Create tp output array, of shape (nb pred, nb iou thresholds)
    nb_pred = pred_xyxy.shape[0]
    tp = np.zeros((nb_pred, len(iouv)), dtype=bool)

    # If no pred or no GT, return all-zero (all-false) tp
    if nb_pred == 0 or gt_xyxy.shape[0] == 0:
        return tp

    # Use torch to compute IoU matrix, of shape (nb pred, nb GT)
    pred_t = torch.from_numpy(pred_xyxy).float()
    gt_t = torch.from_numpy(gt_xyxy).float()
    # Each element ious[a,b] = IoU[pred box a, gt box b]
    ious = box_iou(pred_t, gt_t).cpu().numpy()

    # Same class restriction: different classes IoU = 0
    if gt_cls.size and pred_cls.size:
        same_cls = (pred_cls[:, None] == gt_cls[None, :])
        ious = np.where(same_cls, ious, 0.0)

    for thr_idx, thr in enumerate(iouv):
        # Find all candidate elements in ious[pred_i, gt_j] with IoU >= thr
        pred_i, gt_j = np.where(ious >= thr)    # pred_i and gt_j are 1D arrays of indices
        if pred_i.size == 0:
            continue

        # Order candidate elements in ious[pred_i, gt_j] by IoU descending
        order = np.argsort(-ious[pred_i, gt_j])
        pred_i = pred_i[order]
        gt_j = gt_j[order]

        # Create sets to keep track of matched preds and gts, avoid multiple matches
        matched_pred = set()
        matched_gt = set()

        # Global one-to-one matching: prioritize elements in ious[pred_i, gt_j] with the highest IoU
        for p, g in zip(pred_i, gt_j):
            if p in matched_pred or g in matched_gt:
                continue
            tp[p, thr_idx] = True
            matched_pred.add(p)
            matched_gt.add(g)
    return tp

def evaluate_two_folders_intersection(
    gt_labels_folder: str,
    det_labels_folder: str,
    names: dict,
    det_conf_threshold: float | None = None,):
    """
    Evaluate box metrics (precision/recall/mAP50/mAP50-95) in an Ultralytics-like manner
    by comparing predicted labels against GT labels.

    Evaluation set definition:
    - We use the intersection of filenames existing in BOTH gt_labels_folder and det_labels_folder.
    - Reason: the official GT label set is missing annotations for 2 images, so using the full GT set
      would make the evaluation set inconsistent.
    - Predictions are sorted by confidence descending before AP computation to match PR/AP definition.

    Input label formats:
    - GT txt:  cls cx cy w h (normalized)
    - Pred txt: cls cx cy w h conf (normalized)
    """
    # Get paths of GT and detection label folders
    gt_dir = Path(gt_labels_folder)
    det_dir = Path(det_labels_folder)
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT labels folder not found: {gt_dir}")
    if not det_dir.exists():
        raise FileNotFoundError(f"Detection labels folder not found: {det_dir}")

    # Get a sorted list of common txt filenames in both folders: common_names
    gt_files = {p.name: p for p in gt_dir.glob("*.txt")}
    det_files = {p.name: p for p in det_dir.glob("*.txt")}
    common_names = sorted(set(gt_files.keys()) & set(det_files.keys()))
    if not common_names:
        raise RuntimeError(f"No matched txt files between:\n  gt: {gt_dir}\n  det: {det_dir}")

    # IoU thresholds: 0.50:0.95 step 0.05
    iouv = np.linspace(0.5, 0.95, 10)

    # Initialize metrics object to collect tp, conf, pred_cls, target_cls, target_img for each image
    metrics = DetMetrics(names=names)

    # For each image in the evaluation set
    for img_idx, img_name in enumerate(common_names):
        # Get GT and detection txt file paths
        gt_txt = gt_files[img_name]
        det_txt = det_files[img_name]
    
        # Load txt files
        gt = load_label_txt(gt_txt, with_conf=False)
        pr = load_label_txt(det_txt, with_conf=True, conf_threshold=det_conf_threshold)
    
        # Get classes. If no boxes, create empty arrays
        gt_cls = gt[:, 0].astype(int) if gt.shape[0] else np.array([], dtype=int)
        pr_cls = pr[:, 0].astype(int) if pr.shape[0] else np.array([], dtype=int)

        # Convert boxes from xywh normalized to xyxy normalized. If no boxes, create empty arrays
        gt_xyxy = xywh_norm_to_xyxy_norm(gt[:, 1:5]) if gt.shape[0] else np.zeros((0, 4), dtype=np.float32)
        pr_xyxy = xywh_norm_to_xyxy_norm(pr[:, 1:5]) if pr.shape[0] else np.zeros((0, 4), dtype=np.float32)

        # Get pred confidences. If no boxes, create empty array
        pr_conf = pr[:, 5].astype(np.float32) if pr.shape[0] else np.array([], dtype=np.float32)

        # Order pred boxes coordinates, classes, confidences by confidence descending
        if pr_conf.size:
            order = np.argsort(-pr_conf)
            pr_xyxy = pr_xyxy[order]
            pr_cls = pr_cls[order]
            pr_conf = pr_conf[order]

        # Match pred to GT boxes at different IoU thresholds to get TP matrix
        tp = match_predictions(pr_xyxy, pr_cls, gt_xyxy, gt_cls, iouv)

        # Aliment metrics object with results from this image
        metrics.stats["tp"].append(tp.astype(np.bool_))
        metrics.stats["conf"].append(pr_conf.astype(np.float32))
        metrics.stats["pred_cls"].append(pr_cls.astype(np.float32))
        metrics.stats["target_cls"].append(gt_cls.astype(np.float32))
        metrics.stats["target_img"].append(np.full(gt_cls.shape, img_idx, dtype=np.float32))

    # Process metrics and get results dict
    metrics.process(plot=False)
    r = metrics.results_dict

    return {"Nb_images": int(len(common_names)),
            "mAP50-95(B)": float(r["metrics/mAP50-95(B)"]),
            "mAP50(B)": float(r["metrics/mAP50(B)"]),
            "precision(B)": float(r["metrics/precision(B)"]),
            "recall(B)": float(r["metrics/recall(B)"]),}

if __name__ == "__main__":
    from coco128_cfg import IMAGES_FOLDER
    from coco128_dict import COCO128_DICT
    import tkinter as tk
    from tkinter import filedialog

    current_folder = Path(__file__).resolve().parent

    # Ask user to select result_det folder, or use latest if cancelled
    root = tk.Tk()
    root.withdraw()
    chosen_dir = filedialog.askdirectory(
        initialdir=current_folder,
        title="Select result_det folder (Cancel to use latest)") or None
    root.update()
    root.destroy()
    det_dir = Path(chosen_dir) if chosen_dir else get_latest_result_dir(current_folder)

    # Read SAM3_CONF from cfg.txt under the selected/latest result directory
    cfg_conf = None
    cfg_path = det_dir / "cfg.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SAM3_CONF"):
                parts = line.split("=")
                if len(parts) == 2:
                    # Extract confidence value
                    cfg_conf = float(parts[1].strip().strip('"').strip("'"))
                break

    # Ask user for a new confidence threshold
    user_input = input(f"Inference SAM3_CONF = {cfg_conf}, define a new threshold? (blank to skip): ").strip()
    det_conf_threshold = float(user_input) if user_input else None
    suffix = ""
    if det_conf_threshold is not None:
        suffix = f"_conf{int(round(det_conf_threshold * 100)):03d}"

    det_labels_folder = det_dir / "labels"
    gt_labels_folder = Path(str(IMAGES_FOLDER).replace("images", "labels"))

    out = evaluate_two_folders_intersection(
        gt_labels_folder=str(gt_labels_folder),
        det_labels_folder=str(det_labels_folder),
        names=COCO128_DICT,
        det_conf_threshold=det_conf_threshold,)

    # Print each metric on its own line for readability
    keys = ("Nb_images", "mAP50-95(B)", "mAP50(B)", "precision(B)", "recall(B)")
    for key in keys:
        print(f"{key}: {out.get(key)}")

    # Save metrics to txt file
    metrics_txt = "\n".join(f"{key}: {out.get(key)}" for key in keys)
    metrics_path = det_dir / f"metrics{suffix}.txt"
    metrics_path.write_text(metrics_txt + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")