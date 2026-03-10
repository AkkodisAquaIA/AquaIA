import sys
from pathlib import Path
import numpy as np
from ultralytics.utils.ops import xywh2xyxy

def get_latest_result_dir(base_dir: Path) -> Path | None:
    """Return the newest [model]_result_det_YYYYMMDDHHmm folder directory under base_dir."""
    candidates = [
        p for p in base_dir.glob("*_result_det_*")
        if p.is_dir() and p.name.split("_result_det_")[-1].isdigit()]
    if not candidates:
        print("No [model]_result_det_YYYYMMDDHHmm directories found. Exiting.")
        sys.exit(1)
    latest = max(candidates, key=lambda p: p.name.split("_result_det_")[-1])
    return latest

def load_label_txt(path: Path, with_conf: bool, conf_threshold: float | None = None):
    """Reads txt label file (GT or detections). Returns clean box coordinates with or without confidence. 
    If with_conf is True and conf_threshold is provided, rows with conf < conf_threshold are dropped.
    Args:
        path: path to txt file
        with_conf: whether to expect confidence column in the txt file
        conf_threshold: if provided keep only rows with confidence >= conf_threshold (if with_conf is True)
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

    # Keep only detections with confidence >= provided conf_threshold (if with_conf is True)
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