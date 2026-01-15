import sys
from pathlib import Path
from coco128_cfg import LABELS_FOLDER
current_folder = Path(__file__).resolve().parent

def get_latest_result_dir(base_dir: Path = current_folder) -> Path | None:
    """Return the newest result_det_YYYYMMDDHHmm* directory under base_dir."""
    candidates = [
        p for p in base_dir.glob("result_det_*")
        if p.is_dir() and p.name.replace("result_det_", "").isdigit()]
    if not candidates:
        print("No result_det_YYYYMMDDHHmm* directories found. Exiting.")
        sys.exit(1)
    latest = max(candidates, key=lambda p: p.name.replace("result_det_", ""))
    return latest

latest_dir = get_latest_result_dir()
det_labels_folder = latest_dir / "labels"
gt_labels_folder = Path(LABELS_FOLDER)
