"""
Evaluate a SAM3 model on COCO128 for object detection (box-only, PCS text prompts).

Workflow:
- Load COCO128 YAML to gather image/label paths and class names.
- Initialize SAM3 in text-prompt mode (no masks needed) and reuse extracted image
  features for all 80 class prompts in small batches.
- Save predictions in COCO detection JSON plus visualized images.
- Compute YOLO-style metrics: precision(B), recall(B), mAP50(B), mAP50-95(B).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision.ops import nms
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.models.sam import SAM3SemanticPredictor


SAM3_WEIGHTS = Path("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt")
DATASET_YAML = Path("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/AquaIA/detection_sam3/coco128.yaml")
CLASS_BATCH = 8  # number of text prompts per forward pass (reuses image features)
CONF_THRES = 0.25
NMS_IOU = 0.6
DEVICE = 0 if torch.cuda.is_available() else "cpu"
FP16 = torch.cuda.is_available()
NUM_IMAGE: Optional[int] = 5  # set to an int to limit number of images from COCO128


@dataclass
class ImageEntry:
    image_id: int
    path: Path
    width: int
    height: int
    labels: List[Tuple[int, float, float, float, float]]  # cls, x1, y1, x2, y2


def load_dataset(dataset_yaml: Path) -> Tuple[List[ImageEntry], List[str]]:
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", dataset_yaml.parent))
    rel = cfg.get("val", cfg["train"])
    img_dir = root / rel
    names = list(cfg["names"].values())

    entries: List[ImageEntry] = []
    image_paths = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if NUM_IMAGE:
        image_paths = image_paths[:NUM_IMAGE]

    for img_id, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        parts = list(img_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
        label_path = Path(*parts).with_suffix(".txt")

        labels: List[Tuple[int, float, float, float, float]] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                labels.append((int(cls), x1, y1, x2, y2))

        entries.append(ImageEntry(img_id, img_path, w, h, labels))

    return entries, names


def chunk(seq: Sequence[str], size: int) -> List[List[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def iou_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """box: (4,), boxes: (N, 4) in xyxy."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


def classwise_nms(dets: List[Dict[str, float]], iou_thr: float) -> List[Dict[str, float]]:
    final: List[Dict[str, float]] = []
    if not dets:
        return final
    classes = sorted({int(d["category_id"]) for d in dets})
    for cls in classes:
        cls_dets = [d for d in dets if int(d["category_id"]) == cls]
        boxes = torch.tensor([d["bbox_xyxy"] for d in cls_dets], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in cls_dets], dtype=torch.float32)
        keep = nms(boxes, scores, iou_thr)
        for idx in keep.tolist():
            final.append(cls_dets[idx])
    return final


def save_coco_files(
    out_dir: Path,
    images: List[ImageEntry],
    names: List[str],
    detections: List[Dict[str, float]],
) -> Tuple[Path, Path]:
    coco_images = [
        {"id": entry.image_id, "file_name": entry.path.name, "width": entry.width, "height": entry.height}
        for entry in images
    ]
    coco_annotations = []
    ann_id = 1
    for entry in images:
        for cls, x1, y1, x2, y2 in entry.labels:
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": entry.image_id,
                    "category_id": cls,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]
    gt_json = {
        "info": {"description": "COCO128 subset"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    for det in detections:
        det["bbox"] = [
            det["bbox_xyxy"][0],
            det["bbox_xyxy"][1],
            det["bbox_xyxy"][2] - det["bbox_xyxy"][0],
            det["bbox_xyxy"][3] - det["bbox_xyxy"][1],
        ]
        det.pop("bbox_xyxy", None)

    gt_path = out_dir / "coco_gt.json"
    det_path = out_dir / "coco_det.json"
    gt_path.write_text(json.dumps(gt_json, indent=2), encoding="utf-8")
    det_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return gt_path, det_path


def run_coco_eval(gt_json: Path, det_json: Path) -> Dict[str, float]:
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(det_json))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = {
        "coco_AP": float(coco_eval.stats[0]),
        "coco_AP50": float(coco_eval.stats[1]),
        "coco_AR": float(coco_eval.stats[8]),
    }

    # Align with Ultralytics-style outputs at IoU=0.50
    ious = coco_eval.params.iouThrs
    if 0.5 in ious:
        i_idx = int(np.where(np.isclose(ious, 0.5))[0][0])
    else:
        i_idx = 0
    precisions = coco_eval.eval["precision"]  # (T,R,K,A,M)
    recalls = coco_eval.eval["recall"]  # (T,K,A,M)

    def safe_mean(x: np.ndarray) -> Optional[float]:
        x = x[x > -1]
        return float(x.mean()) if x.size else None

    # area=all (index 0), maxDet = last (usually 100)
    p = precisions[i_idx, :, :, 0, -1]
    r = recalls[i_idx, :, 0, -1]
    precision_b = safe_mean(p)
    recall_b = safe_mean(r)
    stats.update(
        precision_B=precision_b if precision_b is not None else 0.0,
        recall_B=recall_b if recall_b is not None else 0.0,
        mAP50_B=float(coco_eval.stats[1]),
        mAP50_95_B=float(coco_eval.stats[0]),
    )
    return stats


def draw_and_save(image_path: Path, dets: List[Dict[str, float]], names: List[str], out_dir: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for det in dets:
        x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
        cls = int(det["category_id"])
        score = det["score"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{names[cls]} {score:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    out_path = out_dir / f"{image_path.stem}_dec.jpg"
    cv2.imwrite(str(out_path), img)


def main() -> None:
    images, names = load_dataset(DATASET_YAML)
    ts = datetime.now().strftime("%Y%m%d%H%M")
    out_dir = Path(__file__).parent / f"result_det_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(
        conf=CONF_THRES,
        task="segment",
        mode="predict",
        model=str(SAM3_WEIGHTS),
        half=FP16,
        device=DEVICE,
        save=False,
        verbose=False,
    )
    predictor_feat = SAM3SemanticPredictor(overrides=overrides)
    predictor_query = SAM3SemanticPredictor(overrides=overrides)

    detections: List[Dict[str, float]] = []

    for entry in images:
        predictor_feat.set_image(str(entry.path))
        src_shape = (entry.height, entry.width)
        predictor_query.setup_model()

        img_dets: List[Dict[str, float]] = []
        for batch_idx, batch in enumerate(chunk(names, CLASS_BATCH)):
            masks, boxes = predictor_query.inference_features(
                predictor_feat.features, src_shape=src_shape, text=batch
            )
            if boxes is None:
                continue
            boxes_np = boxes.detach().cpu().numpy()
            scores = (
                boxes_np[:, 4]
                if boxes_np.shape[1] > 4
                else np.full((boxes_np.shape[0],), 1.0, dtype=np.float32)
            )
            xyxy = boxes_np[:, :4]

            for i, b in enumerate(xyxy):
                cls_global = batch_idx * CLASS_BATCH + min(i, len(batch) - 1)
                img_dets.append(
                    dict(
                        image_id=entry.image_id,
                        category_id=cls_global,
                        score=float(scores[i]),
                        bbox_xyxy=b.tolist(),
                    )
                )

        img_dets = classwise_nms(img_dets, NMS_IOU)
        detections.extend(img_dets)
        draw_and_save(entry.path, img_dets, names, out_dir)

    gt_json, det_json = save_coco_files(out_dir, images, names, detections.copy())
    summary = run_coco_eval(gt_json, det_json)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Metrics:", json.dumps(summary, indent=2))
    print(f"COCO files saved to: {gt_json} and {det_json}")
    print(f"Visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
