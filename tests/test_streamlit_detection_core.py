from pathlib import Path

import numpy as np

from deploy.streamlit_detection.core import filter_models, image_detection_metrics, parse_yolo_labels, summarize_training_run


def test_filter_models_uses_all_experiment_choices():
    models = [
        {
            "id": "baseline",
            "architecture": "yolo11n",
            "imgsz": 640,
            "roboflow_prefinetuning": False,
            "lpl_finetuning": True,
            "yolo_augmentation": False,
            "copy_paste": False,
        },
        {
            "id": "copy-paste",
            "architecture": "yolo11n",
            "imgsz": 640,
            "roboflow_prefinetuning": False,
            "lpl_finetuning": True,
            "yolo_augmentation": False,
            "copy_paste": True,
        },
    ]
    choices = {
        "architecture": "yolo11n",
        "imgsz": 640,
        "roboflow_prefinetuning": False,
        "lpl_finetuning": True,
        "yolo_augmentation": False,
        "copy_paste": True,
    }
    assert [model["id"] for model in filter_models(models, choices)] == ["copy-paste"]


def test_parse_labels_and_compute_single_image_metrics():
    target_labels, target_boxes = parse_yolo_labels("0 0.5 0.5 0.4 0.2", 100, 200)
    np.testing.assert_allclose(target_boxes, [[30, 80, 70, 120]])

    metrics = image_detection_metrics(
        predicted_labels=np.array([0, 0]),
        predicted_boxes=np.array([[30, 80, 70, 120], [0, 0, 10, 10]], dtype=np.float32),
        predicted_scores=np.array([0.9, 0.8]),
        target_labels=target_labels,
        target_boxes=target_boxes,
        iou_threshold=0.5,
    )
    assert metrics["Vrais positifs"] == 1
    assert metrics["Faux positifs"] == 1
    assert metrics["Objets manqués"] == 0
    assert metrics["Précision image"] == 0.5
    assert metrics["Rappel image"] == 1.0


def test_training_summary_selects_best_map_50_95(tmp_path: Path):
    (tmp_path / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.5,0.6,0.7,0.4\n"
        "2,0.8,0.9,0.95,0.75\n",
        encoding="utf-8",
    )
    summary = summarize_training_run(tmp_path)
    assert summary["Meilleure époque"] == 2
    assert summary["Val mAP50-95"] == 0.75
