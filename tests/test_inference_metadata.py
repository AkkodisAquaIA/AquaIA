from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from detection.inference_context import save_inference_metadata


def test_save_inference_metadata_records_checkpoint_and_configs(tmp_path: Path) -> None:
    run_dir = tmp_path / "training_run"
    output_dir = tmp_path / "inference_output"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    output_dir.mkdir()

    checkpoint_path = weights_dir / "best.pt"
    checkpoint_bytes = b"exact checkpoint used for inference"
    checkpoint_path.write_bytes(checkpoint_bytes)
    (run_dir / "args.yaml").write_text("epochs: 300\nimgsz: 640\n", encoding="utf-8")

    inference_config = {
        "run": {"checkpoint": "weights/best.pt"},
        "data": {"infer_data_root": "datasets/example", "split": "test"},
        "inference": {"imgsz": 640, "batch": 16},
    }
    training_config = {
        "model": {"family": "yolo11", "size": "n", "init": "pretrained"},
        "data": {"dataset_yaml": "datasets/train"},
        "training": {"epochs": 300, "imgsz": 640},
    }
    context = {
        "run_dir": run_dir,
        "run_config": training_config,
        "checkpoint_path": checkpoint_path,
        "infer_data_root": "datasets/example",
        "output_dir": output_dir,
        "device": "cpu",
        "use_amp": False,
    }

    metadata_path = save_inference_metadata(
        inference_config,
        context,
        config_path=tmp_path / "infer_config.yaml",
    )
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path == output_dir / "inference_metadata.yaml"
    assert metadata["model"]["configuration"] == training_config["model"]
    assert metadata["model"]["checkpoint"]["name"] == "best.pt"
    assert metadata["model"]["checkpoint"]["sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert metadata["training"]["resolved_config"] == training_config
    assert metadata["training"]["ultralytics_args"] == {"epochs": 300, "imgsz": 640}
    assert metadata["inference"]["config"] == inference_config
    assert metadata["inference"]["split"] == "test"
