from pathlib import Path

import torch

from detection.benchmark import build_ranking, dependency_ids, parse_training_metrics
from detection.metric import evaluate_map


def test_dependency_ids_include_transitive_initializations():
    experiments = {
        "M0": {"id": "M0", "init": {"type": "pretrained"}},
        "M3": {"id": "M3", "init": {"type": "experiment", "experiment": "M0"}},
        "M4": {"id": "M4", "init": {"type": "experiment", "experiment": "M3"}},
    }
    assert dependency_ids(experiments, {"M4"}) == {"M0", "M3", "M4"}


def test_parse_training_metrics_uses_ultralytics_fitness(tmp_path: Path):
    (tmp_path / "results.csv").write_text(
        "epoch,time,metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,10,0.99,0.50\n"
        "2,20,0.80,0.70\n",
        encoding="utf-8",
    )
    metrics = parse_training_metrics(tmp_path)
    assert metrics["best_epoch"] == 2
    assert metrics["val_map50_95"] == 0.70
    assert metrics["epochs_completed"] == 2
    assert metrics["training_seconds"] == 20


def test_map_and_operating_precision_recall_use_separate_confidence_thresholds():
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [45.0, 45.0, 55.0, 55.0]]),
            "scores": torch.tensor([0.9, 0.01]),
            "labels": torch.tensor([0, 0]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.tensor([0])}]
    metrics = evaluate_map(
        predictions=predictions,
        targets=targets,
        imgsz=100,
        split="test",
        device="cpu",
        pr_conf_thresh=0.05,
    )
    assert metrics["test_map_50"] > 0
    assert metrics["test_precision"] == 0
    assert metrics["test_recall"] == 0


def test_ranking_is_descending_and_limited_to_one_test_dataset():
    rows = [
        {"Modèle": "M1", "Statut": "complete", "Jeu de test": "lpl", "Test mAP50-95": "0.7000"},
        {"Modèle": "M2", "Statut": "complete", "Jeu de test": "aquatic", "Test mAP50-95": "0.9000"},
        {"Modèle": "M3", "Statut": "complete", "Jeu de test": "lpl", "Test mAP50-95": "0.8000"},
    ]
    for row in rows:
        row.update(
            {
                "Initialisation": "M0",
                "Entraînement": "test",
                "Résolution": 640,
                "Test mAP50": "0.9000",
                "Précision": "0.8000",
                "Rappel": "0.8000",
            }
        )
    ranking = build_ranking(rows, ranking_dataset="lpl")
    assert [row["Modèle"] for row in ranking] == ["M3", "M1"]
    assert [row["Rang"] for row in ranking] == [1, 2]
