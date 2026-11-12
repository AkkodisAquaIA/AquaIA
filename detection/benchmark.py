from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_COLUMNS = (
    "Modèle",
    "Initialisation",
    "Entraînement",
    "Résolution",
    "Jeu de test",
    "Val mAP50",
    "Val mAP50-95",
    "Test mAP50",
    "Test mAP50-95",
    "Précision",
    "Rappel",
    "Meilleure époque",
    "Époques réalisées",
    "Temps",
    "Statut",
)
RANKING_COLUMNS = (
    "Rang",
    "Modèle",
    "Initialisation",
    "Entraînement",
    "Résolution",
    "Jeu de test",
    "Test mAP50-95",
    "Test mAP50",
    "Précision",
    "Rappel",
)


def deep_merge(base: dict, override: dict | None) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(payload, yaml_file, sort_keys=False, allow_unicode=True)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_suite(config: dict) -> None:
    required = {"benchmark", "model", "datasets", "training_defaults", "inference_defaults", "augmentation_profiles", "experiments"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Sections manquantes dans le benchmark : {sorted(missing)}")

    experiment_ids = [experiment["id"] for experiment in config["experiments"]]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Chaque expérience doit avoir un identifiant unique.")

    known_ids = set(experiment_ids)
    for experiment in config["experiments"]:
        init = experiment.get("init", {})
        if init.get("type") == "experiment" and init.get("experiment") not in known_ids:
            raise ValueError(f"{experiment['id']} dépend d'une expérience inconnue : {init.get('experiment')}")
        train = experiment.get("train")
        if train:
            if train.get("dataset") not in config["datasets"]:
                raise ValueError(f"Dataset d'entraînement inconnu pour {experiment['id']} : {train.get('dataset')}")
            if train.get("augmentation") not in config["augmentation_profiles"]:
                raise ValueError(f"Profil d'augmentation inconnu pour {experiment['id']} : {train.get('augmentation')}")
        evaluation = experiment.get("evaluate", {})
        if evaluation.get("dataset") not in config["datasets"]:
            raise ValueError(f"Dataset d'évaluation inconnu pour {experiment['id']} : {evaluation.get('dataset')}")


def dependency_ids(experiments: dict[str, dict], selected: set[str]) -> set[str]:
    expanded = set()

    def include(experiment_id: str) -> None:
        if experiment_id in expanded:
            return
        if experiment_id not in experiments:
            raise ValueError(f"Expérience inconnue : {experiment_id}")
        init = experiments[experiment_id].get("init", {})
        if init.get("type") == "experiment":
            include(init["experiment"])
        expanded.add(experiment_id)

    for experiment_id in selected:
        include(experiment_id)
    return expanded


def experiment_checkpoint(experiment: dict, completed: dict, model_defaults: dict, allow_missing: bool = False) -> Path:
    init = experiment.get("init", {})
    init_type = init.get("type", "pretrained")
    if init_type == "pretrained":
        checkpoint = init.get("checkpoint") or model_defaults.get("pretrained_checkpoint")
        if not checkpoint:
            raise ValueError(f"Aucun checkpoint préentraîné défini pour {experiment['id']}")
        checkpoint_path = resolve_path(checkpoint)
    elif init_type == "checkpoint":
        checkpoint_path = resolve_path(init["checkpoint"])
    elif init_type == "experiment":
        dependency = init["experiment"]
        if dependency not in completed:
            raise RuntimeError(f"La dépendance {dependency} de {experiment['id']} n'est pas terminée.")
        checkpoint_path = Path(completed[dependency]["checkpoint"])
    else:
        raise ValueError(f"Type d'initialisation inconnu pour {experiment['id']} : {init_type}")
    if not allow_missing and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    return checkpoint_path.resolve()


def build_train_config(config: dict, experiment: dict, checkpoint: Path, suite_dir: Path) -> dict:
    train = experiment["train"]
    dataset = config["datasets"][train["dataset"]]
    profile = config["augmentation_profiles"][train["augmentation"]]
    training = deep_merge(config["training_defaults"], profile.get("training"))
    training = deep_merge(training, train.get("overrides"))
    if "imgsz" not in (train.get("overrides") or {}):
        training["imgsz"] = int(experiment.get("imgsz", training.get("imgsz", 640)))

    train_config = {
        "model": {
            "family": config["model"]["family"],
            "size": config["model"].get("size", "n"),
            "init": "checkpoint",
            "checkpoint": str(checkpoint),
        },
        "data": {"dataset_yaml": str(resolve_path(dataset["root"]))},
        "output": {
            "project": str(suite_dir / "training" / experiment["id"]),
            "exist_ok": True,
            "save_period": 0,
            "plots": True,
        },
        "training": training,
    }
    copy_paste = profile.get("copy_paste_data")
    if copy_paste:
        train_config["copy_paste_data"] = copy.deepcopy(copy_paste)
    return train_config


def create_reference_run(experiment: dict, checkpoint: Path, dataset_root: Path, suite_dir: Path, model_defaults: dict) -> Path:
    run_dir = suite_dir / "references" / experiment["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(
        run_dir / "resolved_config.yaml",
        {
            "model": {
                "family": model_defaults["family"],
                "size": model_defaults.get("size", "n"),
                "init": "checkpoint",
                "checkpoint": str(checkpoint),
            },
            "data": {"dataset_yaml": str(dataset_root)},
            "training": {"task": "detect", "device": "auto"},
            "output": {"run_dir": str(run_dir)},
        },
    )
    return run_dir


def build_infer_config(config: dict, experiment: dict, checkpoint: Path, run_dir: Path, suite_dir: Path) -> dict:
    evaluation = experiment["evaluate"]
    dataset = config["datasets"][evaluation["dataset"]]
    inference = deep_merge(config["inference_defaults"], evaluation.get("overrides"))
    if "imgsz" not in (evaluation.get("overrides") or {}):
        inference["imgsz"] = int(experiment.get("imgsz", inference.get("imgsz", 640)))
    return {
        "run": {
            "runs_root": str(run_dir.parent),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
        },
        "data": {
            "infer_data_root": str(resolve_path(dataset["root"])),
            "split": evaluation.get("split", "test"),
            "img_format": dataset.get("img_format", "jpg"),
        },
        "inference": inference,
        "output": {"output_dir": str(suite_dir / "evaluation" / experiment["id"])},
    }


def newest_artifact(root: Path, marker: str, created_after: float | None = None) -> Path:
    candidates = list(root.glob(f"**/{marker}"))
    if created_after is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= created_after - 1]
    if not candidates:
        raise FileNotFoundError(f"Aucun artefact {marker} trouvé sous {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_command(arguments: list[str]) -> None:
    printable = " ".join(arguments)
    print(f"\n$ {printable}\n", flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def ensure_dataset_stats(dataset_root: Path, imgsz: int) -> Path:
    stats_path = dataset_root / f"stats_{imgsz}.npy"
    if stats_path.is_file():
        return stats_path
    print(f"Statistiques absentes, génération de {stats_path}...", flush=True)
    from data_processing.stats import compute_and_save_stats

    compute_and_save_stats(dataset_name=dataset_root, image_size=imgsz)
    if not stats_path.is_file():
        raise FileNotFoundError(f"La génération des statistiques a échoué : {stats_path}")
    return stats_path


def parse_training_metrics(run_dir: Path | None) -> dict:
    empty = {
        "val_map50": None,
        "val_map50_95": None,
        "best_epoch": None,
        "epochs_completed": None,
        "training_seconds": None,
    }
    if run_dir is None:
        return empty
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return empty
    with results_path.open("r", encoding="utf-8", newline="") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        return empty

    def value(row: dict, key: str) -> float:
        return float(row[key].strip())

    # The detection checkpoint best.pt is selected from validation mAP50-95.
    best = max(rows, key=lambda row: value(row, "metrics/mAP50-95(B)"))
    return {
        "val_map50": value(best, "metrics/mAP50(B)"),
        "val_map50_95": value(best, "metrics/mAP50-95(B)"),
        "best_epoch": int(value(best, "epoch")),
        "epochs_completed": int(value(rows[-1], "epoch")),
        "training_seconds": value(rows[-1], "time"),
    }


def parse_inference_metrics(inference_dir: Path | None) -> dict:
    empty = {"test_map50": None, "test_map50_95": None, "precision": None, "recall": None}
    if inference_dir is None:
        return empty
    metrics_path = inference_dir / "inference_metrics.yaml"
    if not metrics_path.is_file():
        return empty
    metrics = load_yaml(metrics_path)
    return {
        "test_map50": metrics.get("test_map_50"),
        "test_map50_95": metrics.get("test_map_50_95"),
        "precision": metrics.get("test_precision"),
        "recall": metrics.get("test_recall"),
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{float(value):.4f}"


def report_row(experiment: dict, record: dict) -> dict:
    training = parse_training_metrics(Path(record["run_dir"])) if record.get("run_dir") else parse_training_metrics(None)
    inference = parse_inference_metrics(Path(record["inference_dir"])) if record.get("inference_dir") else parse_inference_metrics(None)
    return {
        "Modèle": experiment["id"],
        "Initialisation": experiment.get("initialization_label", ""),
        "Entraînement": experiment.get("training_label", ""),
        "Résolution": experiment.get("imgsz", ""),
        "Jeu de test": experiment.get("evaluate", {}).get("dataset", ""),
        "Val mAP50": format_metric(training["val_map50"]),
        "Val mAP50-95": format_metric(training["val_map50_95"]),
        "Test mAP50": format_metric(inference["test_map50"]),
        "Test mAP50-95": format_metric(inference["test_map50_95"]),
        "Précision": format_metric(inference["precision"]),
        "Rappel": format_metric(inference["recall"]),
        "Meilleure époque": training["best_epoch"] if training["best_epoch"] is not None else "",
        "Époques réalisées": training["epochs_completed"] if training["epochs_completed"] is not None else "",
        "Temps": format_duration(training["training_seconds"]),
        "Statut": record.get("status", "non exécuté"),
    }


def build_ranking(rows: list[dict], ranking_dataset: str | None = None) -> list[dict]:
    eligible = [
        row
        for row in rows
        if row["Statut"] == "complete"
        and row["Test mAP50-95"] != ""
        and (ranking_dataset is None or row["Jeu de test"] == ranking_dataset)
    ]
    eligible.sort(key=lambda row: float(row["Test mAP50-95"]), reverse=True)
    return [
        {
            "Rang": rank,
            **{column: row[column] for column in RANKING_COLUMNS if column != "Rang"},
        }
        for rank, row in enumerate(eligible, start=1)
    ]


def write_reports(
    suite_dir: Path,
    experiments: list[dict],
    state: dict,
    ranking_dataset: str | None = None,
) -> list[dict]:
    rows = [
        report_row(experiment, state.get("experiments", {}).get(experiment["id"], {}))
        for experiment in experiments
    ]
    csv_path = suite_dir / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = suite_dir / "benchmark_results.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, ensure_ascii=False, indent=2)

    markdown_path = suite_dir / "benchmark_results.md"
    with markdown_path.open("w", encoding="utf-8") as markdown_file:
        markdown_file.write("| " + " | ".join(REPORT_COLUMNS) + " |\n")
        markdown_file.write("| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |\n")
        for row in rows:
            markdown_file.write("| " + " | ".join(str(row[column]) for column in REPORT_COLUMNS) + " |\n")

    ranking = build_ranking(rows, ranking_dataset=ranking_dataset)
    with (suite_dir / "benchmark_ranking.csv").open("w", encoding="utf-8", newline="") as ranking_file:
        writer = csv.DictWriter(ranking_file, fieldnames=RANKING_COLUMNS)
        writer.writeheader()
        writer.writerows(ranking)
    with (suite_dir / "benchmark_ranking.json").open("w", encoding="utf-8") as ranking_file:
        json.dump(ranking, ranking_file, ensure_ascii=False, indent=2)
    with (suite_dir / "benchmark_ranking.md").open("w", encoding="utf-8") as ranking_file:
        ranking_file.write("| " + " | ".join(RANKING_COLUMNS) + " |\n")
        ranking_file.write("| " + " | ".join("---" for _ in RANKING_COLUMNS) + " |\n")
        for row in ranking:
            ranking_file.write("| " + " | ".join(str(row[column]) for column in RANKING_COLUMNS) + " |\n")
    return rows


def run_benchmark(config_path: str | Path, only: list[str] | None = None, dry_run: bool = False, force: bool = False) -> Path:
    config_path = resolve_path(config_path)
    config = load_yaml(config_path)
    validate_suite(config)

    suite_dir = resolve_path(config["benchmark"]["output_dir"])
    suite_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = suite_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    state_path = suite_dir / "benchmark_state.yaml"
    state = load_yaml(state_path) if state_path.is_file() else {"experiments": {}}
    state.setdefault("experiments", {})
    state["config_path"] = str(config_path)
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    all_enabled = [experiment for experiment in config["experiments"] if experiment.get("enabled", True)]
    experiments = {experiment["id"]: experiment for experiment in all_enabled}
    selected = set(only or experiments.keys())
    selected = dependency_ids(experiments, selected)
    ordered = [experiment for experiment in config["experiments"] if experiment["id"] in selected and experiment.get("enabled", True)]

    completed = {}
    for experiment in ordered:
        experiment_id = experiment["id"]
        fingerprint = stable_hash(
            {
                "model": config["model"],
                "datasets": config["datasets"],
                "training_defaults": config["training_defaults"],
                "inference_defaults": config["inference_defaults"],
                "augmentation_profiles": config["augmentation_profiles"],
                "experiment": experiment,
            }
        )
        previous = state["experiments"].get(experiment_id, {})
        reusable = (
            not force
            and previous.get("status") == "complete"
            and previous.get("fingerprint") == fingerprint
            and Path(previous.get("checkpoint", "")).is_file()
            and Path(previous.get("inference_dir", "")).joinpath("inference_metrics.yaml").is_file()
        )
        if reusable:
            print(f"[{experiment_id}] Résultat existant réutilisé.")
            completed[experiment_id] = previous
            continue

        # A training may be complete even if its following inference failed
        # (for example because visualization exhausted GPU memory). Reuse that
        # run and retry only inference instead of paying for another training.
        previous_run_dir = Path(previous.get("run_dir", ""))
        previous_checkpoint = Path(previous.get("checkpoint", ""))
        recover_completed_training = (
            not force
            and previous.get("fingerprint") == fingerprint
            and previous_checkpoint.is_file()
            and previous_run_dir.joinpath("results.csv").is_file()
        )

        print(f"\n[{experiment_id}] {experiment.get('training_label', '')}")
        checkpoint = experiment_checkpoint(experiment, completed, config["model"], allow_missing=dry_run)
        evaluation_dataset = config["datasets"][experiment["evaluate"]["dataset"]]
        run_dir = None
        train_config_path = None
        reuse = experiment.get("reuse")

        if recover_completed_training:
            checkpoint = previous_checkpoint.resolve()
            run_dir = previous_run_dir.resolve()
            previous_train_config = previous.get("train_config")
            train_config_path = Path(previous_train_config) if previous_train_config else None
            print(f"  entraînement déjà terminé, reprise de l'inférence : {run_dir}")
        elif reuse:
            checkpoint = resolve_path(reuse["checkpoint"])
            run_dir = resolve_path(reuse.get("run_dir") or checkpoint.parent.parent)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint réutilisé introuvable pour {experiment_id} : {checkpoint}")
            if not (run_dir / "results.csv").is_file():
                raise FileNotFoundError(f"Run réutilisé sans results.csv pour {experiment_id} : {run_dir}")
            print(f"  entraînement réutilisé : {run_dir}")
        elif experiment.get("train"):
            train_config = build_train_config(config, experiment, checkpoint, suite_dir)
            train_config_path = generated_dir / f"{experiment_id}_train.yaml"
            save_yaml(train_config_path, train_config)
            if not dry_run:
                started_at = time.time()
                try:
                    run_command([sys.executable, "main.py", "train", "--config", str(train_config_path)])
                except Exception as error:
                    failed_record = {
                        "status": "failed",
                        "fingerprint": fingerprint,
                        "checkpoint": str(checkpoint),
                        "run_dir": "",
                        "inference_dir": "",
                        "train_config": str(train_config_path),
                        "error": str(error),
                    }
                    state["experiments"][experiment_id] = failed_record
                    save_yaml(state_path, state)
                    write_reports(
                        suite_dir,
                        all_enabled,
                        state,
                        ranking_dataset=config["benchmark"].get("ranking_dataset"),
                    )
                    if config["benchmark"].get("continue_on_error", False):
                        continue
                    raise
                run_dir = newest_artifact(Path(train_config["output"]["project"]), "weights/best.pt", created_after=started_at).parent.parent
                checkpoint = run_dir / "weights" / "best.pt"
            else:
                run_dir = Path(train_config["output"]["project"]) / "<run_timestamp>"
                checkpoint = run_dir / "weights" / "best.pt"
        else:
            run_dir = create_reference_run(experiment, checkpoint, resolve_path(evaluation_dataset["root"]), suite_dir, config["model"])

        infer_config = build_infer_config(config, experiment, checkpoint, run_dir, suite_dir)
        infer_config_path = generated_dir / f"{experiment_id}_infer.yaml"
        save_yaml(infer_config_path, infer_config)

        if dry_run:
            print(f"  train: {train_config_path or 'aucun'}")
            print(f"  infer: {infer_config_path}")
            completed[experiment_id] = {
                "status": "planned",
                "checkpoint": str(checkpoint),
                "run_dir": str(run_dir),
                "inference_dir": "",
                "fingerprint": fingerprint,
            }
            continue

        if config["benchmark"].get("auto_compute_stats", True):
            ensure_dataset_stats(
                resolve_path(infer_config["data"]["infer_data_root"]),
                int(infer_config["inference"]["imgsz"]),
            )

        record = {
            "status": "running",
            "fingerprint": fingerprint,
            "checkpoint": str(checkpoint.resolve()),
            "run_dir": str(run_dir.resolve()),
            "train_config": str(train_config_path) if train_config_path else None,
            "infer_config": str(infer_config_path),
            "reused_training": bool(reuse or recover_completed_training),
        }
        state["experiments"][experiment_id] = record
        save_yaml(state_path, state)
        try:
            started_at = time.time()
            run_command([sys.executable, "main.py", "infer", "--config", str(infer_config_path)])
            inference_dir = newest_artifact(Path(infer_config["output"]["output_dir"]), "inference_metrics.yaml", created_after=started_at).parent
            record["inference_dir"] = str(inference_dir.resolve())
            record["status"] = "complete"
            record["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            completed[experiment_id] = record
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            if not config["benchmark"].get("continue_on_error", False):
                raise
        finally:
            state["experiments"][experiment_id] = record
            state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            save_yaml(state_path, state)
            write_reports(
                suite_dir,
                all_enabled,
                state,
                ranking_dataset=config["benchmark"].get("ranking_dataset"),
            )

    if not dry_run:
        rows = write_reports(
            suite_dir,
            all_enabled,
            state,
            ranking_dataset=config["benchmark"].get("ranking_dataset"),
        )
        print(f"\nTableaux disponibles sous : {suite_dir}")
        ranking = build_ranking(rows, ranking_dataset=config["benchmark"].get("ranking_dataset"))
        if ranking:
            print("Classement par Test mAP50-95 :")
            for row in ranking:
                print(f"  {row['Rang']}. {row['Modèle']} — {row['Test mAP50-95']}")
    return suite_dir


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Grand fichier YAML décrivant la suite d'expériences.")
    parser.add_argument("--only", nargs="+", help="Limiter l'exécution à certains modèles, dépendances incluses.")
    parser.add_argument("--dry-run", action="store_true", help="Valider et générer les YAML sans lancer de calcul.")
    parser.add_argument("--force", action="store_true", help="Rejouer les expériences même si elles sont déjà terminées.")


def handle_benchmark(args) -> Path:
    return run_benchmark(args.config, only=args.only, dry_run=args.dry_run, force=args.force)
