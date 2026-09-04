import json
from datetime import datetime
from pathlib import Path


def _registry_path(config: dict) -> Path:
    registry = config.get("logging", {}).get("registry_path", "runs/registry.jsonl")
    return Path(registry)


def register_run(config: dict, run_id: str, run_dir: str, pid: int) -> None:
    """Z: register a new run in the registry.jsonl file."""
    path = _registry_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    entry = {
        "run_id": run_id,
        "model": f"{model_cfg.get('family', '')}_{model_cfg.get('size', '')}",
        "dataset": str(data_cfg.get("dataset_yaml", "")),
        "epochs": training_cfg.get("epochs", 0),
        "status": "running",
        "run_dir": str(run_dir),
        "started_at": datetime.utcnow().isoformat(),
        "pid": pid,
    }
    with path.open("a", encoding="utf-8") as f:
        # Z: json.dumps converts the entry dictionary to a JSON string
        f.write(json.dumps(entry) + "\n")
        f.flush()


def update_run_status(config: dict, run_id: str, status: str) -> None:
    """Z: update the status of a run in the registry.jsonl file."""
    path = _registry_path(config)
    if not path.exists():
        return

    # Z: read registry.jsonl, split into lines
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = []
    # Z: each line represents a run
    for line in lines:
        if not line.strip():
            continue
        try:
            # Z: parse json line into python object
            entry = json.loads(line)
            if entry.get("run_id") == run_id:
                entry["status"] = status
                entry["finished_at"] = datetime.utcnow().isoformat()
            # Z: json.dumps converts the entry dictionary to a JSON string
            updated.append(json.dumps(entry))
        except json.JSONDecodeError:
            updated.append(line)

    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
