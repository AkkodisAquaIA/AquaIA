"""CLI: python -m detection.list_runs"""
import json
import sys
from pathlib import Path

REGISTRY_PATH = Path("runs/registry.jsonl")

_COLOR = {
    "running":     "\033[33m",
    "done":        "\033[32m",
    "error":       "\033[31m",
    "interrupted": "\033[35m",
}
_RESET = "\033[0m"


def main() -> None:
    if not REGISTRY_PATH.exists():
        print(f"No registry found at {REGISTRY_PATH}. No runs have been recorded yet.")
        sys.exit(0)

    lines = [l.strip() for l in REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print("Registry is empty.")
        sys.exit(0)

    runs = []
    for line in lines:
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Most recent first
    runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)

    header = f"{'run_id':<22} {'model':<20} {'dataset':<28} {'ep':>4}  {'status':<14} {'started_at'}"
    print(header)
    print("─" * 100)

    for r in runs:
        status = r.get("status", "?")
        color = _COLOR.get(status, "")
        started = r.get("started_at", "")[:16].replace("T", " ")
        dataset = Path(r.get("dataset", "")).name[:27]
        status_col = f"{color}{status:<12}{_RESET}"
        print(
            f"{r.get('run_id', '?'):<22} "
            f"{r.get('model', '?'):<20} "
            f"{dataset:<28} "
            f"{r.get('epochs', '?'):>4}  "
            f"{status_col}  "
            f"{started}"
        )


if __name__ == "__main__":
    main()
