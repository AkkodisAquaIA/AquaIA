"""
Test script for the training logs system — no GPU, no dataset, no torch required.

Usage:
    python -m detection.test_training_logs          # full run (5 epochs)
    python -m detection.test_training_logs --crash  # Ctrl+C after epoch 2 to test interrupt
    python -m detection.test_training_logs --list   # show registry after a run

What is tested:
    - train.jsonl  : one line per epoch, flushed immediately
    - train.log    : human-readable log file
    - heartbeat    : updated every 2 batches (simulated)
    - run_meta.json: status = running → done / interrupted / error
    - registry     : one entry added per run
    - list_runs    : display of the registry
"""

import argparse
import json
import math
import pathlib
import random
import tempfile
import time


def _fake_loss(epoch: int, split: str) -> float:
    """Simulated loss that decreases over epochs."""
    base = 2.0 if split == "train" else 1.8
    return round(base * math.exp(-0.3 * epoch) + random.uniform(-0.05, 0.05), 4)


def run_test(crash_at=None, use_tmp: bool = False) -> pathlib.Path:
    from detection.logging import TrainingLogger, CheckpointManager, register_run, update_run_status

    config = {
        "model": {"family": "test_model", "size": "nano", "init": "pretrained"},
        "training": {"epochs": 5, "batch": 4, "lr0": 0.001},
        "logging": {
            "save_period": 2,
            "heartbeat_every_n_batches": 2,
            "registry_path": "runs/registry.jsonl",
        },
        "data": {"dataset_yaml": "datasets/test_dataset"},
    }

    run_dir = pathlib.Path("runs") / time.strftime("%Y%m%d_%H%M%S_test")
    if use_tmp:
        run_dir = pathlib.Path(tempfile.mkdtemp())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(exist_ok=True)

    run_id = run_dir.name
    logger = TrainingLogger(run_dir=str(run_dir), run_id=run_id, config=config)
    register_run(config=config, run_id=run_id, run_dir=str(run_dir), pid=__import__("os").getpid())

    logger.log_device("cpu", False, "test_dataset (train=100 val=20)")
    logger.info("Starting simulated training loop (no real model)")

    best_val = float("inf")

    try:
        for epoch in range(1, config["training"]["epochs"] + 1):
            logger.info(f"========== Epoch {epoch}/{config['training']['epochs']} ==========")

            # Simulate batches
            n_batches = 10
            for batch_idx in range(1, n_batches + 1):
                time.sleep(0.05)  # ~50ms per batch
                logger.heartbeat(epoch, batch_idx, n_batches)

                # Simulate crash via Ctrl+C at requested epoch
                if crash_at and epoch == crash_at and batch_idx == 5:
                    print(f"\n[TEST] Simulating Ctrl+C at epoch {epoch} batch {batch_idx}")
                    raise KeyboardInterrupt

            train_loss = _fake_loss(epoch, "train")
            val_loss = _fake_loss(epoch, "val")
            metric_dict = {"train": {"loss": train_loss}, "val": {"loss": val_loss}}
            lr = config["training"]["lr0"] * (0.95**epoch)

            logger.log_epoch(epoch, config["training"]["epochs"], metric_dict, lr)

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                logger.log_best(epoch, val_loss)

        logger.finish()
        update_run_status(config, run_id, "done")

    except KeyboardInterrupt:
        logger.interrupted()
        update_run_status(config, run_id, "interrupted")

    except Exception as exc:
        logger.crash(str(exc))
        update_run_status(config, run_id, "error")
        raise

    return run_dir


def verify(run_dir: pathlib.Path) -> None:
    print(f"\n{'='*60}")
    print(f"Verifying outputs in: {run_dir}")
    print("=" * 60)

    ok = True

    # train.jsonl
    jsonl_path = run_dir / "train.jsonl"
    if jsonl_path.exists():
        lines = jsonl_path.read_text().strip().splitlines()
        print(f"\n[train.jsonl] {len(lines)} line(s)")
        for line in lines:
            entry = json.loads(line)
            print(f"  epoch={entry['epoch']} train_loss={entry['train_loss']} val_loss={entry['val_loss']} elapsed={entry['elapsed_s']}s")
    else:
        print("[train.jsonl] MISSING")
        ok = False

    # train.log (last 5 lines)
    log_path = run_dir / "train.log"
    if log_path.exists():
        log_lines = log_path.read_text().splitlines()
        print(f"\n[train.log] {len(log_lines)} line(s) — last 5:")
        for line in log_lines[-5:]:
            print(f"  {line}")
    else:
        print("[train.log] MISSING")
        ok = False

    # heartbeat
    hb_path = run_dir / "heartbeat"
    if hb_path.exists():
        print(f"\n[heartbeat] {hb_path.read_text().strip()}")
    else:
        print("[heartbeat] MISSING")
        ok = False

    # run_meta.json
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"\n[run_meta.json]")
        print(f"  status      : {meta['status']}")
        print(f"  current_epoch: {meta['current_epoch']}")
        print(f"  best_epoch  : {meta['best_epoch']}")
        print(f"  best_val_loss: {meta['best_val_loss']}")
    else:
        print("[run_meta.json] MISSING")
        ok = False

    print(f"\n{'='*60}")
    print("RESULT:", "ALL GOOD" if ok else "SOME FILES MISSING")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash", action="store_true", help="Simulate Ctrl+C at epoch 2")
    parser.add_argument("--list", action="store_true", help="Show run registry after test")
    args = parser.parse_args()

    crash_at = 2 if args.crash else None
    run_dir = run_test(crash_at=crash_at)
    verify(run_dir)

    if args.list:
        print("\n--- Registry (python -m detection.list_runs) ---")
        from detection.list_runs import main as list_runs
        list_runs()


if __name__ == "__main__":
    main()
