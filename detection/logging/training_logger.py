import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class TrainingLogger:
    """Persistent training logger: JSONL per epoch, text log, heartbeat, run_meta.json."""

    def __init__(self, run_dir: str, run_id: str, config: dict, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.start_time = time.time()
        self._batch_counter = 0
        self._heartbeat_every = config.get("logging", {}).get("heartbeat_every_n_batches", 10)

        self._jsonl_path = self.run_dir / "train.jsonl"
        self._log_path = self.run_dir / "train.log"
        self._heartbeat_path = self.run_dir / "heartbeat"
        self._meta_path = self.run_dir / "run_meta.json"

        # Python logger — clear any handlers from a previous run with the same run_id
        self._logger = logging.getLogger(f"training.{run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(self._log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})

        if resume and self._meta_path.exists():
            self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._meta["status"] = "resumed"
            self._meta["pid"] = os.getpid()
            self._meta["last_updated"] = datetime.utcnow().isoformat()
        else:
            self._meta = {
                "run_id": run_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "pid": os.getpid(),
                "config": {
                    "model": f"{model_cfg.get('family', '')}_{model_cfg.get('size', '')}",
                    "epochs": training_cfg.get("epochs", 0),
                    "batch": training_cfg.get("batch", 0),
                    "lr": training_cfg.get("lr0"),
                },
                "best_epoch": None,
                "best_val_loss": None,
                "current_epoch": 0,
                "total_epochs": training_cfg.get("epochs", 0),
            }
        self._write_meta()

        model_str = f"{model_cfg.get('family', '')}_{model_cfg.get('size', '')} ({model_cfg.get('init', '')})"
        action = "RESUMED" if resume else "started"
        self._logger.info(f"Run {action} — run_id={run_id} | pid={os.getpid()}")
        self._logger.info(f"Run dir: {run_dir}")
        self._logger.info(f"Model: {model_str} | epochs={training_cfg.get('epochs')} | batch={training_cfg.get('batch')} | lr={training_cfg.get('lr0')}")

    # ── Public API ──────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def log_device(self, device: str, use_amp: bool, dataset_info: Optional[str] = None) -> None:
        self._logger.info(f"Device: {device} | AMP: {use_amp}")
        if dataset_info:
            self._logger.info(f"Dataset: {dataset_info}")

    def log_epoch(self, epoch: int, total_epochs: int, metric_dict: dict, lr: float) -> None:
        elapsed = time.time() - self.start_time
        train_loss = metric_dict.get("train", {}).get("loss", 0.0)
        val_loss = metric_dict.get("val", {}).get("loss", 0.0)

        # JSONL — flush immediately so it survives a crash
        entry = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "lr": lr,
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_s": round(elapsed),
        }
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()

        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        self._logger.info(f"[EPOCH {epoch:>3}/{total_epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e} | {elapsed_str}")

        self._meta["current_epoch"] = epoch
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    def log_best(self, epoch: int, val_loss: float) -> None:
        self._logger.info(f"[BEST ] New best at epoch {epoch} — val_loss={val_loss:.4f}")
        self._meta["best_epoch"] = epoch
        self._meta["best_val_loss"] = round(float(val_loss), 6)
        self._write_meta()

    def heartbeat(self, epoch: int, batch_idx: int, total_batches: int) -> None:
        self._batch_counter += 1
        if self._batch_counter % self._heartbeat_every != 0:
            return
        ts = datetime.utcnow().isoformat()
        self._heartbeat_path.write_text(f"{ts} epoch={epoch} batch={batch_idx}/{total_batches}\n")

    def finish(self) -> None:
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        self._logger.info(f"Training complete — total time: {elapsed_str}")
        self._meta["status"] = "done"
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    def crash(self, error: str) -> None:
        self._logger.error(f"Training crashed: {error}")
        self._meta["status"] = "error"
        self._meta["error"] = error
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    def interrupted(self) -> None:
        self._logger.warning("Training interrupted (KeyboardInterrupt)")
        self._meta["status"] = "interrupted"
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _write_meta(self) -> None:
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)
            f.flush()
