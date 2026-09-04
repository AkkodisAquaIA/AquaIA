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
        # Z: create new or get used logger with specified name
        self._logger = logging.getLogger(f"training.{run_id}")
        # Z: set logging level to INFO (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        self._logger.setLevel(logging.INFO)
        # Z: prohibit propagation to parent loggers (avoid duplicate logs)
        self._logger.propagate = False
        # Z: handlers decide where the log messages go (file, console, etc.)
        # Z: clear all handlers bound to this logger
        self._logger.handlers.clear()
        # Z: create a log formatter
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Z: create a file handler to append logs to a file
        fh = logging.FileHandler(self._log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        # Z: add the file handler to the logger, now logs will be appended to the file
        self._logger.addHandler(fh)
        # Z: create a stream handler to output logs to the console
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        # Z: add the stream handler to the logger, now logs will be output to the console
        self._logger.addHandler(sh)

        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})

        if resume and self._meta_path.exists():
            # Z: read meta information and load as a dictionary
            self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._meta["status"] = "resumed"
            self._meta["pid"] = os.getpid()
            # Z: update the last_updated timestamp to the current UTC time
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
        # Z: write the meta information to the run_meta.json file
        self._write_meta()

        model_str = f"{model_cfg.get('family', '')}_{model_cfg.get('size', '')} ({model_cfg.get('init', '')})"
        action = "RESUMED" if resume else "started"
        # Z: log an INFO level message into train.log + console 
        self._logger.info(f"Run {action} — run_id={run_id} | pid={os.getpid()}")
        self._logger.info(f"Run dir: {run_dir}")
        self._logger.info(f"Model: {model_str} | epochs={training_cfg.get('epochs')} | batch={training_cfg.get('batch')} | lr={training_cfg.get('lr0')}")

    # ── Public API ──────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        """Z: simple encapsulation of internal Python logger to log an INFO level message into train.log + console."""
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Z: simple encapsulation of internal Python logger to log a WARNING level message into train.log + console."""
        self._logger.warning(msg)

    def log_device(self, device: str, use_amp: bool, dataset_info: Optional[str] = None) -> None:
        """Z: log device, AMP usage and dataset info into train.log + console."""
        self._logger.info(f"Device: {device} | AMP: {use_amp}")
        if dataset_info:
            self._logger.info(f"Dataset: {dataset_info}")

    def log_epoch(self, epoch: int, total_epochs: int, metric_dict: dict, lr: float) -> None:
        """Z: log epoch metrics into train.jsonl, train.log + console and update meta information."""
        elapsed = time.time() - self.start_time
        train_loss = metric_dict.get("train", {}).get("loss", 0.0)
        val_loss = metric_dict.get("val", {}).get("loss", 0.0)

        # JSONL — flush immediately so it survives a crash
        entry = {
            "epoch": epoch,
            # Z: round the train and val losses to 6 decimal places
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "lr": lr,
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_s": round(elapsed),
        }
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            # Z: json.dumps converts the entry dictionary to a JSON string
            f.write(json.dumps(entry) + "\n")
            # Z: flush the file buffer to ensure the data is written to disk immediately
            f.flush()

        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        self._logger.info(f"[EPOCH {epoch:>3}/{total_epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e} | {elapsed_str}")

        self._meta["current_epoch"] = epoch
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    def log_best(self, epoch: int, val_loss: float) -> None:
        """Z: log when a new best validation loss is achieved into train.log + console and update meta information."""
        self._logger.info(f"[BEST ] New best at epoch {epoch} — val_loss={val_loss:.4f}")
        self._meta["best_epoch"] = epoch
        self._meta["best_val_loss"] = round(float(val_loss), 6)
        self._write_meta()

    def heartbeat(self, epoch: int, batch_idx: int, total_batches: int) -> None:
        """Z: write current timestamp, epoch, batch index every N batches into heartbeat file."""
        self._batch_counter += 1
        # Z: % operator returns the remainder of the division
        if self._batch_counter % self._heartbeat_every != 0:
            return
        ts = datetime.utcnow().isoformat()
        self._heartbeat_path.write_text(f"{ts} epoch={epoch} batch={batch_idx}/{total_batches}\n")

    def finish(self) -> None:
        """Z: log training completion, total elapsed time into train.log + console and update meta information."""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        self._logger.info(f"Training complete — total time: {elapsed_str}")
        self._meta["status"] = "done"
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    def crash(self, error: str) -> None:
        """Z: log training crash, error message into train.log + console and update meta information."""
        self._logger.error(f"Training crashed: {error}")
        self._meta["status"] = "error"
        self._meta["error"] = error
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    def interrupted(self) -> None:
        """Z: log training interrupted (KeyboardInterrupt) into train.log + console and update meta information."""
        self._logger.warning("Training interrupted (KeyboardInterrupt)")
        self._meta["status"] = "interrupted"
        self._meta["last_updated"] = datetime.utcnow().isoformat()
        self._write_meta()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _write_meta(self) -> None:
        """Z: write the current meta information to the run_meta.json file."""
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)
            f.flush()
