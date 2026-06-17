from pathlib import Path


class CheckpointManager:
    """Saves best.pt on improvement and last.pt every save_period epochs."""

    def __init__(self, run_dir: str, save_period: int = 0):
        self.weights_dir = Path(run_dir) / "weights"
        self.run_dir = Path(run_dir)
        # 0 = only best, N = also save last every N epochs
        self.save_period = save_period

    def step(self, epoch: int, model, optimizer, scaler, scheduler, is_best: bool) -> None:
        if is_best:
            from detection.checkpoint import save_model_checkpoint

            save_model_checkpoint(path=str(self.weights_dir / "best.pt"), model=model)

        if self.save_period > 0 and epoch % self.save_period == 0:
            self._save_last(epoch, model, optimizer, scaler, scheduler)

    def save_final(self, epoch: int, model, optimizer, scaler, scheduler) -> None:
        """Always called at end of training (normal completion)."""
        self._save_last(epoch, model, optimizer, scaler, scheduler)

    def _save_last(self, epoch: int, model, optimizer, scaler, scheduler) -> None:
        from detection.checkpoint import save_model_checkpoint, save_training_state_checkpoint

        save_model_checkpoint(path=str(self.weights_dir / "last.pt"), model=model)
        save_training_state_checkpoint(
            path=str(self.run_dir / "last_training_state.pt"),
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        )
