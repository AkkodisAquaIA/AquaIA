# Training logging

The custom logging system is currently integrated with DINO training. YOLO
training continues to use Ultralytics logging and checkpoints.

## Configuration

The logging section in `detection/train_config.yaml` controls periodic DINO
checkpoints:

```yaml
logging:
  # Save last.pt and the training state every N epochs.
  # With 0, they are saved only after a normal completion.
  save_period: 5
```

## Generated files

Each DINO run writes:

```text
<run_dir>/
├── train.log                   # Human-readable log, also printed to the console
├── run_meta.json              # Status, PID, progress and best validation loss
├── last_training_state.pt     # Epoch, optimizer, scaler and scheduler state
└── weights/
    ├── best.pt                # Model with the lowest validation loss
    └── last.pt                # Latest periodic or final model checkpoint
```

`run_meta.json` uses the statuses `running`, `resumed`, `done`, `error`, and
`interrupted`. It is updated after every epoch and when the process finishes,
fails, or receives `KeyboardInterrupt`.

Checkpoint behavior:

- `best.pt` is replaced whenever validation loss improves or equals the best value.
- With a positive `save_period`, both `last` files are replaced every N epochs.
- A normal completion always writes both `last` files, even when
  `save_period` is `0`.
- Resume reuses the run directory, appends to `train.log`, and restores the
  available model, optimizer, scaler, scheduler, saved epoch, and best
  validation loss.

## Commands

Run commands from the repository root:

```bash
# Start with the default configuration
python main.py train

# Start with another configuration
python main.py train --config path/to/train_config.yaml

# Resume an existing DINO run
python main.py train \
  --config detection/train_config.yaml \
  --resume results/detect/dinov3_small_pretrained/<run_id>

# Follow the text log from another terminal
tail -f results/detect/dinov3_small_pretrained/<run_id>/train.log

# Inspect current run metadata
cat results/detect/dinov3_small_pretrained/<run_id>/run_meta.json
```

## Running through SSH with tmux

Use `tmux` so training survives an SSH disconnection:

```bash
tmux new -s aquaia-training

# Inside tmux, from the repository root
python main.py train --config detection/train_config.yaml
```

Detach without stopping training with `Ctrl+B`, then `D`. Later:

```bash
tmux ls
tmux attach -t aquaia-training
```

Open another shell or tmux window to follow `train.log` while training
continues. If tmux is unavailable, a minimal alternative is:

```bash
nohup python main.py train --config detection/train_config.yaml \
  > runs/current.log 2>&1 &
echo $! > runs/current.pid
```

## Known limitations and planned work

- Integrate the custom text log, `run_meta.json`, and lifecycle handling with
  YOLO through Ultralytics callbacks; support YOLO resume through its native
  resume mechanism.
- Treat post-training evaluation and artifact generation as part of the run
  lifecycle. The current DINO code marks a run `done` before those steps, so a
  later failure is not reflected in the status.
- Validate resume inputs as a complete, consistent checkpoint set. Missing
  model or training-state files are currently skipped independently.
- Preserve total elapsed time across resumed sessions; the current elapsed
  timer restarts when the process resumes.
