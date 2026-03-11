# DINOv3 Folder Inference Script (`infer_dinov3.py`)

This script runs **inference on a folder of images** using a trained **DINOv3 classifier** checkpoint produced by the training pipeline (e.g. `train_dinov3.py`). 
It supports two modes:

- **Labeled mode** (`LABELED_BY_SUBFOLDER=True`): expects `INPUT_DIR/<class_name>/*.jpg` and computes metrics + confusion matrix.
- **Unlabeled mode** (`LABELED_BY_SUBFOLDER=False`): recursively scans `INPUT_DIR/**` and outputs predictions + confidence.

---

## Features

### ✅ Inference
- Loads a trained checkpoint (`best.pt` or `last.pt`) from `RUN_DIR/checkpoints/`
- Uses the same Hugging Face image processor as the backbone (`AutoImageProcessor`)
- Mixed precision inference (AMP) when CUDA is available

### ✅ Outputs
Depending on the mode, the script exports:

**Common outputs (both modes)**
- `RUN_DIR/inference_outputs/metrics.json`
- `RUN_DIR/inference_outputs/predictions.csv`

**Labeled mode only**
- `classification_report.txt` (precision/recall/F1 per class)
- `confusion_matrix.npy`
- `confusion_matrix.png` (normalized by true class)

### TensorBoard (optional)
If `WRITE_TENSORBOARD=True`, logs are written to:
- `RUN_DIR/inference_outputs/tb/`

---

## Requirements

- Python 3.8+
- torch
- torchvision
- transformers
- tensorboard
- numpy
- scikit-learn
- matplotlib
- pillow

Install:

```bash
pip install torch torchvision transformers tensorboard numpy scikit-learn matplotlib pillow
```

---

## Inputs & Folder Structure

### 1) Checkpoint directory

Set `RUN_DIR` to a training run directory that contains:

```
RUN_DIR/
  checkpoints/
    best.pt
    last.pt
```

Choose which checkpoint to load:

```python
CHECKPOINT_NAME = "best.pt"  # or "last.pt"
```

### 2) Input folder

Set `INPUT_DIR` to the folder containing images.

#### Labeled mode (`LABELED_BY_SUBFOLDER=True`)
Expected structure:

```
INPUT_DIR/
  classA/
    *.jpg
  classB/
    *.jpg
  ...
```

Notes:
- Class folder names must match the classes the model was trained on.
- Missing class folders are ignored.
- If no image matches model classes, the script raises an error.

#### Unlabeled mode (`LABELED_BY_SUBFOLDER=False`)
The script scans recursively:

```
INPUT_DIR/
  any/subfolders/you/want/
    *.jpg *.png ...
```

Supported extensions are defined in `EXTS`.

---

## How to Run

```bash
python3 infer_dinov3.py
```

---

## Configuration (Top of the Script)

Key parameters you can edit:

- `RUN_DIR`: training run directory containing `checkpoints/`
- `CHECKPOINT_NAME`: `"best.pt"` or `"last.pt"`
- `INPUT_DIR`: input folder to infer on
- `LABELED_BY_SUBFOLDER`: labeled vs. unlabeled mode
- `BATCH_SIZE`, `NUM_WORKERS`
- `USE_AMP`: mixed precision (recommended on GPU)
- `OUT_DIR_NAME`: output subfolder inside `RUN_DIR`
- `WRITE_TENSORBOARD`: enable TensorBoard logging

---

## Outputs

All outputs are written to:

```
RUN_DIR/inference_outputs/
```

### `predictions.csv`

#### Labeled mode columns
- `path`: image filepath
- `true_class`: ground truth folder name
- `pred_class`: predicted class

#### Unlabeled mode columns
- `path`: image filepath
- `pred_class`: predicted class
- `confidence`: max softmax probability

### `metrics.json`
Example (labeled mode):

```json
{
  "mode": "labeled_by_subfolder",
  "input_dir": "...",
  "n_images": 123,
  "loss": 0.42,
  "acc": 0.88,
  "macro_f1": 0.81,
  "balanced_acc": 0.85,
  "model_id": "...",
  "checkpoint": "...",
  "run_dir": "..."
}
```

---

## TensorBoard

If enabled:

```bash
tensorboard --logdir RUN_DIR/inference_outputs/tb
```

Then open:

http://localhost:6006

---

## Notes / Troubleshooting

- **Checkpoint not found**: verify `RUN_DIR/checkpoints/<CHECKPOINT_NAME>` exists.
- **No images found**:
  - labeled mode: ensure folder names match training classes and contain images
  - unlabeled mode: ensure images have extensions in `EXTS`
- **CUDA out of memory**: reduce `BATCH_SIZE`.

---

## Related Files

- `common_dinov3.py` provides shared utilities:
  - `DinoV3Classifier`
  - checkpoint loading helpers (e.g. `load_checkpoint`)

