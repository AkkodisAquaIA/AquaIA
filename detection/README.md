# AquaIA Detection

## Overview

This package provides object detection training, inference, evaluation, checkpointing, and prediction visualization. It supports DINO backbones with a DETR detection head and Ultralytics YOLO models through a shared command-line entry point.

In this repository, the detection module covers the top-level `main.py` and `benchmark_train.py` files, as well as all files under `data_processing/`, `dataloading/`, and `detection/`. For a file-by-file description, see **Repository structure** at the end of this document.

## Current support

| Backend | Training | Inference | Resume training | Data loading |
|---|---:|---:|---:|---|
| DINOv2 / DINOv3 + DETR | Yes | Yes | Yes | PIL or NVIDIA DALI |
| Ultralytics YOLO | Yes | Yes | No | Ultralytics for training / PIL for inference |

The DINO pipeline supports `small`, `base`, and `large` DINOv2 backbones, and `small`, `plus`, `base`, and `large` DINOv3 backbones. NVIDIA DALI is optional; the DINO pipeline falls back to the PIL-based loader when DALI is unavailable.

## Dataset format

Datasets use YOLO detection labels and the following layout:

```text
datasets/<dataset_name>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── <dataset_name>.yaml
└── stats.npy
```

Each label line must follow the normalized YOLO format:

```text
class_id x_center y_center width height
```

The dataset YAML defines the dataset paths and class names. `stats.npy` contains the channel mean and standard deviation used by the custom data loaders and can be generated with `data_processing/stats.py`.

## Quick start

Run commands from the repository root.

Train a model:

```bash
python main.py train
```

Resume a DINO training run:

```bash
python main.py train --resume <run_directory>
```

Run inference on the dataset and split selected in the inference configuration:

```bash
python main.py infer
```

Use a custom configuration file:

```bash
python main.py train --config <train_config_path>
python main.py infer --config <infer_config_path>
```

## Configuration files

The active configuration files are:

- `detection/train_config.yaml` for training.
- `detection/infer_config.yaml` for inference and evaluation.

## Output directories

Training runs are stored under:

```text
results/<task>/<model_family>_<model_size>_<initialization>/<run_id>/
```

A DINO run can contain model weights, the training state, the resolved configuration, metrics, logs, and prediction visualizations:

```text
<run_directory>/
├── weights/
│   ├── best.pt
│   └── last.pt
├── last_training_state.pt
├── resolved_config.yaml
├── metrics.npy
├── best_metric.npy
├── eval_predictions/
└── train_predictions/
```

By default, inference results are stored inside the selected training run:

```text
<run_directory>/inference/<dataset_name>_<timestamp>/
```

## Logging

For logging files, checkpoints, run status, resume behavior, and usage, see [TRAINING_LOGS.md](TRAINING_LOGS.md). For implementation details and design history, see [JOURNAL_TRAINING_LOGS.md](JOURNAL_TRAINING_LOGS.md).

## Repository structure

The Detection part contains the following folders and files.

```text
├── data_processing/
│   ├── coco_custom_split.py      # Splits the 2017 Train into train and test sets.
│   ├── preprocess_to_npy.py      # Creates npy_images.npy and stats.npy for datasets.py, images normalized to [0,1]. Windows doesn’t support dali package. Obsolete.
│   └── stats.py                  # Computes mean and std matching the original DALI pipeline  stats.npy.
│
├── dataloading/
│   └─ datasets.py                # For dataset loading, JpgDALIDataset, JpgDetectionDataset, DALIDetectionDataLoader.
│
├── detection/
│   ├── dino/
│   │   ├── DETR/
│   │   │   ├── detr.py           # DETR, prediction heads, aux_loss controls multioutput.
│   │   │   └── transformer.py    # Encoder, decoder, transformer for DETR. return_intermediate_dec controls multioutput.
│   │   │
│   │   ├── inference/
│   │   │   └── run.py            # Main inference process.
│   │   │
│   │   ├── training/
│   │   │   └── run.py            # Main training process.
│   │   │
│   │   ├── utils/
│   │   │   ├── matcher.py        # Hungarian matcher with class cost (modified to FocalLoss), bbox cost, GIoU cost.
│   │   │   └── misc.py           # Only accuracy, is_dist_avail_and_initialized, get_world_size used.
│   │   │
│   │   ├── backbone_id_map.py    # DINO model registration, where to find model weights.
│   │   ├── dino_detector.py      # Combines DINO and DETR to a complet model.
│   │   ├── loss.py               # Loss for DETR after backbone (class loss modified to FocalLoss).
│   │   ├── position_encoding.py  # 2D positional encoding for DETR.
│   │   └── predict.py            # One function to round image size, one function to infer on a batch of samples (for evaluation or visualization) and return predictions.
│   │
│   ├── logging/
│   │   ├── __init__.py           # Declares logging package.
│   │   ├── checkpoint_manager.py # Saves best.pt on improvement; last.pt + last_training_state.pt every save_period epochs and at the end of training.
│   │   ├── run_registry.py       # Registers a new run or update a record in registry.jsonl file.
│   │   └── training_logger.py    # TrainingLogger (train.jsonl, train.log, heartbeat, run_meta.json).
│   │
│   ├── utils/
│   │   ├── box_ops.py            # Bbox operations.
│   │   ├── config_utils.py       # Functions for saving model params and various states for training resume.
│   │   ├── import_utils.py       # Centralizes DALI imports, determines whether DALI can be used in current env.
│   │   ├── plot_utils.py         # Functions to annotate images, save some visualizations and plot metric curves.
│   │   └── profiling.py          # A pytorch profiler factory function, for execution performance monitoring.
│   │
│   ├── yolo/
│   │   ├── inference/
│   │   │   └── run.py            # Main inference process, loads the best YOLO checkpoint and evaluates it on the test dataset.
│   │   │
│   │   ├── training/
│   │   │   └── run.py            # Main training process, resolves the Ultralytics model identifier and launches training.
│   │   │
│   │   ├── batch_eval.py         # Evaluates multiple YOLO runs with yolo_run_diagnostics.py and generates CSV and Markdown reports.
│   │   ├── plot_metrics.py       # Plots training metrics for one YOLO run or compares metrics across multiple runs.
│   │   ├── predict.py            # Adapts Ultralytics YOLO predictions to the common detection prediction format.
│   │   └── yolo_run_diagnostics.py # Evaluates one YOLO run, analyzes prediction errors and IoU, and writes TensorBoard diagnostics.
│   │
│   ├── checkpoint.py             # Checkpoint tools, save model checkpoint, load model checkpoint.
│   ├── config_printer.py         # Prints config when training.
│   ├── infer_config.yaml         # Inference config.
│   ├── infer.py                  # Initializes test with config, detection/dino/inference/run.py/test_dino or detection/yolo/inference/run.py/test_yolo.
│   ├── JOURNAL_TRAINING_LOGS.md  # Explication of logging mechanism’s implementation.
│   ├── list_runs.py              # Reads the training run registry and displays all valid runs in a sorted, color-coded table. python -m detection.list_runs
│   ├── metric.py                 # Metrics’ update, print, save, calculate funcitons.
│   ├── run_utils.py              # Context and metric tools for inference.
│   ├── test_training_logs.py     # Test script for the training logs system — no GPU, no dataset, no torch required.
│   ├── train_config.yaml         # Training config.
│   ├── train.py                  # Initialize training with config, detection/dino/training/run.py/train_dino or detection/yolo/training/run.py/train_yolo.
│   └── TRAINING_LOGS.md          # Explication of logging mechanism and how to use.
│
└── main.py                       # Entry point, train (train.py/train_from_config) or infer (infer.py/infer_from_config).
```
