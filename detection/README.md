# AquaIA Detection

## Overview

This package provides object detection training, inference, evaluation, checkpointing, and prediction visualization. It supports DINO backbones with a DETR detection head and Ultralytics YOLO models through a shared command-line entry point.

In this repository, the detection module covers the top-level `main.py` file, as well as all files under `data_processing/`, `dataloading/`, and `detection/`. For a file-by-file description, see **Repository structure** at the end of this document.

## Current support

| Backend | Training | Inference | Resume training | Data loading |
|---|---:|---:|---:|---|
| DINOv2 / DINOv3 + DETR | Yes | Yes | Yes | PIL + PyTorch DataLoader |
| Ultralytics YOLO | Yes | Yes | No | Ultralytics for training / PIL for inference |

The DINO pipeline supports `small`, `base`, and `large` DINOv2 backbones, and `small`, `plus`, `base`, and `large` DINOv3 backbones.

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
└── stats_<image_size>.npy
```

Each label line must follow the normalized YOLO format:

```text
class_id x_center y_center width height
```

The dataset YAML defines the dataset paths and class names. `stats_<image_size>.npy` contains the channel mean and standard deviation used by the custom data loaders and can be generated with `data_processing/stats.py`.

## Quick start

Run commands from the repository root. By default, `python main.py train` and `python main.py infer` use the DINO configuration files: `detection/train_config_dino.yaml` and `detection/infer_config_dino.yaml`, respectively.

Train a DINO model:

```bash
python main.py train
```

Resume a DINO training run:

```bash
python main.py train --resume <run_directory>
```

Run DINO inference on the dataset and split selected in the inference configuration:

```bash
python main.py infer
```

For YOLO training and inference, explicitly select the corresponding YOLO configuration file:

```bash
python main.py train --config detection/train_config_yolo.yaml
python main.py infer --config detection/infer_config_yolo.yaml
```

Use a custom configuration file:

```bash
python main.py train --config <train_config_path>
python main.py infer --config <infer_config_path>
```

## Configuration files

The active configuration files are:

- `detection/train_config_dino.yaml` for DINO training (default).
- `detection/infer_config_dino.yaml` for DINO inference and evaluation (default).
- `detection/train_config_yolo.yaml` for YOLO training.
- `detection/infer_config_yolo.yaml` for YOLO inference and evaluation.
- `num_queries` in [dino/dino_detector.py](dino/dino_detector.py) sets the maximum number of objects the DINO / DETR model can predict per image; remember to adjust it for your dataset before training.

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

For the DINO text log, run metadata, checkpoints, resume behavior, CLI
commands, and tmux usage, see [logging/LOGGING.md](logging/LOGGING.md).

## Repository structure

The Detection part contains the following folders and files.

```text
├── data_processing/
│   ├── coco_custom_split.py      # Splits the 2017 Train into train and test sets.
│   ├── sample_augmentation.py    # Visualizes sample images and bounding boxes before and after applying detection augmentations.
│   ├── sample_coco_one_percent.py # Creates a reproducible 1% subset of each COCO split while preserving image-label pairs and ensuring coverage of all 80 classes.
│   └── stats.py                  # Computes image channel mean and standard deviation --> stats_<image_size>.npy.
│
├── dataloading/
│   ├── datasets.py               # Detection dataset, batch collation, and batch parsing helpers.
│   └── det_augmentation.py       # Builds Ultralytics-based detection augmentations and converts dataset samples to the label format required by those transforms.
│
├── detection/
│   ├── dino/
│   │   ├── DETR/
│   │   │   ├── __init__.py       # Exposes the DETR class.
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
│   │   ├── dino_detector.py      # Combines DINO and DETR, using only the final DETR decoder layer output for training / inference, without intermediate auxiliary outputs.
│   │   ├── loss.py               # Loss for DETR after backbone (class loss modified to FocalLoss).
│   │   ├── position_encoding.py  # 2D positional encoding for DETR.
│   │   └── predict.py            # One function to round image size, one function to infer on a batch of samples (for evaluation or visualization) and return predictions.
│   │
│   ├── logging/
│   │   ├── __init__.py           # Declares logging package.
│   │   ├── checkpoint_manager.py # Saves best.pt on improvement; last.pt + last_training_state.pt every save_period epochs and at the end of training.
│   │   ├── LOGGING.md            # Current logging behavior, usage, limitations, and planned work.
│   │   └── training_logger.py    # TrainingLogger (train.log and run_meta.json).
│   │
│   ├── utils/
│   │   ├── box_ops.py            # Bbox operations.
│   │   ├── config_utils.py       # Loads and saves configurations; resolves output directories and loads class names.
│   │   ├── plot_utils.py         # Functions to annotate images, save some visualizations and plot metric curves.
│   │   └── profiling.py          # A pytorch profiler factory function, for execution performance monitoring.
│   │
│   ├── yolo/
│   │   ├── inference/
│   │   │   └── run.py            # Main inference process, loads the best YOLO checkpoint and evaluates it on the configured dataset split.
│   │   │
│   │   ├── training/
│   │   │   └── run.py            # Main training process, resolves the Ultralytics model identifier and launches training.
│   │   │
│   │   ├── batch_eval.py         # Evaluates multiple YOLO runs with yolo_run_diagnostics.py and generates CSV and Markdown reports.
│   │   ├── plot_metrics.py       # Plots training metrics for one YOLO run or compares metrics across multiple runs.
│   │   ├── predict.py            # Adapts Ultralytics YOLO predictions to the common detection prediction format.
│   │   └── yolo_run_diagnostics.py # Evaluates one YOLO run, analyzes prediction errors and IoU, and writes TensorBoard diagnostics.
│   │
│   ├── checkpoint.py             # Saves model weights and saves/loads optimizer, scaler, scheduler, and epoch state.
│   ├── config_printer.py         # Prints config when training.
│   ├── infer_config_dino.yaml    # Inference config for DINO (default).
│   ├── infer_config_yolo.yaml    # Inference config for YOLO.
│   ├── inference_context.py      # Shared run context and header tools for inference.
│   ├── metric.py                 # Metrics’ update, print, save, calculate functions.
│   ├── runner.py                 # Loads configs and dispatches training (detection/<model>/training/run.py/train_<model>) or inference (detection/<model>/inference/run.py/infer_<model>).
│   ├── train_config_dino.yaml    # Training config for DINO (default).
│   └── train_config_yolo.yaml    # Training config for YOLO.
│
└── main.py                       # Entry point, dispatches train or infer commands through detection/runner.py/train_from_config or infer_from_config.
```
