# DINOv3 Image Classification Pipeline

> **PyTorch** • **Hugging Face Transformers** • **DINOv3** •
> **TensorBoard**

## Overview

This repository provides a complete pipeline for **training**,
**validation** and **inference** of an image classifier based on the
**DINOv3 Vision Transformer** backbone.

The framework supports:

-   Fine-tuning of DINOv3 with selective backbone unfreezing
-   Cross-Entropy and Focal Loss
-   Automatic class weighting for imbalanced datasets
-   Mixed Precision Training (AMP)
-   Early Stopping
-   Learning Rate Scheduling
-   TensorBoard logging
-   Automatic checkpointing
-   Comprehensive inference metrics

------------------------------------------------------------------------

## Repository Structure

``` text
.
├── train_dinov3.py      # Training & validation pipeline
├── infer_dinov3.py      # Inference / evaluation
├── common_dinov3.py     # Shared utilities and model definition
├── losses.py            # Focal Loss implementation
└── README.md
```

------------------------------------------------------------------------

## Dataset Structure

The dataset follows the standard `torchvision.datasets.ImageFolder`
format.

``` text
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

> **Important:** Train and validation folders must contain the
> exact same class names.

------------------------------------------------------------------------

# Training Pipeline

The training script performs the following steps:

1.  Load training and validation datasets.
2.  Apply the Hugging Face `AutoImageProcessor`.
3.  Load the pretrained DINOv3 backbone.
4.  Freeze the entire backbone.
5.  Unfreeze the last *N* transformer blocks.
6.  Train the classification head.
7.  Evaluate on the validation set.
8.  Save the best and latest checkpoints.
9.  Log metrics to TensorBoard.

## Configurable parameters

The main parameters are defined inside the `Config` dataclass.

  Parameter                  Description
  -------------------------- ----------------------------------------
  `model_id`                 Hugging Face DINOv3 model
  `epochs`                   Number of epochs
  `batch_size`               Batch size
  `lr_head`                  Learning rate of classification head
  `lr_backbone`              Learning rate of unfrozen backbone
  `dropout`                  Dropout before classifier
  `unfreeze_last_n_blocks`   Number of trainable transformer blocks
  `scheduler`                Cosine or Plateau
  `early_patience`           Early stopping patience
  `loss_name`                CrossEntropy or Focal Loss

------------------------------------------------------------------------

# Loss Functions

Two losses are available.

## Cross Entropy

``` python
loss_name = "ce"
```

## Focal Loss

``` python
loss_name = "focal"
```

Features:

-   automatic class weighting
-   configurable gamma
-   ignore_index support
-   class-balanced training

------------------------------------------------------------------------

# Inference Pipeline

Two inference modes are available.

## Labeled dataset

    LABELED_BY_SUBFOLDER = True

Outputs:

-   Accuracy
-   Loss
-   Macro F1-score
-   Balanced Accuracy
-   Classification Report
-   Confusion Matrix
-   Predictions CSV

## Unlabeled dataset

    LABELED_BY_SUBFOLDER = False

Outputs:

-   Predicted class
-   Confidence score
-   CSV predictions

------------------------------------------------------------------------

# Generated Files

## Training

``` text
RUN_DIR/
└── YYYYMMDD-HHMMSS/
    ├── config.json
    ├── class_to_idx.json
    ├── results_trainval.json
    ├── checkpoints/
    │   ├── best.pt
    │   └── last.pt
    └── tb/
```

## Inference

``` text
inference_outputs/
├── metrics.json
├── predictions.csv
├── classification_report.txt
├── confusion_matrix.npy
├── confusion_matrix.png
└── tb/
```

------------------------------------------------------------------------

# Environment Variables

``` bash
export HF_TOKEN="YOUR_TOKEN"
export DATA_DIR="/path/to/dataset"
export RUN_DIR="/path/to/results"
export EPOCHS=30
export UNFREEZE=1
export EARLY_PATIENCE=15
```

------------------------------------------------------------------------

# Usage

## Train

``` bash
python train_dinov3.py
```

## Inference

``` bash
python infer_dinov3.py
```

## TensorBoard

``` bash
tensorboard --logdir <RUN_DIR>
```

------------------------------------------------------------------------

# Main Components

## train_dinov3.py

Responsible for:

-   data loading
-   optimization
-   checkpointing
-   validation
-   TensorBoard logging

## infer_dinov3.py

Responsible for:

-   checkpoint loading
-   evaluation
-   metrics computation
-   confusion matrix generation
-   CSV export

## common_dinov3.py

Shared utilities:

-   model definition
-   reproducibility
-   freeze/unfreeze utilities
-   checkpoint management

## losses.py

Custom implementation of the multi-class Focal Loss.

------------------------------------------------------------------------

# Requirements

-   Python ≥ 3.10
-   PyTorch
-   torchvision
-   transformers
-   scikit-learn
-   matplotlib
-   Pillow
-   tensorboard
-   NumPy

------------------------------------------------------------------------

# Notes

-   A valid **HF_TOKEN** is required to download the DINOv3 backbone.
-   The best model is selected according to the **validation loss**.
-   Class mappings are stored inside every checkpoint to ensure
    reproducible inference.
-   Mixed Precision (AMP) is automatically enabled when CUDA is
    available.
