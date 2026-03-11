# DINOv3 Training Script (train_dinov3.py)

This script trains an image classifier based on a **DINOv3 backbone**
using PyTorch and Hugging Face Transformers.

## Features

-   Uses a pretrained DINOv3 model
    (`facebook/dinov3-vits16-pretrain-lvd1689m` by default)
-   Custom classification head
-   Optional partial fine-tuning of the backbone
-   Mixed precision training (AMP)
-   TensorBoard logging
-   Checkpointing (last / best)
-   Early stopping

## Requirements

-   Python 3.8+
-   torch
-   torchvision
-   transformers
-   tensorboard

Install dependencies:

``` bash
pip install torch torchvision transformers tensorboard
```

## Dataset Structure

The dataset directory must follow the ImageFolder format:

dataset_root/ train/ class_0/ class_1/ val/ class_0/ class_1/

Train and validation folders must contain the **same classes**.

## Running the Training

``` bash
python3 train_dinov3.py
```

Outputs are saved in the configured `run_dir`:

-   config.json
-   class_to_idx.json
-   TensorBoard logs (`tb/`)
-   checkpoints/
    -   last.pt
    -   best.pt

## Configuration

Training parameters are defined in the `Config` dataclass inside the
script:

-   epochs
-   batch_size
-   learning rates
-   scheduler
-   early stopping
-   AMP usage

Modify them directly in the script before running.

## TensorBoard

Launch TensorBoard:

``` bash
tensorboard --logdir <run_dir_parent>
```

Open:

http://localhost:6006

## Checkpoints

-   `last.pt` → latest epoch
-   `best.pt` → best validation loss

## Notes

-   Backbone is frozen by default (`unfreeze_last_n_blocks = 0`)
-   Increase `unfreeze_last_n_blocks` for fine-tuning
