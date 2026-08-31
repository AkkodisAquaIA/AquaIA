# Author : GPT 5.6 Sol High. Exactitude à vérifier!
# ruff: noqa: E402

# Samples two training images with random seed 42.
# Reads augmentation settings from detection/train_config.yaml.
# Applies each supported detection augmentation independently.
# Saves two before/after comparison groups for every augmentation.
# Uses the same Ultralytics transforms and label format as dataloading\augmentation.py.

import argparse
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics.data.augment import (
    Compose,
    CutMix,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)

from dataloading.augmentation import DetectionAugmentation
from dataloading.datasets import JpgDetectionDataset


DEFAULT_CONFIG = BASE_DIR / "detection" / "train_config.yaml"
DEFAULT_DATASET = BASE_DIR / "datasets" / "coco_cus_mat_hun"
DEFAULT_OUTPUT = BASE_DIR / "results" / "augmentation_samples"
SEED = 42
NUM_GROUPS = 2
TITLE_FONT_SIZE = 24
PANEL_FONT_SIZE = 20

AUGMENTATION_NAMES = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "cutmix",
    "close_mosaic",
)

DEFAULT_AUGMENTATION_VALUES = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Save two before/after examples for each configured DINO/DETR augmentation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_augmentation_config(config_path):
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config.get("training", {})
    augmentation_config = {name: training_config.get(name, default) for name, default in DEFAULT_AUGMENTATION_VALUES.items()}
    augmentation_config["augment"] = bool(training_config.get("augment", False))
    augmentation_config["imgsz"] = int(training_config.get("imgsz", 640))
    augmentation_config["epochs"] = int(training_config.get("epochs", 1))
    return augmentation_config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def build_formatter():
    return Format(
        bbox_format="xywh",
        normalize=True,
        batch_idx=False,
        bgr=0.0,
    )


def build_letterbox(img_size):
    return LetterBox(new_shape=(img_size, img_size))


def build_affine(config, img_size, active_name=None):
    values = {name: config[name] if name == active_name else 0.0 for name in ("degrees", "translate", "scale", "shear", "perspective")}
    return RandomPerspective(
        degrees=values["degrees"],
        translate=values["translate"],
        scale=values["scale"],
        shear=values["shear"],
        perspective=values["perspective"],
        pre_transform=build_letterbox(img_size),
    )


def build_multi_image_pre_transform(dataset, img_size, mosaic_probability=0.0):
    mosaic = Mosaic(
        dataset=dataset,
        imgsz=img_size,
        p=mosaic_probability,
    )
    affine = RandomPerspective(
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        pre_transform=build_letterbox(img_size),
    )
    return Compose([mosaic, affine])


def format_labels(labels):
    return build_formatter()(labels)


def build_before_sample(dataset, index, img_size):
    labels = dataset.get_image_and_label(index)
    labels = build_letterbox(img_size)(labels)
    return format_labels(labels)


def apply_single_image_augmentation(dataset, index, name, config, img_size):
    labels = dataset.get_image_and_label(index)

    if name in {"hsv_h", "hsv_s", "hsv_v"}:
        labels = build_letterbox(img_size)(labels)
        labels = RandomHSV(
            hgain=config["hsv_h"] if name == "hsv_h" else 0.0,
            sgain=config["hsv_s"] if name == "hsv_s" else 0.0,
            vgain=config["hsv_v"] if name == "hsv_v" else 0.0,
        )(labels)
    elif name in {"degrees", "translate", "scale", "shear", "perspective"}:
        labels = build_affine(
            config=config,
            img_size=img_size,
            active_name=name,
        )(labels)
    elif name in {"flipud", "fliplr"}:
        labels = build_letterbox(img_size)(labels)
        labels = RandomFlip(
            p=1.0,
            direction="vertical" if name == "flipud" else "horizontal",
        )(labels)
    else:
        raise ValueError(f"Unsupported single-image augmentation: {name}")

    return format_labels(labels)


def apply_mosaic(dataset, index, img_size):
    labels = dataset.get_image_and_label(index)
    labels = build_multi_image_pre_transform(
        dataset=dataset,
        img_size=img_size,
        mosaic_probability=1.0,
    )(labels)
    return format_labels(labels)


def apply_mixup(dataset, index, img_size):
    pre_transform = build_multi_image_pre_transform(
        dataset=dataset,
        img_size=img_size,
    )
    labels = pre_transform(dataset.get_image_and_label(index))
    labels = MixUp(
        dataset=dataset,
        pre_transform=pre_transform,
        p=1.0,
    )(labels)
    return format_labels(labels)


def apply_cutmix(dataset, index, img_size, seed):
    result = None
    for attempt in range(100):
        set_random_seed(seed + attempt)
        pre_transform = build_multi_image_pre_transform(
            dataset=dataset,
            img_size=img_size,
        )
        labels = pre_transform(dataset.get_image_and_label(index))
        original_img = labels["img"].copy()
        labels = CutMix(
            dataset=dataset,
            pre_transform=pre_transform,
            p=1.0,
        )(labels)
        result = labels
        if not np.array_equal(original_img, labels["img"]):
            break
    return format_labels(result)


def apply_multi_image_augmentation(dataset, index, name, img_size, seed):
    if name == "mosaic":
        return apply_mosaic(dataset, index, img_size)
    if name == "mixup":
        return apply_mixup(dataset, index, img_size)
    if name == "cutmix":
        return apply_cutmix(dataset, index, img_size, seed)
    raise ValueError(f"Unsupported multi-image augmentation: {name}")


def apply_close_mosaic(dataset, index, img_size, seed):
    config = dict(DEFAULT_AUGMENTATION_VALUES)
    config.update(
        {
            "mosaic": 1.0,
            "mixup": 1.0,
            "cutmix": 1.0,
        }
    )
    augmentation = DetectionAugmentation(
        dataset=dataset,
        img_size=img_size,
        config=config,
    )

    set_random_seed(seed)
    before_close = augmentation(dataset.get_image_and_label(index))
    augmentation.close_mosaic()
    set_random_seed(seed)
    after_close = augmentation(dataset.get_image_and_label(index))
    return before_close, after_close


def formatted_to_image_and_boxes(labels):
    image = labels["img"].detach().cpu().permute(1, 2, 0).numpy()
    image = np.ascontiguousarray(image.astype(np.uint8, copy=False))

    boxes = labels["bboxes"].detach().cpu().numpy().reshape(-1, 4)
    classes = labels["cls"].detach().cpu().numpy().reshape(-1).astype(np.int64)
    height, width = image.shape[:2]

    boxes_xyxy = np.empty_like(boxes)
    if len(boxes):
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, width - 1)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, height - 1)
    return image, boxes_xyxy, classes


def draw_boxes(labels, class_names):
    image, boxes, classes = formatted_to_image_and_boxes(labels)
    rendered = Image.fromarray(image)
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    for box, class_id in zip(boxes, classes):
        x1, y1, x2, y2 = box.tolist()
        color = (
            int((37 * (class_id + 1)) % 255),
            int((97 * (class_id + 1)) % 255),
            int((157 * (class_id + 1)) % 255),
        )
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        text = f"{class_id}: {class_name}"
        text_box = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    return rendered


def load_font(size):
    font_paths = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def add_panel_header(image, header, header_height=48):
    panel = Image.new("RGB", (image.width, image.height + header_height), color=(245, 245, 245))
    panel.paste(image, (0, header_height))
    draw = ImageDraw.Draw(panel)
    font = load_font(PANEL_FONT_SIZE)
    draw.text((10, 12), header, fill=(20, 20, 20), font=font)
    return panel


def save_comparison(output_path, title, comparisons, class_names):
    gap = 12
    title_height = 60
    panels = []
    for group_index, (before_labels, after_labels) in enumerate(comparisons, start=1):
        before = add_panel_header(
            draw_boxes(before_labels, class_names),
            f"Group {group_index} - before",
        )
        after = add_panel_header(
            draw_boxes(after_labels, class_names),
            f"Group {group_index} - after",
        )
        panels.append((before, after))

    panel_width = panels[0][0].width
    panel_height = panels[0][0].height
    canvas = Image.new(
        "RGB",
        (
            panel_width * 2 + gap,
            title_height + panel_height * len(panels) + gap * (len(panels) - 1),
        ),
        color=(225, 225, 225),
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(TITLE_FONT_SIZE)
    draw.text((10, 16), title, fill=(10, 10, 10), font=font)

    for row, (before, after) in enumerate(panels):
        y = title_height + row * (panel_height + gap)
        canvas.paste(before, (0, y))
        canvas.paste(after, (panel_width + gap, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def augmentation_title(name, config):
    if name in {"flipud", "fliplr", "mosaic", "mixup", "cutmix"}:
        return f"{name}: configured probability={config[name]}, preview probability=1.0"
    if name == "close_mosaic":
        close_epoch = max(config["epochs"] - int(config["close_mosaic"]), 0)
        return f"close_mosaic={config['close_mosaic']}, epochs={config['epochs']}, close at zero-based epoch {close_epoch}, preview multi-image probability=1.0"
    return f"{name}={config[name]}"


def main():
    args = parse_args()
    config_path = args.config.resolve()
    dataset_root = args.dataset.resolve()
    output_dir = args.output.resolve()
    config = load_augmentation_config(config_path)
    img_size = config["imgsz"]

    dataset = JpgDetectionDataset(
        dataset_root=dataset_root,
        data_split="train",
        img_size=img_size,
        augment=False,
    )
    if len(dataset) < NUM_GROUPS:
        raise ValueError(f"At least {NUM_GROUPS} training images are required, found {len(dataset)}")

    sampled_indices = random.Random(args.seed).sample(range(len(dataset)), NUM_GROUPS)
    sampled_names = [dataset.target_files[index].stem for index in sampled_indices]

    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_root / 'images' / 'train'}")
    print(f"Seed: {args.seed}")
    print(f"Selected images: {sampled_names}")
    if not config["augment"]:
        print("Warning: training.augment is false; previews still force each requested operation for inspection.")

    for augmentation_index, name in enumerate(AUGMENTATION_NAMES):
        comparisons = []
        for group_index, sample_index in enumerate(sampled_indices):
            operation_seed = args.seed + augmentation_index * 1000 + group_index * 100
            set_random_seed(operation_seed)

            if name == "close_mosaic":
                before, after = apply_close_mosaic(
                    dataset=dataset,
                    index=sample_index,
                    img_size=img_size,
                    seed=operation_seed,
                )
            else:
                before = build_before_sample(dataset, sample_index, img_size)
                set_random_seed(operation_seed)
                if name in {"mosaic", "mixup", "cutmix"}:
                    after = apply_multi_image_augmentation(
                        dataset=dataset,
                        index=sample_index,
                        name=name,
                        img_size=img_size,
                        seed=operation_seed,
                    )
                else:
                    after = apply_single_image_augmentation(
                        dataset=dataset,
                        index=sample_index,
                        name=name,
                        config=config,
                        img_size=img_size,
                    )
            comparisons.append((before, after))

        output_path = output_dir / f"{augmentation_index + 1:02d}_{name}.jpg"
        save_comparison(
            output_path=output_path,
            title=augmentation_title(name, config),
            comparisons=comparisons,
            class_names=dataset.class_names,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
