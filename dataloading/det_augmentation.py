# YOLO style augmentation module for AquaIA detection

import numpy as np

from ultralytics.data.augment import (
    RandomHSV,
    RandomPerspective,
    RandomFlip,
    Mosaic,
    CutMix,
    Compose,
    Format,
    LetterBox,
)
from ultralytics.utils.instance import Instances


class DetectionAugmentation:
    """Ultralytics detection augmentations adapted to the AquaIA dataset format.
    Supports hsv, degrees, translate, scale, flipud, fliplr, mosaic, cutmix.
    Augmentations like shear, perspective, bgr, mixup are discussed to be not useful."""

    def __init__(self, dataset, img_size, config):
        # dataset is an instance of BaseDetectionDataset
        self.dataset = dataset
        self.img_size = int(img_size)
        self.config = config
        # If True enable Mosaic and CutMix, if close_mosaic() then False
        self.multi_image_enabled = True
        # Construct data augmentation pipeline
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        """Build the data augmentation pipeline based on the configuration."""
        mosaic_probability = self.config.get("mosaic", 0.0) if self.multi_image_enabled else 0.0
        cutmix_probability = self.config.get("cutmix", 0.0) if self.multi_image_enabled else 0.0

        # Construct mosaic object
        mosaic = Mosaic(
            # Provide dataset access for sampling auxiliary images
            dataset=self.dataset,
            imgsz=self.img_size,
            p=mosaic_probability,
        )
        # Construct geometric augmentation object
        affine = RandomPerspective(
            degrees=self.config.get("degrees", 0.0),
            translate=self.config.get("translate", 0.0),
            scale=self.config.get("scale", 0.0),
            # Adjust image size and bbox before applying (only if mosaic is not applied)
            pre_transform=LetterBox(new_shape=(self.img_size, self.img_size)),
        )
        # Compose a pipeline of pre-transformations (mosaic then affine, same as YOLO)
        # affine is necessary for image size and bbox adjustment
        # Unify image size, adjust bbox
        # Applied to main image: build final augmented image
        # Applied to auxiliary image: build element for CutMix
        pre_transform = Compose([mosaic, affine])

        # Construct and return a full augmentation pipeline
        return Compose(
            [
                # Pre transform main image
                pre_transform,
                # Construct cutmix object
                CutMix(
                    # Provide dataset access for sampling an auxiliary image
                    dataset=self.dataset,
                    # Pre transform auxiliary image
                    # mosaic is not necessary for CutMix but multi-augmentation is allowed here
                    # affine is necessary to preprocess auxiliary image including image size and bbox adjustment
                    pre_transform=pre_transform,
                    p=cutmix_probability,
                ),
                RandomHSV(
                    hgain=self.config.get("hsv_h", 0.0),
                    sgain=self.config.get("hsv_s", 0.0),
                    vgain=self.config.get("hsv_v", 0.0),
                ),
                RandomFlip(
                    p=self.config.get("flipud", 0.0),
                    direction="vertical",
                ),
                RandomFlip(
                    p=self.config.get("fliplr", 0.0),
                    direction="horizontal",
                ),
                # Convert augmented outputs into format for downstream Dataset and DINO training pipelines
                Format(
                    bbox_format="xywh",
                    normalize=True,
                    batch_idx=False,
                    bgr=0.0,
                ),
            ]
        )

    def __call__(self, labels):
        """Make the DetectionAugmentation instance callable like a function.
        "labels" contains image, class, bbox. Process image, labels and return augmentation results."""
        return self.transforms(labels)

    def close_mosaic(self):
        """Disable Mosaic and CutMix while keeping single-image transforms active."""
        if self.multi_image_enabled:
            self.multi_image_enabled = False
            self.transforms = self._build_transforms()


def build_ultralytics_labels(img, labels, boxes, img_path):
    """Build the label dictionary expected by Ultralytics detection transforms."""
    height, width = img.shape[:2]
    # Placeholder
    segments = np.zeros((0, 1000, 2), dtype=np.float32)
    instances = Instances(
        bboxes=boxes.astype(np.float32, copy=True),
        segments=segments,
        bbox_format="xywh",
        normalized=True,
    )
    return {
        "img": img,
        "cls": labels.astype(np.float32, copy=True).reshape(-1, 1),
        "instances": instances,
        "im_file": str(img_path),
        "ori_shape": (height, width),
        "resized_shape": (height, width),
    }
