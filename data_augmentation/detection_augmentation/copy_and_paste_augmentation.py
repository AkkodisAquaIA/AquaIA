"""Création de datasets YOLO par augmentation copy-paste.

Le script contient trois fonctions principales :
- generate_augmented_dataset : crée le dataset YOLO à partir de PNG détourés ;
- create_cutouts_sam3 : détoure les insectes présents sur des images globales ;
- create_cutouts_sam2_otsu : détoure des images déjà cropées.

Pour une utilisation simple, modifier les paramètres globaux puis lancer ce fichier.
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Paramètres principaux à modifier
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUTOUT_DIR = PROJECT_ROOT / "datasets/LPL_dataset_cutouts_single"
BACKGROUND_DIR = PROJECT_ROOT / "datasets/Images_fond_blanc_&_bruits"
OUTPUT_DIR = PROJECT_ROOT / "datasets/LPL_datasets_augment"

NUMBER_OF_IMAGES = 10
MIN_MACROINVERTEBRATES = 4
MAX_MACROINVERTEBRATES = 10
PADDING = 0.08
MAX_ROTATION = 180
ALLOW_OVERLAP = True
IOU_MAX = 1.0

CREATE_SPLITS = True
SPLIT_RATIOS = [70, 10, 20]
MULTI_CLASS = False
RANDOM_SEED = None

CREATE_VALIDATION = True
VALIDATION_COUNT = 10
VALIDATION_MAX_SIZE = 1600
PNG_COMPRESS_LEVEL = 0
OUTPUT_IMAGE_FORMAT = "png"

# Paramètres de segmentation SAM3 pour des images contenant plusieurs insectes.
SAM3_SOURCE_DIR = PROJECT_ROOT / "datasets/DATA/LPL_Photos_31072026"
SAM3_OUTPUT_DIR = PROJECT_ROOT / "datasets/LPL_dataset_cutouts_single"
SAM3_MODEL_ID = "facebook/sam3"
SAM3_PROMPT = "insect"
SAM3_THRESHOLD = 0.7
SAM3_DETECTION_MAX_SIZE = 3000
SAM3_CONTEXT_RATIO = 3.0
SAM3_REFINE_THRESHOLD = 0.3
SAM3_REFINE_MAX_SIZE = 1008

# Paramètres de segmentation SAM2 + Otsu pour des images déjà cropées.
SAM2_SOURCE_DIR = PROJECT_ROOT / "datasets/LPL_dataset_crop"
SAM2_OUTPUT_DIR = PROJECT_ROOT / "datasets/LPL_dataset_cutouts_sam2_otsu"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LOSSLESS_EXTENSIONS = {".png", ".tif", ".tiff"}


# -----------------------------------------------------------------------------
# Fonctions communes
# -----------------------------------------------------------------------------


def list_images(directory, extensions=IMAGE_EXTENSIONS):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {directory.resolve()}")

    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.suffix.lower() in extensions
    ]


def boxes_are_nested(first, second):
    def contains(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    return contains(first, second) or contains(second, first)


def box_iou(first, second):
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height

    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def split_counts(number_of_images, split_ratios):
    if len(split_ratios) != 3 or any(ratio < 0 for ratio in split_ratios):
        raise ValueError("split_ratios doit contenir trois valeurs positives.")

    total = sum(split_ratios)
    if total <= 0:
        raise ValueError("La somme de split_ratios doit être supérieure à zéro.")

    exact_counts = [number_of_images * ratio / total for ratio in split_ratios]
    counts = [int(value) for value in exact_counts]

    # Les images restantes vont aux splits ayant les plus grandes décimales.
    missing = number_of_images - sum(counts)
    order = sorted(
        range(3), key=lambda index: exact_counts[index] - counts[index], reverse=True
    )
    for index in order[:missing]:
        counts[index] += 1

    return counts


def write_yolo_yaml(dataset_dir, class_names, create_splits, filename="dataset.yaml"):
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images" if create_splits else "images",
        "val": "val/images" if create_splits else "images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if create_splits:
        data["test"] = "test/images"

    with (dataset_dir / filename).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def save_lossless_image(image, output_path, image_format, png_compress_level):
    if image_format == "png":
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(
            str(output_path),
            array,
            [cv2.IMWRITE_PNG_COMPRESSION, png_compress_level],
        )
        if not success:
            raise OSError(f"Impossible d'enregistrer {output_path}")
    else:
        image.save(output_path, format="TIFF", compression="raw")


def save_validation_preview(
    image,
    annotations,
    class_names,
    multi_class,
    output_path,
    max_size,
    png_compress_level,
):
    original_width, original_height = image.size
    scale = min(1, max_size / max(original_width, original_height))
    preview_width = max(1, int(original_width * scale))
    preview_height = max(1, int(original_height * scale))
    preview_array = cv2.resize(
        np.asarray(image),
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    preview = Image.fromarray(preview_array)
    scale_x = preview.width / original_width
    scale_y = preview.height / original_height
    draw = ImageDraw.Draw(preview)

    for class_id, (left, top, right, bottom) in annotations:
        shown_id = class_id if multi_class else 0
        box = (
            left * scale_x,
            top * scale_y,
            right * scale_x,
            bottom * scale_y,
        )
        draw.rectangle(box, outline="red", width=4)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{shown_id}: {class_names[shown_id]}",
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )

    preview.save(
        output_path, format="PNG", compress_level=png_compress_level
    )


def validate_image_label_pairs(
    dataset_dir, split_names, with_labels_multi, image_extension
):
    for split_name in split_names:
        base_dir = dataset_dir / split_name if split_name else dataset_dir
        image_stems = {
            path.stem
            for path in (base_dir / "images").glob(f"*{image_extension}")
        }
        label_stems = {path.stem for path in (base_dir / "labels").glob("*.txt")}

        if image_stems != label_stems:
            raise RuntimeError(f"Images et labels différents dans {base_dir}")

        if with_labels_multi:
            multi_stems = {
                path.stem for path in (base_dir / "labels_multi").glob("*.txt")
            }
            if image_stems != multi_stems:
                raise RuntimeError(f"Images et labels_multi différents dans {base_dir}")


# -----------------------------------------------------------------------------
# 1. Création du dataset augmenté final
# -----------------------------------------------------------------------------


def generate_augmented_dataset(
    cutout_dir,
    background_dir,
    output_dir,
    number_of_images=1000,
    min_macroinvertebrates=4,
    max_macroinvertebrates=10,
    padding=0.08,
    max_rotation=180,
    allow_overlap=True,
    iou_max=1.0,
    create_splits=True,
    split_ratios=(70, 10, 20),
    multi_class=False,
    create_validation=True,
    validation_count=10,
    validation_max_size=1600,
    png_compress_level=0,
    output_image_format="png",
    random_seed=None,
):
    """Crée un dataset YOLO à partir de fonds et de PNG transparents."""

    if number_of_images <= 0:
        raise ValueError("number_of_images doit être supérieur à zéro.")
    if not 0 <= padding < 0.5:
        raise ValueError("padding doit être compris entre 0 et 0.5.")
    if not 0 <= iou_max <= 1:
        raise ValueError("iou_max doit être compris entre 0 et 1.")
    if min_macroinvertebrates < 1 or max_macroinvertebrates < min_macroinvertebrates:
        raise ValueError("Vérifier min_macroinvertebrates et max_macroinvertebrates.")
    if validation_count < 0 or validation_max_size <= 0:
        raise ValueError("Les paramètres de validation doivent être positifs.")
    if not 0 <= png_compress_level <= 9:
        raise ValueError("png_compress_level doit être compris entre 0 et 9.")
    output_image_format = output_image_format.lower().lstrip(".")
    if output_image_format not in {"png", "tif", "tiff"}:
        raise ValueError("output_image_format doit être 'png' ou 'tiff'.")
    if output_image_format == "tif":
        output_image_format = "tiff"
    image_extension = f".{output_image_format}"

    rng = random.Random(random_seed)
    cutout_dir = Path(cutout_dir)
    background_dir = Path(background_dir)
    output_dir = Path(output_dir)

    backgrounds = list_images(background_dir, LOSSLESS_EXTENSIONS)
    insects = list_images(cutout_dir, {".png"})
    if not backgrounds:
        raise FileNotFoundError("Aucun fond PNG ou TIFF trouvé.")
    if not insects:
        raise FileNotFoundError("Aucun cutout PNG trouvé.")

    multi_names = sorted({path.parent.name for path in insects})
    class_to_id = {name: index for index, name in enumerate(multi_names)}
    active_names = multi_names if multi_class else ["macroinvertebrates"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_dir = output_dir / timestamp
    if dataset_dir.exists():
        dataset_dir = output_dir / f"{timestamp}_{time.time_ns() % 1_000_000_000:09d}"

    if create_splits:
        counts = split_counts(number_of_images, split_ratios)
        sample_splits = [
            split_name
            for split_name, count in zip(("train", "val", "test"), counts)
            for _ in range(count)
        ]
        rng.shuffle(sample_splits)
        split_names = ["train", "val", "test"]
    else:
        sample_splits = [None] * number_of_images
        split_names = [None]

    for split_name in split_names:
        base_dir = dataset_dir / split_name if split_name else dataset_dir
        (base_dir / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / "labels").mkdir(parents=True, exist_ok=True)
        (base_dir / "labels_multi").mkdir(parents=True, exist_ok=True)

    if create_validation:
        validation_dir = dataset_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        preview_count = min(10, validation_count, number_of_images)
        preview_indices = set(rng.sample(range(number_of_images), preview_count))
    else:
        preview_indices = set()

    progress = tqdm(total=number_of_images, desc="Création du dataset YOLO")
    sample_index = 0
    generation_attempts = 0

    while sample_index < number_of_images:
        generation_attempts += 1
        if generation_attempts > number_of_images * 30:
            raise RuntimeError(
                "Impossible de placer assez d'insectes. Réduire leur nombre, le padding "
                "ou les contraintes de superposition."
            )

        with Image.open(rng.choice(backgrounds)) as image:
            background = image.convert("RGB")

        padding_x = int(background.width * padding)
        padding_y = int(background.height * padding)
        target_count = rng.randint(min_macroinvertebrates, max_macroinvertebrates)
        annotations = []

        for _ in range(max(50, target_count * 20)):
            if len(annotations) == target_count:
                break

            insect_path = rng.choice(insects)
            with Image.open(insect_path) as image:
                insect = image.convert("RGBA")

            insect = insect.rotate(
                rng.uniform(-max_rotation, max_rotation),
                expand=True,
                resample=Image.Resampling.BICUBIC,
            )
            alpha_box = insect.getchannel("A").getbbox()
            if alpha_box is None:
                continue
            insect = insect.crop(alpha_box)

            max_x = background.width - padding_x - insect.width
            max_y = background.height - padding_y - insect.height
            if max_x < padding_x or max_y < padding_y:
                continue

            x = rng.randint(padding_x, max_x)
            y = rng.randint(padding_y, max_y)
            box = (x, y, x + insect.width, y + insect.height)
            overlap_limit = iou_max if allow_overlap else 0.0
            invalid_position = any(
                boxes_are_nested(box, other_box)
                or box_iou(box, other_box) > overlap_limit
                for _, other_box in annotations
            )
            if invalid_position:
                continue

            background.paste(insect, (x, y), insect)
            annotations.append((class_to_id[insect_path.parent.name], box))

        if len(annotations) < min_macroinvertebrates:
            continue

        split_name = sample_splits[sample_index]
        base_dir = dataset_dir / split_name if split_name else dataset_dir
        stem = f"sample_{sample_index + 1:06d}"
        image_path = base_dir / "images" / f"{stem}{image_extension}"

        single_labels = []
        multi_labels = []
        for class_id, (left, top, right, bottom) in annotations:
            center_x = (left + right) / 2 / background.width
            center_y = (top + bottom) / 2 / background.height
            width = (right - left) / background.width
            height = (bottom - top) / background.height
            coordinates = f"{center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            single_labels.append(f"0 {coordinates}")
            multi_labels.append(f"{class_id} {coordinates}")

        active_labels = multi_labels if multi_class else single_labels
        (base_dir / "labels" / f"{stem}.txt").write_text(
            "\n".join(active_labels) + "\n", encoding="utf-8"
        )
        (base_dir / "labels_multi" / f"{stem}.txt").write_text(
            "\n".join(multi_labels) + "\n", encoding="utf-8"
        )
        save_lossless_image(
            background, image_path, output_image_format, png_compress_level
        )

        if sample_index in preview_indices:
            save_validation_preview(
                background,
                annotations,
                active_names,
                multi_class,
                validation_dir / f"{stem}.png",
                validation_max_size,
                png_compress_level,
            )

        sample_index += 1
        progress.update(1)

    progress.close()

    classes = {
        "nc": len(active_names),
        "names": {index: name for index, name in enumerate(active_names)},
    }
    (dataset_dir / "classes.json").write_text(
        json.dumps(classes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    multi_classes = {
        "nc": len(multi_names),
        "names": {index: name for index, name in enumerate(multi_names)},
    }
    (dataset_dir / "classes_multi.json").write_text(
        json.dumps(multi_classes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    yaml_name = "dataset_multi.yaml" if multi_class else "dataset_1class.yaml"
    write_yolo_yaml(dataset_dir, active_names, create_splits, "dataset.yaml")
    write_yolo_yaml(dataset_dir, active_names, create_splits, yaml_name)

    validate_image_label_pairs(
        dataset_dir, split_names, with_labels_multi=True, image_extension=image_extension
    )
    print(f"Dataset YOLO enregistré dans : {dataset_dir.resolve()}")
    return dataset_dir


# -----------------------------------------------------------------------------
# 2. Segmentation SAM3 + GrabCut depuis des images globales
# -----------------------------------------------------------------------------


def refine_mask_full_resolution(image, mask):
    """Affine le bord du masque sur les pixels de l'image originale."""

    y, x = np.where(mask)
    if len(x) == 0:
        return mask

    padding = max(10, int(max(x.max() - x.min(), y.max() - y.min()) * 0.05))
    left = max(0, x.min() - padding)
    top = max(0, y.min() - padding)
    right = min(mask.shape[1], x.max() + padding + 1)
    bottom = min(mask.shape[0], y.max() + padding + 1)
    image_roi = image[top:bottom, left:right]
    mask_roi = mask[top:bottom, left:right].astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_roi, kernel, iterations=2)
    eroded = cv2.erode(mask_roi, kernel, iterations=2)
    grabcut_mask = np.full(mask_roi.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[dilated > 0] = cv2.GC_PR_BGD
    grabcut_mask[mask_roi > 0] = cv2.GC_PR_FGD
    grabcut_mask[eroded > 0] = cv2.GC_FGD

    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(image_roi, cv2.COLOR_RGB2BGR),
            grabcut_mask,
            None,
            background,
            foreground,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return mask

    refined_roi = np.isin(grabcut_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    intersection = np.logical_and(refined_roi, mask_roi).sum()
    union = np.logical_or(refined_roi, mask_roi).sum()
    area_ratio = refined_roi.sum() / max(1, mask_roi.sum())
    if intersection / max(1, union) < 0.5 or not 0.5 <= area_ratio <= 1.5:
        return mask

    refined = np.zeros_like(mask)
    refined[top:bottom, left:right] = refined_roi
    return refined


def move_sam3_outputs_to_cpu(outputs):
    outputs.pred_masks = outputs.pred_masks.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    outputs.pred_logits = outputs.pred_logits.cpu()
    if outputs.presence_logits is not None:
        outputs.presence_logits = outputs.presence_logits.cpu()


def segment_all_sam3(image, model, processor, device, prompt, threshold):
    import torch

    inputs = processor(
        images=Image.fromarray(image), text=prompt, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    move_sam3_outputs_to_cpu(outputs)
    return processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]


def refine_sam3_instance(
    global_mask,
    original_image,
    model,
    processor,
    device,
    context_ratio=3.0,
    refine_threshold=0.3,
    refine_max_size=1008,
    full_resolution=True,
):
    import torch

    original_height, original_width = original_image.shape[:2]
    reduced_height, reduced_width = global_mask.shape
    scale_x = original_width / reduced_width
    scale_y = original_height / reduced_height

    y, x = np.where(global_mask)
    left = int(x.min() * scale_x)
    top = int(y.min() * scale_y)
    right = min(original_width, int((x.max() + 1) * scale_x))
    bottom = min(original_height, int((y.max() + 1) * scale_y))

    margin_x = int((right - left) * context_ratio)
    margin_y = int((bottom - top) * context_ratio)
    crop_left = max(0, left - margin_x)
    crop_top = max(0, top - margin_y)
    crop_right = min(original_width, right + margin_x)
    crop_bottom = min(original_height, bottom + margin_y)
    image_crop = original_image[crop_top:crop_bottom, crop_left:crop_right]

    model_image = Image.fromarray(image_crop)
    model_image.thumbnail((refine_max_size, refine_max_size), Image.Resampling.LANCZOS)
    model_scale_x = model_image.width / image_crop.shape[1]
    model_scale_y = model_image.height / image_crop.shape[0]
    local_box = [
        [
            (left - crop_left) * model_scale_x,
            (top - crop_top) * model_scale_y,
            (right - crop_left) * model_scale_x,
            (bottom - crop_top) * model_scale_y,
        ]
    ]

    inputs = processor(
        images=model_image,
        input_boxes=[local_box],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    move_sam3_outputs_to_cpu(outputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=refine_threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    low_left = max(0, int(crop_left / scale_x))
    low_top = max(0, int(crop_top / scale_y))
    low_right = min(reduced_width, int(np.ceil(crop_right / scale_x)))
    low_bottom = min(reduced_height, int(np.ceil(crop_bottom / scale_y)))
    global_crop = global_mask[low_top:low_bottom, low_left:low_right]
    global_crop = np.asarray(
        Image.fromarray(global_crop).resize(model_image.size, Image.Resampling.NEAREST)
    ).astype(bool)
    fallback = np.asarray(
        Image.fromarray(global_crop).resize(
            (image_crop.shape[1], image_crop.shape[0]), Image.Resampling.NEAREST
        )
    ).astype(bool)
    if full_resolution:
        fallback = refine_mask_full_resolution(image_crop, fallback)

    candidates = [
        mask.numpy().astype(bool) for mask in result["masks"] if mask.any()
    ]
    global_area = max(1, global_crop.sum())
    candidates = [
        mask
        for mask in candidates
        if 0.5 <= mask.sum() / global_area <= 1.8
        and not (
            mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        )
    ]
    if not candidates:
        return image_crop, fallback

    ious = [
        np.logical_and(mask, global_crop).sum()
        / max(1, np.logical_or(mask, global_crop).sum())
        for mask in candidates
    ]
    best = int(np.argmax(ious))
    if ious[best] < 0.5:
        return image_crop, fallback

    refined_mask = np.asarray(
        Image.fromarray(candidates[best]).resize(
            (image_crop.shape[1], image_crop.shape[0]), Image.Resampling.NEAREST
        )
    ).astype(bool)
    if full_resolution:
        refined_mask = refine_mask_full_resolution(image_crop, refined_mask)
    return image_crop, refined_mask


def create_cutouts_sam3(
    source_dir,
    output_dir,
    model_id="facebook/sam3",
    prompt="insect",
    threshold=0.7,
    detection_max_size=3000,
    context_ratio=3.0,
    refine_threshold=0.3,
    refine_max_size=1008,
    full_resolution=True,
    class_name="macroinvertebrates",
):
    """Détecte tous les insectes des images globales et crée les PNG détourés."""

    import torch
    from transformers import Sam3Model, Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de SAM3 sur {device}...")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()

    source_dir = Path(source_dir)
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(source_dir)
    if not image_paths:
        raise FileNotFoundError("Aucune image à segmenter avec SAM3.")
    saved_count = 0

    for image_path in tqdm(image_paths, desc="Segmentation SAM3"):
        with Image.open(image_path) as image:
            original_pil = image.convert("RGB")
        original_image = np.asarray(original_pil)
        detection_image = original_pil.copy()
        detection_image.thumbnail(
            (detection_max_size, detection_max_size), Image.Resampling.LANCZOS
        )
        instances = segment_all_sam3(
            np.asarray(detection_image), model, processor, device, prompt, threshold
        )
        global_masks = instances["masks"].cpu()
        relative_name = "__".join(
            image_path.relative_to(source_dir).with_suffix("").parts
        )

        for index, global_mask in enumerate(global_masks, start=1):
            global_mask = global_mask.numpy().astype(bool)
            if not global_mask.any():
                continue

            image_crop, refined_mask = refine_sam3_instance(
                global_mask,
                original_image,
                model,
                processor,
                device,
                context_ratio,
                refine_threshold,
                refine_max_size,
                full_resolution,
            )
            if not refined_mask.any():
                continue

            y, x = np.where(refined_mask)
            image_crop = image_crop[y.min() : y.max() + 1, x.min() : x.max() + 1]
            refined_mask = refined_mask[y.min() : y.max() + 1, x.min() : x.max() + 1]
            rgba = np.dstack((image_crop, refined_mask.astype(np.uint8) * 255))
            Image.fromarray(rgba).save(
                class_dir / f"{relative_name}_{index:03d}.png",
                compress_level=PNG_COMPRESS_LEVEL,
            )
            saved_count += 1

        del instances, global_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{saved_count} cutout(s) SAM3 enregistré(s) dans : {class_dir.resolve()}")
    return Path(output_dir)


# -----------------------------------------------------------------------------
# 3. Segmentation SAM2 + Otsu depuis des crops
# -----------------------------------------------------------------------------


def mask_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mask > 0


def mask_sam2_from_otsu(image, model, processor, device):
    import torch

    otsu_mask = mask_otsu(image)
    inputs = processor(images=Image.fromarray(image), return_tensors="pt").to(device)
    prompt = torch.from_numpy(otsu_mask.astype(np.float32))[None, None].to(device)

    with torch.inference_mode():
        outputs = model(**inputs, input_masks=prompt, multimask_output=False)

    return processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0][0, 0].numpy().astype(bool)


def create_cutouts_sam2_otsu(
    source_dir,
    output_dir,
    model_id="facebook/sam2.1-hiera-tiny",
):
    """Segmente des crops avec Otsu utilisé comme prompt de SAM2."""

    import torch
    from transformers import Sam2Model, Sam2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de SAM2 sur {device}...")
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device).eval()

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    image_paths = list_images(source_dir)
    if not image_paths:
        raise FileNotFoundError("Aucune image à segmenter avec SAM2 + Otsu.")
    saved_count = 0

    for image_path in tqdm(image_paths, desc="Segmentation SAM2 + Otsu"):
        with Image.open(image_path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
        mask = mask_sam2_from_otsu(rgb_image, model, processor, device)
        if not mask.any():
            continue

        y, x = np.where(mask)
        rgba = np.dstack((rgb_image, mask.astype(np.uint8) * 255))
        rgba = rgba[y.min() : y.max() + 1, x.min() : x.max() + 1]
        output_path = output_dir / image_path.relative_to(source_dir).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(
            output_path, format="PNG", compress_level=PNG_COMPRESS_LEVEL
        )
        saved_count += 1

    print(
        f"{saved_count} cutout(s) SAM2 + Otsu enregistré(s) dans : "
        f"{output_dir.resolve()}"
    )
    return output_dir


# -----------------------------------------------------------------------------
# Exécution du script
# -----------------------------------------------------------------------------


def main():
    # Fonction finale utilisée par défaut.
    generate_augmented_dataset(
        cutout_dir=CUTOUT_DIR,
        background_dir=BACKGROUND_DIR,
        output_dir=OUTPUT_DIR,
        number_of_images=NUMBER_OF_IMAGES,
        min_macroinvertebrates=MIN_MACROINVERTEBRATES,
        max_macroinvertebrates=MAX_MACROINVERTEBRATES,
        padding=PADDING,
        max_rotation=MAX_ROTATION,
        allow_overlap=ALLOW_OVERLAP,
        iou_max=IOU_MAX,
        create_splits=CREATE_SPLITS,
        split_ratios=SPLIT_RATIOS,
        multi_class=MULTI_CLASS,
        create_validation=CREATE_VALIDATION,
        validation_count=VALIDATION_COUNT,
        validation_max_size=VALIDATION_MAX_SIZE,
        png_compress_level=PNG_COMPRESS_LEVEL,
        output_image_format=OUTPUT_IMAGE_FORMAT,
        random_seed=RANDOM_SEED,
    )

    # Décommenter pour segmenter des images globales avec SAM3 + GrabCut.
    # create_cutouts_sam3(
    #     source_dir=SAM3_SOURCE_DIR,
    #     output_dir=SAM3_OUTPUT_DIR,
    #     model_id=SAM3_MODEL_ID,
    #     prompt=SAM3_PROMPT,
    #     threshold=SAM3_THRESHOLD,
    #     detection_max_size=SAM3_DETECTION_MAX_SIZE,
    #     context_ratio=SAM3_CONTEXT_RATIO,
    #     refine_threshold=SAM3_REFINE_THRESHOLD,
    #     refine_max_size=SAM3_REFINE_MAX_SIZE,
    # )

    # Décommenter pour segmenter des crops avec SAM2 guidé par Otsu.
    # create_cutouts_sam2_otsu(
    #     source_dir=SAM2_SOURCE_DIR,
    #     output_dir=SAM2_OUTPUT_DIR,
    #     model_id=SAM2_MODEL_ID,
    # )


if __name__ == "__main__":
    main()
