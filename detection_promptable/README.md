## detection_promptable

This folder provides a simple promptable detection workflow:

1. Define the classes you want to detect in `dataset_dict.py` (COCO128 is provided as an example).
2. Configure the model and inference settings in `model_cfg.py`.
3. Run detection with `detection_entry.py`. A custom (non-native) per-class NMS is optionally applied.
4. Evaluate results with `metric.py`.

## 1. Configure classes (dataset_dict.py)

`DATASET_DICT` maps class IDs to class names. The keys are the class IDs used in labels.

- Example: COCO128 uses IDs 0–79.
- The order of sorted keys is used to build the text prompts passed into the model.

Update this file to match your dataset:

- Add or remove classes.
- Use the correct class IDs.

## 2. Configure model and inputs (model_cfg.py)

Key settings:

- `IMAGES_FOLDER`: folder with images to run inference on.
- `MODEL_NAME`: `"sam3"` or `"yoloe26"`.
- `MODEL_CFG`: per-model settings, including `CONF`, `PATH`, `IMGSZ`, `HALF`, and `NMS`.

`NMS` in this project is **not** the native model NMS. It is a custom per-class NMS applied after inference:

- Set `NMS` to a float IoU threshold (e.g., `0.7`) to enable.
- Set `NMS` to `False` to disable.

## 3. Run detection (detection_entry.py)

`detection_entry.py`:

- Builds text prompts from `DATASET_DICT`.
- Runs the selected model on all images under `IMAGES_FOLDER`.
- Optionally applies the custom NMS.
- Saves labels to `detection_promptable/[MODEL]_result_det_YYYYMMDDHHmm/labels`.
- Writes a `cfg.txt` with the exact run configuration.

Label format (per image, normalized):

```
cls cx cy w h conf
```

If no detections are found, an empty `.txt` file is created for that image.

## 4. Evaluate metrics (metric.py)

`metric.py` compares predictions to ground-truth labels and outputs:

- `mAP50-95(B)`, `mAP50(B)`, `precision(B)`, `recall(B)`, and number of images.

Notes:

- It evaluates only the **intersection** of filenames in GT and prediction folders.
- It uses normalized coordinates, so no image size is required.
- You can choose a result folder via GUI, or it will use the latest one automatically.
- It can apply an additional confidence threshold on detections.

Output:

- Metrics are printed to console.
- Metrics are saved to `metrics[_confXXX].txt` inside the selected result folder.