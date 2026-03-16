## detection_promptable
This folder provides a promptable detection workflow:
1. Define the classes you want to detect in `dataset_dict.yaml` (COCO128 is provided as an example).
2. Configure the model and inference settings in `model_cfg.yaml`.
3. Run detection with `detection_entry.py`. A custom non-native per-class NMS is optionally applied.
4. Crop detected boxes with `crop.py`.
5. Count empty detections with `count_empty.py`.
6. Evaluate results with `metric.py`.

## 1. Configure classes (dataset_dict.yaml)
Maps class IDs to class names. The keys are the class IDs used in labels.
- Example: COCO128 uses IDs 0-79.
- The order of sorted keys is used to build the text prompts passed into the model.

Update this file to match your dataset:
- Add or remove classes.
- Use the correct class IDs.

## 2. Configure model and inputs (model_cfg.yaml)
Key settings:
- `IMAGES_FOLDER`: folder with images to run inference on.
- `MODEL_NAME`: `"sam3"` or `"yoloe26"`.
- `MODEL_CFG`: per-model settings.
- `sam3`: `CONF`, `TASK`, `MODE`, `PATH`, `HALF`, `SAVE`, `IMGSZ`, `NMS`, `UNIC`.
- `yoloe26`: `CONF`, `PATH`, `HALF`, `SAVE`, `IMGSZ`, `BATCH`, `NMS`, `UNIC`.

Notes:
- The config is loaded from YAML by both `detection_entry.py` and `metric.py`.
- `NMS` in this project is **not** the native model NMS. It is a custom per-class NMS applied after inference:
- Set `NMS` to a float IoU threshold (e.g., `0.7`) to enable.
- Set `NMS` to `False` to disable.
- `UNIC` keeps only one bbox per image after all other post-processing:
- Set `UNIC` to `True` to keep only the bbox with the highest `conf`.
- Set `UNIC` to `False` to keep the original behavior.

## 3. Run detection (detection_entry.py)
- Builds text prompts from `DATASET_DICT`.
- Runs the selected model on all images under `IMAGES_FOLDER`.
- Optionally applies the custom NMS.
- Optionally applies `UNIC` after NMS and keeps only the highest-confidence bbox for each image.
- Saves labels under `detection_promptable/[MODEL]_result_det_YYYYMMDDHHmm/.../labels`.
- If images are directly under `IMAGES_FOLDER`, labels are in `.../labels`.
- If images are in subfolders, output mirrors subfolder structure (each subfolder has its own `labels` folder).
- Writes a `cfg.txt` with the exact run configuration.

Notes:
- Label format (per image, normalized): cls cx cy w h conf
- If no detections are found, an empty `.txt` file is created for that image.
- Inference terminal logs are generated before custom post-NMS filtering.
- Saved labels and visualizations are generated after custom post-processing (`NMS`, then `UNIC` if enabled).

## 4. Crop detections (crop.py)
`crop.py` reads detection labels and writes cropped image patches.
- You can choose a result folder via GUI, or it automatically uses the latest `*_result_det_YYYYMMDDHHmm` folder.
- It reads labels from `<result_dir>/<relative_image_folder>/labels/<image_name>.txt`.
- It saves crops under `<result_dir>/00crop/<relative_image_folder>/`.
- Crop filename format is `<image_stem>_<class_name>_<index>.ext`.
- Confidence is ignored during cropping (boxes are read from the first 5 label columns: `cls cx cy w h`).

## 5. Count empty detections (count_empty.py)
`count_empty.py` checks how many images have empty detection labels (no boxes).

Notes:
- It scans images recursively from `IMAGES_FOLDER`.
- You can choose a result folder via GUI, or it automatically uses the latest `*_result_det_YYYYMMDDHHmm` folder.
- For each image, it reads prediction labels from `<result_dir>/<relative_image_folder>/labels/<image_name>.txt`.
- Empty label files are counted as empties.
- Missing label files are reported as warnings and skipped.

Output:
- A summary file `empty_detection.txt` is saved inside the selected result folder.
- Format includes per-subfolder counts: `Subfolder, Empaty, Total`, plus a `TOTAL` line.

## 6. Evaluate metrics (metric.py)
`metric.py` compares predictions to ground-truth labels and outputs:
- `mAP50-95(B)`, `mAP50(B)`, `precision(B)`, `recall(B)`, and number of images.

Notes:
- It evaluates only the **intersection** of filenames in GT and prediction folders.
- GT folder is auto-derived as `IMAGES_FOLDER` with `"images"` replaced by `"labels"`.
- It uses normalized coordinates, so no image size is required.
- You can choose a result folder via GUI, or it will use the latest one automatically (`*_result_det_YYYYMMDDHHmm`).
- It can apply an additional confidence threshold on detections (blank input = no extra threshold).
- Current evaluation reads only `det_dir/labels/*.txt` and is **not recursive**.
- If your predictions are in nested folders (mirrored structure from `detection_entry.py`), aggregate them first or adapt `metric.py`.

Output:
- Metrics are printed to console.
- Metrics are saved to `metrics[_confXXX].txt` inside the selected result folder.