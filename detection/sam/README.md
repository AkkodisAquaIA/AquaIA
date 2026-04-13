## detection_promptable
This folder provides a promptable detection workflow:
1. Define the classes you want to detect in `dataset_dict.yaml` (COCO128 is provided as an example).
2. Configure the model and inference settings in `model_cfg.yaml`.
3. Run detection with `detection_entry.py`. A custom non-native per-class NMS is optionally applied.
4. Crop detected boxes with `crop.py`.
5. Count no detection images with `count_nodet.py`.
6. Compare crop outputs with `check_image.py`.
7. Evaluate results with `metric.py`.

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
- Copies `model_cfg.yaml` and `dataset_dict.yaml` into the result folder for traceability.

Notes:
- Label format (per image, normalized): cls cx cy w h conf
- If no detections are found, an empty `.txt` file is created for that image.
- Inference terminal logs are generated before custom post-processing (NMS or UNIC).
- Saved labels and visualizations are generated after custom post-processing (`NMS`, then `UNIC` if enabled).

## 4. Crop detections (crop.py)
`crop.py` reads detection labels and writes cropped image patches.
- You can choose a result folder via GUI, or it automatically uses the latest `*_result_det_YYYYMMDDHHmm` folder.
- It reads `model_cfg.yaml` and `dataset_dict.yaml` from the selected result folder when available, so cropping stays aligned with that run.
- It reads labels from `<result_dir>/<relative_image_folder>/labels/<image_name>.txt`.
- It saves crops under `<result_dir>/00crop/<relative_image_folder>/`.
- Crop filename format is `<image_stem>_<class_name>_<index>.ext`.
- Confidence is ignored during cropping (boxes are read from the first 5 label columns: `cls cx cy w h`).

## 5. Count no detection images (count_nodet.py)
`count_nodet.py` checks how many images have 0 detection labels (no boxes).

Notes:
- It scans images recursively from the `IMAGES_FOLDER` stored in the selected result folder's `model_cfg.yaml` when available.
- You can choose a result folder via GUI, or it automatically uses the latest `*_result_det_YYYYMMDDHHmm` folder.
- For each image, it reads prediction labels from `<result_dir>/<relative_image_folder>/labels/<image_name>.txt`.
- Empty label files are counted as no detection images.
- Missing label files are reported as warnings and skipped.

Output:
- A summary file `no_detection.txt` is saved inside the selected result folder.
- Format includes per-subfolder counts: `Subfolder, No detection, Total, Crop boxes`, plus a `TOTAL` line.

## 6. Compare crop outputs (check_image.py)
`check_image.py` compares the cropped detection images in two result folders and reports what is extra or missing relative to a reference run.

Notes:
- You first select a `REFERENCE` `*_result_det_*` folder, then a `CURRENT` one. If no manual selection is made, the helper can fall back to the latest result folder.
- The script compares files inside `00crop/`.
- Comparison is content-based and groups images by:
    - crop subfolder path under `00crop`
    - original image stem extracted from `<image_stem>_<class_name>_<index>`
    - file extension
    - SHA256 hash of the crop content
- This means crops with the same source image stem but different content are treated as different images.

Output:
- A folder `<current_result_dir>/00check_image/` is created.
- `ref+` contains crops present in `CURRENT` but not matched in `REFERENCE`.
- `ref-` contains crops present in `REFERENCE` but not matched in `CURRENT`.
- Original directory structure under `00crop` is preserved in both output folders.
- A summary file `chek_image.txt` is saved in `00check_image/`.

## 7. Evaluate metrics (metric.py)
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