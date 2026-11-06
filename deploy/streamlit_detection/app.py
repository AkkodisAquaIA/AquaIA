from __future__ import annotations

import csv
import io
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
from ultralytics import YOLO

from core import (
    filter_models,
    image_detection_metrics,
    load_catalog,
    load_metrics,
    parse_yolo_labels,
    summarize_training_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = Path(__file__).with_name("model_catalog.yaml")
PREVIEW_MAX_SIZE = (1600, 1600)
SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@st.cache_resource(show_spinner=False)
def load_model(checkpoint: str, device: str) -> YOLO:
    model = YOLO(checkpoint)
    model.to(device)
    return model


def make_preview_and_crops(image_path: Path, detections: list[dict]) -> tuple[bytes, list[dict], dict]:
    with Image.open(image_path) as source:
        metadata = {
            "Largeur originale": source.width,
            "Hauteur originale": source.height,
            "Mode": source.mode,
            "Nombre de pages": int(getattr(source, "n_frames", 1)),
        }
        original = source.convert("RGB")

    crops = []
    for detection_index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        crop_box = (
            max(0, min(original.width, int(np.floor(x1)))),
            max(0, min(original.height, int(np.floor(y1)))),
            max(0, min(original.width, int(np.ceil(x2)))),
            max(0, min(original.height, int(np.ceil(y2)))),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        crop = original.crop(crop_box)
        crop_buffer = io.BytesIO()
        crop.save(crop_buffer, format="PNG", optimize=True)
        crops.append(
            {
                "index": detection_index,
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "png": crop_buffer.getvalue(),
                "filename": f"detection_{detection_index:03d}_{detection['confidence']:.3f}.png",
            }
        )

    preview = original
    original_width, original_height = original.size
    preview.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
    scale_x = preview.width / original_width
    scale_y = preview.height / original_height
    draw = ImageDraw.Draw(preview)
    line_width = max(2, round(max(preview.size) / 500))

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        scaled_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        draw.rectangle(scaled_box, outline=(255, 70, 50), width=line_width)
        text = f"{detection['class_name']} {detection['confidence']:.2f}"
        text_box = draw.textbbox((scaled_box[0], scaled_box[1]), text)
        draw.rectangle(text_box, fill=(255, 70, 50))
        draw.text((scaled_box[0], scaled_box[1]), text, fill="white")

    output = io.BytesIO()
    preview.save(output, format="PNG", optimize=True)
    metadata["Aperçu affiché"] = f"{preview.width} × {preview.height}"
    return output.getvalue(), crops, metadata


def crops_to_zip(crops: list[dict]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for crop in crops:
            zip_file.writestr(crop["filename"], crop["png"])
    return archive.getvalue()


def detections_to_csv(detections: list[dict]) -> bytes:
    text = io.StringIO()
    writer = csv.DictWriter(
        text,
        fieldnames=["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
    )
    writer.writeheader()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        writer.writerow(
            {
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return text.getvalue().encode("utf-8")


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    temporary_file = tempfile.NamedTemporaryFile(prefix="aquaia_", suffix=suffix, delete=False)
    try:
        temporary_file.write(uploaded_file.getbuffer())
    finally:
        temporary_file.close()
    return Path(temporary_file.name)


def format_training_summary(summary: dict) -> dict:
    return {
        "Meilleure époque": summary["Meilleure époque"],
        "Val précision": f"{summary['Val précision']:.3f}",
        "Val rappel": f"{summary['Val rappel']:.3f}",
        "Val mAP50": f"{summary['Val mAP50']:.3f}",
        "Val mAP50-95": f"{summary['Val mAP50-95']:.3f}",
    }


def format_test_summary(metrics: dict) -> dict:
    mapping = {
        "test_precision": "Test précision",
        "test_recall": "Test rappel",
        "test_f1": "Test F1",
        "test_map_50": "Test mAP50",
        "test_map_50_95": "Test mAP50-95",
    }
    return {label: f"{float(metrics[key]):.3f}" for key, label in mapping.items() if key in metrics}


st.set_page_config(page_title="AquaIA - Détection", page_icon="🔬", layout="wide")
st.title("AquaIA - Prototype de détection")
st.caption("Sélection d’un modèle expérimental, inférence sur une image et traçabilité des performances.")

models = load_catalog(CATALOG_PATH, PROJECT_ROOT)

with st.sidebar:
    st.header("1. Sélection du modèle")
    architecture = st.selectbox("Architecture", ["yolo11n", "yolo26n"], format_func=str.upper)
    imgsz = st.selectbox("Résolution d’inférence", [640, 1024, 2048])
    roboflow_prefinetuning = st.checkbox("Pré-fine-tuning Roboflow")
    lpl_finetuning = st.checkbox("Fine-tuning LPL", value=True)
    yolo_augmentation = st.checkbox("Augmentation YOLO")
    copy_paste = st.checkbox("Augmentation Copy & Paste", value=True)

    choices = {
        "architecture": architecture,
        "imgsz": imgsz,
        "roboflow_prefinetuning": roboflow_prefinetuning,
        "lpl_finetuning": lpl_finetuning,
        "yolo_augmentation": yolo_augmentation,
        "copy_paste": copy_paste,
    }
    matching_models = filter_models(models, choices)

    if matching_models:
        selected_model = st.selectbox(
            "Checkpoint correspondant",
            matching_models,
            format_func=lambda model: model["name"],
        )
    else:
        selected_model = None
        st.warning("Aucun checkpoint du catalogue ne correspond exactement à ces options.")

    st.header("2. Paramètres d’inférence")
    confidence_threshold = st.slider("Seuil de confiance", 0.01, 0.95, 0.25, 0.01)
    nms_iou = st.slider("Seuil IoU de la NMS", 0.10, 0.90, 0.50, 0.05)
    evaluation_iou = st.slider("IoU pour l’évaluation annotée", 0.10, 0.90, 0.50, 0.05)

if selected_model is not None:
    checkpoint = Path(selected_model["checkpoint"])
    if not checkpoint.is_file():
        st.error(f"Checkpoint introuvable : {checkpoint}")
        st.stop()

    st.subheader(selected_model["name"])
    st.caption(str(checkpoint.relative_to(PROJECT_ROOT) if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint))

    training_summary = summarize_training_run(selected_model.get("run_dir"))
    test_summary = load_metrics(selected_model.get("test_metrics"))
    summary_columns = st.columns(2)
    with summary_columns[0]:
        st.markdown("**Performances pendant l’entraînement**")
        if training_summary:
            st.dataframe([format_training_summary(training_summary)], hide_index=True, width="stretch")
        else:
            st.info("Ce modèle préentraîné ne possède pas de run de fine-tuning local.")
    with summary_columns[1]:
        st.markdown("**Évaluation enregistrée sur le test LPL**")
        if test_summary:
            st.dataframe([format_test_summary(test_summary)], hide_index=True, width="stretch")
        else:
            st.info("Aucune évaluation LPL n’est référencée pour ce checkpoint.")

st.divider()
st.header("3. Image à analyser")

source_mode = st.radio(
    "Source de l’image",
    ["Téléverser une image", "Chemin local sur la machine"],
    horizontal=True,
)

uploaded_image = None
local_path_text = ""
if source_mode == "Téléverser une image":
    uploaded_image = st.file_uploader(
        "Image TIF, TIFF, PNG ou JPEG (maximum configuré : 1 Go)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
    )
    if uploaded_image is not None:
        st.caption(f"{uploaded_image.name} - {uploaded_image.size / (1024 ** 2):.1f} Mo")
else:
    local_path_text = st.text_input(
        "Chemin absolu ou relatif au projet",
        placeholder="datasets/LPL_dataset_final/dataset/images/test/mon_image.TIF",
    )
    st.caption("Ce mode évite de transférer le fichier quand Streamlit tourne sur la même machine.")

annotation_file = st.file_uploader(
    "Annotation YOLO facultative (.txt)",
    type=["txt"],
    help="Avec une annotation, l’interface calcule précision, rappel et F1 pour cette image au seuil IoU choisi.",
)

can_run = selected_model is not None and (uploaded_image is not None or bool(local_path_text.strip()))
if st.button("Lancer la détection", type="primary", disabled=not can_run):
    temporary_path = None
    try:
        if uploaded_image is not None:
            image_path = save_upload(uploaded_image)
            temporary_path = image_path
            display_name = uploaded_image.name
        else:
            candidate = Path(local_path_text.strip())
            image_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
            image_path = image_path.resolve()
            display_name = image_path.name

        if not image_path.is_file():
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Format non pris en charge : {image_path.suffix}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with st.spinner(f"Chargement du modèle et inférence sur {device}..."):
            model = load_model(str(checkpoint), device)
            started_at = time.perf_counter()
            result = model.predict(
                source=str(image_path),
                imgsz=int(selected_model["imgsz"]),
                conf=confidence_threshold,
                iou=nms_iou,
                device=device,
                verbose=False,
            )[0]
            elapsed_seconds = time.perf_counter() - started_at

        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            class_names = result.names
            for label, score, box in zip(labels, scores, boxes):
                detections.append(
                    {
                        "class_id": int(label),
                        "class_name": str(class_names.get(int(label), label)),
                        "confidence": float(score),
                        "box": [float(value) for value in box],
                    }
                )
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        preview_png, crops, image_metadata = make_preview_and_crops(image_path, detections)
        image_metrics = {
            "Détections": len(detections),
            "Confiance moyenne": float(scores.mean()) if len(scores) else 0.0,
            "Confiance maximale": float(scores.max()) if len(scores) else 0.0,
            "Temps total (s)": elapsed_seconds,
            "Inférence Ultralytics (ms)": float(result.speed.get("inference", 0.0)),
            "Appareil": device,
        }

        ground_truth_metrics = None
        if annotation_file is not None:
            target_labels, target_boxes = parse_yolo_labels(
                annotation_file.getvalue().decode("utf-8"),
                image_width=int(result.orig_shape[1]),
                image_height=int(result.orig_shape[0]),
            )
            ground_truth_metrics = image_detection_metrics(
                predicted_labels=labels,
                predicted_boxes=boxes,
                predicted_scores=scores,
                target_labels=target_labels,
                target_boxes=target_boxes,
                iou_threshold=evaluation_iou,
            )

        st.session_state["detection_result"] = {
            "name": display_name,
            "preview": preview_png,
            "crops": crops,
            "crops_zip": crops_to_zip(crops),
            "detections_csv": detections_to_csv(detections),
            "image_metadata": image_metadata,
            "image_metrics": image_metrics,
            "ground_truth_metrics": ground_truth_metrics,
        }
    except Exception as error:
        st.exception(error)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

if "detection_result" in st.session_state:
    output = st.session_state["detection_result"]
    st.divider()
    st.header("4. Résultat")
    st.image(output["preview"], caption=output["name"], width="stretch")

    first_column, second_column = st.columns(2)
    with first_column:
        st.markdown("**Informations sur l’image**")
        st.dataframe([output["image_metadata"]], hide_index=True, width="stretch")
    with second_column:
        st.markdown("**Statistiques des prédictions sur cette image**")
        display_metrics = dict(output["image_metrics"])
        for key in ("Confiance moyenne", "Confiance maximale", "Temps total (s)", "Inférence Ultralytics (ms)"):
            display_metrics[key] = f"{display_metrics[key]:.3f}"
        st.dataframe([display_metrics], hide_index=True, width="stretch")

    if output["ground_truth_metrics"] is not None:
        st.markdown("**Évaluation par rapport à l’annotation fournie**")
        ground_truth_metrics = dict(output["ground_truth_metrics"])
        for key in ("Précision image", "Rappel image", "F1 image", "IoU moyen des correspondances"):
            ground_truth_metrics[key] = f"{ground_truth_metrics[key]:.3f}"
        st.dataframe([ground_truth_metrics], hide_index=True, width="stretch")
    else:
        st.info(
            "Sans annotation de référence, précision, rappel et mAP ne peuvent pas être calculés. "
            "Le tableau présente donc uniquement des statistiques descriptives des prédictions."
        )

    st.subheader("Détections découpées")
    if output["crops"]:
        crop_columns = st.columns(4)
        for crop_index, crop in enumerate(output["crops"]):
            with crop_columns[crop_index % len(crop_columns)]:
                st.markdown(f"**Détection {crop['index']} - confiance {crop['confidence']:.1%}**")
                st.image(crop["png"], width="stretch")
                st.caption(crop["class_name"])
    else:
        st.warning("Aucune détection ne dépasse le seuil de confiance sélectionné.")

    download_columns = st.columns(3)
    with download_columns[0]:
        st.download_button(
            "Télécharger l’aperçu annoté",
            data=output["preview"],
            file_name=f"{Path(output['name']).stem}_predictions.png",
            mime="image/png",
            width="stretch",
        )
    with download_columns[1]:
        st.download_button(
            "Télécharger les détections CSV",
            data=output["detections_csv"],
            file_name=f"{Path(output['name']).stem}_detections.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_columns[2]:
        st.download_button(
            "Télécharger tous les crops",
            data=output["crops_zip"],
            file_name=f"{Path(output['name']).stem}_crops.zip",
            mime="application/zip",
            width="stretch",
            disabled=not output["crops"],
        )
