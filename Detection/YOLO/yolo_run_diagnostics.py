from __future__ import annotations

import argparse
import csv
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.results import Results


import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from ultralytics import YOLO


def load_yolo_labels(txt_path: Path) -> np.ndarray:
	"""Load a YOLO label file as an (N, 5) float32 array.

	Each valid row is expected to follow the format:
	class_id center_x center_y width height

	Missing or empty files return an empty array with shape (0, 5).
	Invalid lines are ignored.
	"""

	if not txt_path.is_file():
		return np.zeros((0, 5), dtype=np.float32)
	text = txt_path.read_text().strip()
	if not text:
		return np.zeros((0, 5), dtype=np.float32)

	rows = []
	for line in text.splitlines():
		parts = line.strip().split()
		if len(parts) != 5:
			continue
		cls, cx, cy, w, h = map(float, parts)
		rows.append([cls, cx, cy, w, h])
	return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def yolo_norm_to_xyxy(labels: np.ndarray, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
	"""Convert normalized YOLO labels to absolute xyxy boxes and class ids."""

	if labels.size == 0:
		return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
	cls_ids = labels[:, 0].astype(np.int64)
	cx, cy, w, h = (
		labels[:, 1] * img_w,
		labels[:, 2] * img_h,
		labels[:, 3] * img_w,
		labels[:, 4] * img_h,
	)
	boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
	return boxes, cls_ids


def _derive_labels_dir_from_images_dir(images_dir: Path) -> Path:
	"""Derive the matching YOLO labels directory from an images directory."""
	parts = list(images_dir.resolve().parts)
	if "images" in parts:
		idx = len(parts) - 1 - parts[::-1].index("images")
		parts[idx] = "labels"
		return Path(*parts)

	# Fallback for custom dataset layouts that do not contain a literal "images" directory.
	return images_dir.parent / "labels" / images_dir.name


def match_iou_pairs(iou_mat: np.ndarray) -> list[tuple[int, int, float]]:
	"""Return the globally optimal one-to-one assignment using Hungarian matching."""
	if iou_mat.size == 0:
		return []

	# Use a global one-to-one assignment so each prediction and GT box is matched at most once.
	cost = 1.0 - np.clip(iou_mat, 0.0, 1.0)
	row_ind, col_ind = linear_sum_assignment(cost)
	return [(int(i), int(j), float(iou_mat[i, j])) for i, j in zip(row_ind, col_ind)]


def greedy_match_iou(iou_mat: np.ndarray, iou_thr: float) -> tuple[int, int, float]:
	"""Match predictions to GT boxes with a global one-to-one IoU assignment."""
	if iou_mat.size == 0:
		return 0, 0, 0.0

	matched_ious = [iou for _, _, iou in match_iou_pairs(iou_mat) if iou >= iou_thr]
	if not matched_ious:
		return 0, 0, 0.0

	mean_iou = float(np.mean(matched_ious))
	return len(matched_ious), len(matched_ious), mean_iou


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
	"""Compute the pairwise IoU matrix between two sets of xyxy boxes."""

	if boxes1.size == 0 or boxes2.size == 0:
		return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
	x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
	y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
	x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
	y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

	inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
	area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
	area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
	union = np.clip(area1[:, None] + area2[None, :] - inter, 1e-6, None)
	return inter / union


def list_image_files(images_dir: Path) -> List[Path]:
	"""Return all supported image files in a directory, sorted and deduplicated."""

	patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
	files = []
	for pattern in patterns:
		files.extend(images_dir.glob(pattern))
	return sorted(set(files))


def categorize_image(
	pred_boxes: np.ndarray,
	pred_conf: np.ndarray,
	gt_boxes: np.ndarray,
	conf_thr: float,
	iou_thr: float,
) -> tuple[str, float]:
	"""Assign an image-level detection category based on GT coverage.

	Categories:
	- good: all ground-truth boxes are matched
	- missed: at least one ground-truth box is unmatched
	- fp_only: no GT box exists, but predictions are present
	- empty: no GT box and no prediction above threshold

	Returns the category and the mean IoU of matched pairs.
	"""
	keep = pred_conf >= conf_thr
	pred_boxes_f = pred_boxes[keep]

	has_gt = gt_boxes.size > 0
	has_pred = pred_boxes_f.size > 0

	if not has_gt and not has_pred:
		return "empty", 0.0
	if not has_gt and has_pred:
		return "fp_only", 0.0
	if has_gt and not has_pred:
		return "missed", 0.0

	iou_mat = compute_iou_matrix(pred_boxes_f, gt_boxes)
	_, n_gt_matched, mean_iou = greedy_match_iou(iou_mat, iou_thr=iou_thr)

	# Mark the image as good only if every GT box is matched.
	if n_gt_matched == gt_boxes.shape[0]:
		return "good", mean_iou
	return "missed", mean_iou


def collect_iou_stats_for_result(
	r: Results,
	labels_root: Path,
	labels_folder: str,
	ious_all: List[float],
	bad_examples: List[Tuple[float, np.ndarray, str]],
	confs_all: List[np.ndarray],
	low_iou_thresh: float = 0.5,
	conf_thres_iou: float = 0.25,
) -> None:
	"""Collect IoU and confidence diagnostics from one Ultralytics result.

	Predictions are filtered by confidence, then matched to ground-truth boxes using
	an optimal one-to-one IoU assignment. The matched IoUs are appended to ious_all.
	Low-IoU cases can be stored in bad_examples, together with the source filename,
	for later visualization.
	"""

	img = r.orig_img
	if img is None:
		return

	h, w = img.shape[:2]
	img_path = Path(r.path) if r.path else None
	if img_path is None:
		return

	label_path = labels_root / labels_folder / f"{img_path.stem}.txt"
	labels = load_yolo_labels(label_path)
	gt_boxes, _ = yolo_norm_to_xyxy(labels, w, h)

	if r.boxes is None or r.boxes.xyxy is None:
		return

	pred_boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
	pred_conf_arr = r.boxes.conf.cpu().numpy().astype(np.float32)

	if pred_conf_arr.size > 0:
		confs_all.append(pred_conf_arr)

	if gt_boxes.size == 0 or pred_boxes.size == 0:
		return

	keep = pred_conf_arr >= conf_thres_iou
	pred_boxes_f = pred_boxes[keep]
	if pred_boxes_f.size == 0:
		return

	iou_mat = compute_iou_matrix(pred_boxes_f, gt_boxes)
	matched_pairs = match_iou_pairs(iou_mat)
	if not matched_pairs:
		return

	# Keep all matched IoUs for error inspection, but only keep IoUs above the threshold for aggregate stats.
	all_matched_ious = [iou for _, _, iou in matched_pairs]
	matched_ious = [iou for iou in all_matched_ious if iou >= low_iou_thresh]
	if matched_ious:
		ious_all.extend(matched_ious)

	min_iou = min(all_matched_ious)
	if min_iou < low_iou_thresh and len(bad_examples) < 256:
		img_plot = r.plot()
		bad_img = img_plot if img_plot is not None else img
		bad_examples.append((min_iou, bad_img, img_path.name))


def _infer_model_tag_from_path(p: Path) -> str:
	"""Infer a compact model tag from a weights path."""
	s = str(p).lower()
	# Match common YOLO model names such as yolo11n or yolov8m.
	m = re.search(r"(yolo(?:v)?\d+[a-z]+)", s)
	if m:
		return m.group(1)
	# Fall back to the run folder name or the file stem when no model tag is found.
	return p.parent.parent.name.lower() if p.parent and p.parent.parent else p.stem.lower()


def _infer_init_tag_from_path(p: Path) -> str:
	"""Infer whether the run started from pretrained, scratch, or custom weights."""
	s = str(p).lower()
	if "pretrained" in s:
		return "pretrained"
	if "random" in s or "scratch" in s or "fromscratch" in s:
		return "scratch"
	return "custom"


def _infer_epoch_from_path_or_step(p: Path, global_step: int) -> int | None:
	"""Infer an epoch number from the path name, then fall back to global_step."""
	s = str(p).lower()
	m = re.search(r"(?:^|_)e(\d+)(?:_|$)", s)
	if m:
		try:
			return int(m.group(1))
		except Exception as exc:
			logging.debug("Failed to infer epoch from path %s: %s", p, exc)
	if global_step > 0:
		return int(global_step)
	return None


def infer_final_epoch(run_dir: Path, weights_path: Path, global_step: int) -> int | None:
	"""Infer the final epoch from results.csv, weights path, or global_step.

	Priority order:
	1. Last epoch found in results.csv
	2. Epoch encoded in the weights path
	3. Explicit global_step value
	"""
	results_csv = run_dir / "results.csv"
	if results_csv.is_file():
		try:
			with results_csv.open("r", encoding="utf-8", newline="") as f:
				reader = csv.DictReader(f)
				last_epoch = None
				for row in reader:
					value = row.get("epoch")
					if value is not None and str(value).strip() != "":
						last_epoch = int(float(value))
				if last_epoch is not None:
					return last_epoch
		except Exception as exc:
			logging.debug("Failed to infer final epoch from %s: %s", results_csv, exc)

	epoch_from_path = _infer_epoch_from_path_or_step(weights_path, 0)
	if epoch_from_path is not None:
		return epoch_from_path

	if global_step > 0:
		return int(global_step)

	return None


def _build_run_name(
	weights_path: Path,
	epoch: int | None,
	user_base: str | None,
	pred_conf: float | None = None,
	pred_iou: float | None = None,
) -> str:
	"""Build a TensorBoard run name from a user label or inferred run metadata."""

	ts = datetime.now().strftime("%Y%m%d-%H%M%S")

	suffix_parts = []
	if pred_conf is not None:
		suffix_parts.append("conf" + str(pred_conf).replace(".", ""))
	if pred_iou is not None:
		suffix_parts.append("iou" + str(pred_iou).replace(".", ""))
	suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

	if user_base:
		if re.match(r"^\d{8}-\d{6}_", user_base):
			return user_base + suffix
		return f"{ts}_{user_base}{suffix}"

	model_tag = _infer_model_tag_from_path(weights_path)
	init_tag = _infer_init_tag_from_path(weights_path)

	base = f"{model_tag}_{init_tag}"
	if epoch is not None:
		base += f"_e{epoch}"

	return f"{ts}_{base}{suffix}"


def load_train_run(train_run_dir: str, weights_override: str | None, dataset_yaml_override: str | None) -> tuple[Path, Path]:
	"""Resolve the weights file and dataset YAML from a training run directory.

	Values can be overridden explicitly from the CLI. If the stored weights path is
	invalid locally, the function falls back to <run_dir>/weights/best.pt.
	"""
	run_dir = Path(train_run_dir).expanduser().resolve()
	cfg_path = run_dir / "resolved_config.after.yaml"

	if not cfg_path.is_file():
		raise FileNotFoundError(f"Resolved config not found: {cfg_path}. Expected file: resolved_config.after.yaml")

	try:
		cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
	except yaml.YAMLError as e:
		raise RuntimeError(f"[CONFIG ERROR] Failed to parse YAML: {cfg_path}") from e

	if not isinstance(cfg, dict):
		raise ValueError(f"[CONFIG ERROR] YAML root must be a dict: {cfg_path}")

	resolved = cfg.get("resolved", {}) if isinstance(cfg.get("resolved", {}), dict) else {}
	data_block = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}

	if weights_override:
		weights_path = Path(weights_override).expanduser().resolve()
	else:
		best = None
		w = resolved.get("weights", {})
		if isinstance(w, dict):
			best = w.get("best", None)
		weights_path = Path(str(best)).expanduser()
		try:
			weights_path = weights_path.resolve()
		except Exception as exc:
			logging.debug("Failed to resolve weights path %s: %s", weights_path, exc)

	# If not found (typical on Windows), fallback to local run folder
	if not weights_path.is_file():
		weights_path = (run_dir / "weights" / "best.pt").resolve()

	if not weights_path.is_file():
		raise FileNotFoundError(f"best.pt not found: {weights_path}")

	if dataset_yaml_override:
		dataset_yaml_path = Path(dataset_yaml_override).expanduser().resolve()
	else:
		ds = data_block.get("dataset_yaml", None)
		dataset_yaml_path = Path(str(ds)).expanduser() if ds else None
	if dataset_yaml_path is not None:
		try:
			dataset_yaml_path = dataset_yaml_path.resolve()
		except Exception as exc:
			logging.debug(
				"Failed to resolve dataset YAML path %s: %s",
				dataset_yaml_path,
				exc,
			)

	if dataset_yaml_path is None or not dataset_yaml_path.is_file():
		raise FileNotFoundError(f"dataset yaml not found: {dataset_yaml_path}. If you're outside Docker, pass --dataset-yaml-override.")

	return weights_path, dataset_yaml_path


def _log_grid(writer: SummaryWriter, tag: str, imgs: List[np.ndarray], global_step: int) -> None:
	"""Log a batch of CHW uint8 images as a TensorBoard grid."""
	if not imgs:
		return
	grid = make_grid(torch.tensor(np.stack(imgs), dtype=torch.uint8), nrow=4)
	writer.add_image(tag, grid, global_step)


def draw_filename_overlay(
	img: np.ndarray,
	filename: str,
	max_chars: int = 28,
) -> np.ndarray:
	"""Overlay a readable filename label on an RGB image before TensorBoard logging."""
	pil_img = Image.fromarray(img).convert("RGBA")
	overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
	draw = ImageDraw.Draw(overlay)

	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	text = filename
	if len(text) > max_chars:
		text = text[: max_chars - 3] + "..."

	x, y = 8, 8

	if font is not None:
		bbox = draw.textbbox((x, y), text, font=font)
	else:
		bbox = draw.textbbox((x, y), text)

	pad_x = 6
	pad_y = 4
	draw.rounded_rectangle(
		[
			bbox[0] - pad_x,
			bbox[1] - pad_y,
			bbox[2] + pad_x,
			bbox[3] + pad_y,
		],
		radius=6,
		fill=(0, 0, 0, 170),
	)

	if font is not None:
		draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
	else:
		draw.text((x, y), text, fill=(255, 255, 255, 255))

	out = Image.alpha_composite(pil_img, overlay).convert("RGB")
	return np.array(out)


def select_image_subset(img_paths: List[Path], k: int, mode: str, seed: int) -> List[Path]:
	"""Select a deterministic subset of image paths using head or random mode."""
	img_paths = sorted(img_paths)
	if k <= 0:
		return []
	if mode == "head":
		return img_paths[:k]
	# Sample without replacement using a deterministic random seed.
	k = min(k, len(img_paths))
	rng = np.random.default_rng(seed)
	idx = rng.choice(len(img_paths), size=k, replace=False)
	return [img_paths[i] for i in sorted(idx)]


def save_image_list(img_paths: List[Path], out_path: Path) -> None:
	"""Save the selected image paths so future runs can reuse the exact same subset."""
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(
		"\n".join(str(p.resolve()) for p in img_paths),
		encoding="utf-8",
	)


def load_image_list(list_path: Path) -> List[Path]:
	"""Load a previously saved fixed image list and keep only existing files."""
	if not list_path.is_file():
		raise FileNotFoundError(f"Image list not found: {list_path}")
	paths = []
	for line in list_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		p = Path(line).expanduser().resolve()
		if p.is_file():
			paths.append(p)
	if not paths:
		raise ValueError(f"No valid image paths found in: {list_path}")
	return paths


def run_yolo_diagnostics(
	weights_path: str,
	dataset_yaml: str,
	log_dir: str,
	run_name: str,
	split: str = "val",
	max_images: int = 128,
	global_step: int = 0,
	workers: int = 8,
	conf_thres_iou: float = 0.25,
	pred_conf: float = 0.001,
	pred_iou: float = 0.7,
	match_iou: float = 0.5,
	log_images: bool = True,
	subset_mode: str = "random",
	subset_seed: int = 123,
	images_dir_override: str | None = None,
	image_list_file: str | None = None,
	fixed_compare_count: int = 32,
	category_grid_count: int = 32,
	run_val_metrics: bool = True,
) -> None:
	"""Run evaluation diagnostics on a YOLO checkpoint and export them to TensorBoard.

	The function logs:
	- global Ultralytics metrics from model.val(),
	- IoU and confidence diagnostics on a selected image subset,
	- optional image grids for good detections, missed objects, false positives,
	  and low-IoU examples.
	"""
	weights_path = Path(weights_path)

	# Keep shared artifacts (like the fixed image list) in base_log_dir,
	# while TensorBoard outputs for this run go into run_log_dir.
	base_log_dir = Path(log_dir).expanduser().resolve()
	run_log_dir = (base_log_dir / run_name).resolve()
	log_dir = run_log_dir

	if not weights_path.is_file():
		print(f"[ERROR] Weights not found: {weights_path}")
		return

	log_dir.mkdir(parents=True, exist_ok=True)
	print(f"[INFO] TensorBoard log_dir: {log_dir}")
	print(f"[INFO] Loading YOLO from {weights_path}")
	try:
		model = YOLO(str(weights_path))
	except Exception as e:
		raise RuntimeError(f"[MODEL ERROR] Failed to load YOLO weights: {weights_path}") from e

	# Resolve dataset paths, validate the dataset YAML, and identify the target split folder.

	print(f"[INFO] DATASET ANALYSIS: {dataset_yaml}")
	dataset_yaml_path = Path(dataset_yaml).expanduser().resolve()
	dataset_root = dataset_yaml_path.parent

	if not dataset_yaml_path.is_file():
		raise FileNotFoundError(f"[CONFIG ERROR] Dataset YAML not found: {dataset_yaml_path}")
	try:
		config = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
	except yaml.YAMLError as e:
		raise RuntimeError(f"[CONFIG ERROR] Failed to parse dataset YAML: {dataset_yaml_path}") from e

	if not isinstance(config, dict):
		raise ValueError(f"[CONFIG ERROR] Dataset YAML root must be a dict: {dataset_yaml_path}")

	raw_split = str(config.get(split, split))
	yaml_images_dir = Path(raw_split)
	if not yaml_images_dir.is_absolute():
		yaml_images_dir = (dataset_yaml_path.parent / yaml_images_dir).resolve()
	else:
		yaml_images_dir = yaml_images_dir.resolve()

	analysis_images_dir = Path(images_dir_override).expanduser().resolve() if images_dir_override else yaml_images_dir
	analysis_labels_dir = _derive_labels_dir_from_images_dir(analysis_images_dir)
	analysis_folder = analysis_labels_dir.name
	labels_root = analysis_labels_dir.parent

	print(f"[INFO] Running Ultralytics validation on split='{split}'")
	print(f"[INFO] Dataset root: {dataset_root}")
	print(f"[INFO] YAML {split}: '{raw_split}'")
	print(f"[INFO] Labels directory: {analysis_labels_dir}")
	print(f"[INFO] Label files found: {len(list(analysis_labels_dir.glob('*.txt'))) if analysis_labels_dir.exists() else 0}")
	print(f"[INFO] Images directory: {analysis_images_dir}")

	sample_label = next(analysis_labels_dir.glob("*.txt"), None)
	if sample_label:
		print(f"[INFO] Sample label file: {sample_label.name}")
	else:
		print(f"[WARNING] No label files found in: {analysis_labels_dir}")

	# Run Ultralytics validation to collect global detection metrics.

	ultra_val_project = (log_dir / "ultralytics").resolve()
	ultra_val_project.mkdir(parents=True, exist_ok=True)

	metrics = None

	# Allow skipping model.val() to iterate faster on custom diagnostics only.
	if run_val_metrics:
		print(f"[INFO] model.val() -> {ultra_val_project / 'val'}")
		try:
			metrics = model.val(
				data=str(dataset_yaml_path),
				split=split,
				conf=pred_conf,
				iou=pred_iou,
				plots=False,
				save_json=False,
				save=False,
				workers=min(workers, 4),
				verbose=True,
				project=str(ultra_val_project),
				name="val",
			)
		except Exception as e:
			raise RuntimeError(f"[VALIDATION ERROR] model.val() failed for dataset={dataset_yaml_path}, split={split}") from e
	else:
		print("[INFO] Skipping model.val(); running custom diagnostics only")

	# Initialize writer early so cleanup in finally stays safe if SummaryWriter creation fails.
	writer = None
	try:
		writer = SummaryWriter(log_dir=str(log_dir))
	except Exception as e:
		raise RuntimeError(f"[TENSORBOARD ERROR] Failed to create SummaryWriter at: {log_dir}") from e

	try:
		if metrics is not None:
			writer.add_scalar("kpi/map50_95", float(metrics.box.map), global_step)
			writer.add_scalar("kpi/map50", float(metrics.box.map50), global_step)
			writer.add_scalar("kpi/precision", float(metrics.box.mp), global_step)
			writer.add_scalar("kpi/recall", float(metrics.box.mr), global_step)

		# Compute custom IoU and image-level diagnostics on the selected image subset.
		ious_all: List[float] = []
		bad_examples: List[Tuple[float, np.ndarray, str]] = []
		confs_all: List[np.ndarray] = []
		fixed_compare_imgs: List[np.ndarray] = []
		good_imgs: List[np.ndarray] = []
		missed_imgs: List[np.ndarray] = []
		fp_imgs: List[np.ndarray] = []

		val_images_dir = analysis_images_dir
		all_imgs = list_image_files(val_images_dir)

		if not all_imgs:
			raise FileNotFoundError(f"No image files found in: {val_images_dir}")

		print(f"[INFO] Found {len(all_imgs)} images in: {val_images_dir}")

		# Reuse the same image subset across runs so TensorBoard comparisons stay stable.
		image_list_path = Path(image_list_file).expanduser().resolve() if image_list_file else (base_log_dir / f"selected_images_{split}.txt").resolve()
		if image_list_path.is_file():
			picked = load_image_list(image_list_path)
			if max_images > 0:
				picked = picked[: min(max_images, len(picked))]
			print(f"[INFO] Reusing fixed image list: {image_list_path}")
		else:
			picked = select_image_subset(all_imgs, k=max_images, mode=subset_mode, seed=subset_seed)
			save_image_list(picked, image_list_path)
			print(f"[INFO] Saved fixed image list: {image_list_path}")

		print(f"[INFO] Selected {len(picked)} images for TensorBoard analysis")

		results_iter = model(
			picked,
			conf=pred_conf,
			iou=pred_iou,
			verbose=False,
			stream=True,
		)

		for i, r in enumerate(results_iter, start=1):
			if i == 1 or i % 25 == 0 or i == len(picked):
				current_name = Path(r.path).name if getattr(r, "path", None) else "<unknown>"
				print(f"[INFO] Processed {i}/{len(picked)}: {current_name}")

			collect_iou_stats_for_result(
				r,
				labels_root,
				analysis_folder,
				ious_all,
				bad_examples,
				confs_all,
				low_iou_thresh=match_iou,
				conf_thres_iou=conf_thres_iou,
			)

			if not log_images:
				continue

			img = r.orig_img
			if img is None:
				continue

			img_path = Path(r.path) if r.path else None
			if img_path is None:
				continue

			h, w = img.shape[:2]
			label_path = labels_root / analysis_folder / f"{img_path.stem}.txt"
			labels = load_yolo_labels(label_path)
			gt_boxes, _ = yolo_norm_to_xyxy(labels, w, h)

			if r.boxes is None or r.boxes.xyxy is None:
				continue

			pred_boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
			pred_conf_arr = r.boxes.conf.cpu().numpy().astype(np.float32)

			cat, _ = categorize_image(
				pred_boxes=pred_boxes,
				pred_conf=pred_conf_arr,
				gt_boxes=gt_boxes,
				conf_thr=conf_thres_iou,
				iou_thr=match_iou,
			)

			img_plot = r.plot()
			if img_plot is None:
				continue

			img_resized = np.array(Image.fromarray(img_plot).resize((640, 640)))
			img_annotated = draw_filename_overlay(img_resized, img_path.name)
			chw = img_annotated.transpose(2, 0, 1)

			if cat == "good" and len(good_imgs) < category_grid_count:
				good_imgs.append(chw)
			elif cat == "missed" and len(missed_imgs) < category_grid_count:
				missed_imgs.append(chw)
			elif cat == "fp_only" and len(fp_imgs) < category_grid_count:
				fp_imgs.append(chw)

			if len(fixed_compare_imgs) < fixed_compare_count:
				fixed_compare_imgs.append(chw)

		print(f"[INFO] Categorized images: good={len(good_imgs)}, missed={len(missed_imgs)}, fp_only={len(fp_imgs)}")

		if log_images:
			_log_grid(writer, "viz/good_detections", good_imgs, global_step)
			_log_grid(writer, "viz/missed_objects", missed_imgs, global_step)
			_log_grid(writer, "viz/false_positives", fp_imgs, global_step)

		# Log aggregate IoU, confidence, and image summaries to TensorBoard.
		if ious_all:
			ious_np = np.array(ious_all)
			writer.add_scalar("diag/iou_mean", ious_np.mean(), global_step)
			writer.add_scalar("diag/iou_std", ious_np.std(), global_step)
			writer.add_histogram("diag/iou_hist", ious_np, global_step)
			writer.add_scalar("diag/iou_ge_0_5", (ious_np >= 0.5).mean(), global_step)
			print(f"[INFO] Mean IoU: {ious_np.mean():.3f} over {len(ious_all)} matched predictions")
		else:
			print(f"[WARNING] No IoU values computed. Check labels in: {labels_root / analysis_folder}")

		# Log the confidence distribution across all processed predictions.
		if confs_all:
			confs = np.concatenate(confs_all)
			writer.add_histogram("diag/conf_all", confs, global_step)

		# Log one fixed comparison grid for all runs, using the same image order.
		if log_images and fixed_compare_imgs:
			grid = make_grid(torch.tensor(np.stack(fixed_compare_imgs), dtype=torch.uint8), nrow=4)
			writer.add_image("viz/fixed_compare", grid, global_step)

		# Log the lowest-IoU examples to help inspect localization failures.
		if log_images and bad_examples:
			bad_sorted = sorted(bad_examples, key=lambda x: x[0])[:16]
			bad_imgs = []
			for _, img, filename in bad_sorted:
				# Resize images to a common shape and overlay the filename before building the grid.
				img_resized = np.array(Image.fromarray(img).resize((640, 640)))
				img_annotated = draw_filename_overlay(img_resized, filename)
				bad_imgs.append(img_annotated.transpose(2, 0, 1))
			grid = make_grid(torch.tensor(np.stack(bad_imgs), dtype=torch.uint8), nrow=4)
			writer.add_image("viz/errors_low_iou", grid, global_step)
			print(f"[INFO] Low-IoU examples logged: {len(bad_examples)}")

		print(f"[INFO] TensorBoard export complete: iou_samples={len(ious_all)}, fixed_compare_images={len(fixed_compare_imgs)}, log_dir={log_dir}")

	finally:
		if writer is not None:
			writer.close()


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for TensorBoard export and diagnostics."""
	parser = argparse.ArgumentParser("YOLO Run Diagnostics")
	parser.add_argument(
		"--train-run-dir",
		required=True,
		help="Path to a training run folder containing resolved_config.after.yaml",
	)
	parser.add_argument(
		"--log-images",
		dest="log_images",
		action="store_true",
		help="Log image grids to TensorBoard.",
	)
	parser.add_argument(
		"--no-log-images",
		dest="log_images",
		action="store_false",
		help="Disable image logging (faster).",
	)
	parser.add_argument(
		"--skip-val",
		dest="run_val_metrics",
		action="store_false",
		help="Skip model.val() and only run custom per-image diagnostics.",
	)
	parser.set_defaults(run_val_metrics=True)

	parser.set_defaults(log_images=True)

	parser.add_argument(
		"--image-list-file",
		default=None,
		help="Path to a fixed image list (.txt). If it exists, reuse it; otherwise create it.",
	)

	parser.add_argument(
		"--fixed-compare-count",
		type=int,
		default=32,
		help="Number of images to include in the fixed comparison grid.",
	)
	parser.add_argument(
		"--category-grid-count",
		type=int,
		default=32,
		help="Maximum number of images to include in good/missed/fp_only grids.",
	)

	parser.add_argument(
		"--run-name",
		default=None,
		help="Optional TB run name. Default: folder name of train-run-dir.",
	)
	parser.add_argument(
		"--conf-thres-iou",
		type=float,
		default=0.25,
		help="Confidence threshold for filtering predictions before IoU calculation.",
	)
	parser.add_argument(
		"--pred-conf",
		type=float,
		default=0.001,
		help="Confidence threshold used by model.val() and per-image inference.",
	)
	parser.add_argument(
		"--pred-iou",
		type=float,
		default=0.7,
		help="IoU threshold used by model.val() and per-image inference NMS.",
	)
	parser.add_argument(
		"--match-iou",
		type=float,
		default=0.5,
		help="IoU threshold used to match predictions to GT for good/missed categorization.",
	)
	parser.add_argument(
		"--log-dir",
		default="runs/diagnostics",
		help="Root directory for diagnostic logs.",
	)
	parser.add_argument("--split", default="val", choices=["val", "test"])
	parser.add_argument("--max-images", type=int, default=128)
	parser.add_argument(
		"--global-step",
		type=int,
		default=0,
		help="Optional manual override for TensorBoard global step. By default, the script tries to infer the final epoch from results.csv.",
	)
	parser.add_argument("--workers", type=int, default=1)
	parser.add_argument(
		"--weights-override",
		default=None,
		help="Override best.pt path (useful outside Docker when resolved path is /content/...)",
	)
	parser.add_argument(
		"--dataset-yaml-override",
		default=None,
		help="Override dataset.yaml path (useful outside Docker when config uses /content/...)",
	)
	parser.add_argument(
		"--images-dir-override",
		default=None,
		help="Optional override for the image directory used for per-image diagnostics.",
	)
	parser.add_argument(
		"--subset-seed",
		type=int,
		default=123,
		help="Seed for deterministic subset sampling.",
	)
	parser.add_argument(
		"--subset-mode",
		choices=["head", "random"],
		default="random",
		help="How to pick images: first N (head) or seeded random subset.",
	)

	return parser.parse_args()


def main() -> None:
	"""Resolve run inputs, infer the logging step, and export diagnostics to TensorBoard."""
	args = parse_args()
	run_dir = Path(args.train_run_dir).expanduser().resolve()
	run_base = args.run_name or run_dir.name

	weights_path, dataset_yaml_path = load_train_run(
		args.train_run_dir,
		args.weights_override,
		args.dataset_yaml_override,
	)

	effective_step = infer_final_epoch(
		run_dir=run_dir,
		weights_path=weights_path,
		global_step=args.global_step,
	)

	effective_step = 0 if effective_step is None else effective_step

	run_name = _build_run_name(
		weights_path=weights_path,
		epoch=effective_step if effective_step > 0 else None,
		user_base=run_base,
		pred_conf=args.pred_conf,
		pred_iou=args.pred_iou,
	)
	run_yolo_diagnostics(
		weights_path=str(weights_path),
		dataset_yaml=str(dataset_yaml_path),
		log_dir=args.log_dir,
		run_name=run_name,
		split=args.split,
		max_images=args.max_images,
		global_step=effective_step,
		workers=args.workers,
		conf_thres_iou=args.conf_thres_iou,
		pred_conf=args.pred_conf,
		pred_iou=args.pred_iou,
		match_iou=args.match_iou,
		log_images=args.log_images,
		subset_mode=args.subset_mode,
		subset_seed=args.subset_seed,
		images_dir_override=args.images_dir_override,
		image_list_file=args.image_list_file,
		fixed_compare_count=args.fixed_compare_count,
		category_grid_count=args.category_grid_count,
		run_val_metrics=args.run_val_metrics,
	)


if __name__ == "__main__":
	main()
