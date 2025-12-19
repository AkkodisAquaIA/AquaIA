import sys
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from classes.TiffDatasetLoader import TiffDatasetLoader
from classes.model_registry import model_mapping
from classes.ParamConverter import ParamConverter
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.nn import CrossEntropyLoss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt


class Inference:
	def __repr__(self):
		"""
		        Returns a string representation of the Inference class.

		        Returns:
		str: A string indicating the class name.
		"""
		return "Inference"

	def __init__(self, **kwargs):
		self.loss_mapping = {
			"CrossEntropyLoss": CrossEntropyLoss,
		}

		self.param_converter = ParamConverter()
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.subfolders = kwargs.get("subfolders")
		self.data_dir = kwargs.get("data_dir")
		self.run_dir = kwargs.get("run_dir")
		self.hyperparameters = kwargs.get("hyperparameters")

		# Model parameters
		self.model_params = {k: v for k, v in self.hyperparameters.get_parameters()["Model"].items()}
		# Sort to ensure consistent ordering across runs
		self.subfolders = sorted(d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)))

		# Training parameters
		self.training_params = {k: v for k, v in self.hyperparameters.get_parameters()["Training"].items()}
		self.batch_size = self.param_converter._convert_param(self.training_params.get("batch_size", 8))
		self.val_split = self.param_converter._convert_param(self.training_params.get("val_split", 0.8))
		self.epochs = self.param_converter._convert_param(self.training_params.get("epochs", 10))
		self.early_stopping = self.param_converter._convert_param(self.training_params.get("early_stopping", "False"))
		self.metrics_str = self.param_converter._convert_param(self.training_params.get("metrics", ""))

		# Create mapping from class name to integer index
		self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.subfolders)}
		self.num_classes = len(self.subfolders)
		print("num classes = ", self.num_classes)
		self.model_mapping = model_mapping
		self.model = self.initialize_model()

		# Data parameters
		self.data = {k: v for k, v in self.hyperparameters.get_parameters()["Data"].items()}
		self.img_res = self.param_converter._convert_param(self.data.get("img_res", 560))
		self.num_samples = self.param_converter._convert_param(self.data.get("num_samples", 500))

		self.confusion_matrix = bool(kwargs.get("confusion_matrix", False))

		# Loss parameters
		self.loss_params = {k: v for k, v in self.hyperparameters.get_parameters()["Loss"].items()}
		self.weights = self.param_converter._convert_param(self.loss_params.get("weights", "False"))

		if "Data_augmentation" in self.hyperparameters.get_parameters():
			self.data_aug_settings = {k: v for k, v in self.hyperparameters.get_parameters()["Data_augmentation"].items()}
			self.augmentation_mapping = {
				"brightness": self.param_converter._convert_param(self.data_aug_settings.get("brightness", 0)),
				"contrast": self.param_converter._convert_param(self.data_aug_settings.get("contrast", [1.0, 1.0])),
				"angle": self.param_converter._convert_param(self.data_aug_settings.get("angle", [0, 0])),
				"translate": self.param_converter._convert_param(self.data_aug_settings.get("translate", [0, 0])),
				"scale": self.param_converter._convert_param(self.data_aug_settings.get("scale", [0.0, 0.0])),
				"shear": self.param_converter._convert_param(self.data_aug_settings.get("shear", [0, 0])),
				"random_resized_crop_scale": self.param_converter._convert_param(self.data_aug_settings.get("random_resized_crop_scale", [0.85, 1.0])),
				"random_resized_crop_ratio": self.param_converter._convert_param(self.data_aug_settings.get("random_resized_crop_ratio", [0.9, 1.1])),
				"elastic_alpha": self.param_converter._convert_param(self.data_aug_settings.get("elastic_alpha", 30)),
				"elastic_sigma": self.param_converter._convert_param(self.data_aug_settings.get("elastic_sigma", 5)),
				"elastic_affine": self.param_converter._convert_param(self.data_aug_settings.get("elastic_affine", 5)),
				"color_jitter": self.param_converter._convert_param(self.data_aug_settings.get("color_jitter", [0.2, 0.2, 0.15, 0.05])),
				"gauss_noise": self.param_converter._convert_param(self.data_aug_settings.get("gauss_noise", [5.0, 20.0])),
				"coarse_dropout": self.param_converter._convert_param(self.data_aug_settings.get("coarse_dropout", [3, 6])),
				"blur": self.param_converter._convert_param(self.data_aug_settings.get("blur", 3)),
				"horizontal_flip_p": self.param_converter._convert_param(self.data_aug_settings.get("horizontal_flip_p", 0.5)),
				"vertical_flip_p": self.param_converter._convert_param(self.data_aug_settings.get("vertical_flip_p", 0.2)),
			}
		else:
			self.augmentation_mapping = False

	def initialize_model(self) -> nn.Module:
		"""
		        Initializes the model based on the specified model type and loads the pre-trained weights.

		        Returns:
		nn.Module: The initialized model ready for inference.

		        Raises:
		ValueError: If the specified model type is not supported or if there is an error converting parameters.
		"""
		model_name = self.model_params.get("model_type", "UnetVanilla")
		if model_name not in self.model_mapping:
			raise ValueError(f"Model '{model_name}' is not supported. Check your 'model_mapping'.")

		model_class = self.model_mapping[model_name]
		self.model_params["num_classes"] = self.num_classes

		required_params = {k: self.param_converter._convert_param(v) for k, v in self.model_params.items() if k in model_class.REQUIRED_PARAMS}
		optional_params = {k: self.param_converter._convert_param(v) for k, v in self.model_params.items() if k in model_class.OPTIONAL_PARAMS}

		# Ensure `model_type` is not included in the parameters
		required_params.pop("model_type", None)
		optional_params.pop("model_type", None)

		# Ensure 'num_classes' is only in the required parameters, remove it from optional if present
		if "num_classes" in optional_params:
			del optional_params["num_classes"]

		if model_name == "DINOv2":
			optional_params["quantize"] = True
			optional_params["peft"] = True

		try:
			# Convert the required parameters to their correct types as defined by the model class
			typed_required_params = {k: model_class.REQUIRED_PARAMS[k](v) for k, v in required_params.items()}
		except ValueError as e:
			raise ValueError(f"Error converting parameters for model '{model_name}': {e}")

		# Initialize the model
		model = model_class(**typed_required_params, **optional_params).to(self.device)

		# Load pre-trained weights
		checkpoint_path = os.path.join(self.run_dir, "model_best_Accuracy.pth")
		print(checkpoint_path)
		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Ensure the path is correct.")

		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		model.load_state_dict(checkpoint, strict=False)
		model.eval()  # Set the model to evaluation mode

		return model

	def load_dataset(self, test_file_path):
		"""
		        Initializes the dataset loader in inference mode to process images as specified
		        in the test_file_path.

		        Returns:
		DataLoader: A DataLoader object for iterating over the dataset.
		"""

		def load_class_p_scale(file_path):
			result = {}
			with open(file_path, "r") as f:
				for line in f:
					if line.startswith("#") or not line.strip():
						continue
					cls, pscale = line.strip().split(":")
					result[cls.strip()] = float(pscale.strip())
			return result

		def load_data_stats(data_dir):
			"""
			              Loads normalization statistics from a JSON file. Provides default normalization
			              stats if the file is missing or improperly formatted.

			              Args:
			data_dir (str): Directory containing the data stats JSON file.

			              Returns:
			dict: A dictionary containing the loaded data statistics.
			"""
			neutral_stats = [np.array([0.5] * 3), np.array([0.5] * 3)]  # Default mean and std
			json_file_path = os.path.join(data_dir, "data_stats.json")

			if not os.path.exists(json_file_path):
				print(f"File {json_file_path} not found. Using default normalization stats.")
				return {"default": neutral_stats}

			try:
				with open(json_file_path, "r") as file:
					raw_data_stats = json.load(file)

				data_stats_loaded = {}
				for key, value in raw_data_stats.items():
					if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, list) and len(v) == 3 for v in value)):
						raise ValueError(f"Invalid format in data_stats.json for key {key}")

					data_stats_loaded[key] = [np.array(value[0]), np.array(value[1])]

				return data_stats_loaded

			except (json.JSONDecodeError, ValueError) as e:
				print(f"Error loading data stats from {json_file_path}: {e}. Using default normalization stats.")
				return {"default": neutral_stats}

		# Step 1: Get sorted class folders
		class_folders = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

		# Step 2: Map numeric index to class/folder name and vice versa
		idx_to_class = {i: class_folders[i] for i in range(len(class_folders))}
		# class_to_idx = {v: k for k, v in idx_to_class.items()}

		# Step 3: Parse test_file to get image indices per class
		indices = []
		img_data = {}
		img_labels = {}

		# Collect image indices from file
		with open(test_file_path, "r") as f:
			lines = f.readlines()

		# Track which class indices are used
		used_class_indices = set()
		for line in lines:
			class_idx_str, img_idx_str = line.strip().split(",")
			class_idx = int(class_idx_str)
			img_idx = int(img_idx_str)

			indices.append((class_idx, img_idx))
			used_class_indices.add(class_idx)

		# Step 4: Load image paths and labels only for used classes
		for class_idx in used_class_indices:
			class_name = idx_to_class[class_idx]
			class_folder = os.path.join(self.data_dir, class_name)
			img_files = sorted(glob.glob(os.path.join(class_folder, "*.*")))

			img_data[class_name] = img_files
			img_labels[class_name] = [class_idx] * len(img_files)

		data_stats = load_data_stats(self.data_dir)
		class_p_scale = load_class_p_scale(os.path.join(self.run_dir, "class_p_scale.txt"))
		"""
        class_p_scale = {
            'Agrypnia_sp': 1.5,
            'Ameletus_inopinatus': 0.7,
            'Asellus_aquaticus': 0.6,
            'Atherix_ibis': 0.7,
            'Baetis_digitatus': 0.8,
            'Baetis_niger': 0.6,
            'Baetis_rhodani': 0.6,
            'Brachyptera_risi': 1.5,
            'Callicorixa_wollastoni': 1.5,
            'Centroptilum_luteolum': 1.1,
            'Ceraclea_excisa': 0.9,
            'Ceratopogonidae': 0.7,
            'Chelifera': 1.2,
            'Chimarra_marginata': 0.8,
            'Chironomidae': 1.2,
            'Cyrnus_flavidus': 1.5,
            'Dicranota': 0.6,
            'Diura_nanseni': 0.8,
            'Elmis_aenea_adult': 0.6,
            'Elmis_aenea_larva': 0.6,
            'Elodes': 0.9,
            'Eloeophila_sp': 1.0,
            'Ephemerella_aroni_aurivillii': 0.6,
            'Gammarus_lacustris': 0.9,
            'Gyraulus_sp': 0.8,
            'Habrophlebia_sp': 0.6,
            'Hemerodromia': 0.7,
            'Heptagenia_dalecarlica': 0.6,
            'Hydracarina': 0.7,
            'Hydraena_adult': 0.6,
            'Hydropsyche_pellucidula': 0.8,
            'Hydropsyche_saxonica': 1.5,
            'Hydropsyche_siltalai': 1.2,
            'Isoperla_sp': 0.6,
            'Ithytrichia_lamellaris': 0.6,
            'Lepidostoma_hirtum': 1.1,
            'Leptophlebia_sp': 0.6,
            'Leuctra_nigra': 0.8,
            'Leuctra_sp': 0.6,
            'Limnephilidae': 1.0,
            'Limnius_volckmari_adult': 0.6,
            'Micrasema_gelidum': 0.6,
            'Micrasema_setiferum': 0.6,
            'Nemoura_sp': 0.6,
            'Neureclipsis_bimaculata': 1.5,
            'Oulimnius_tuberculatus_adult': 0.8,
            'Oulimnius_tuberculatus_larva': 0.6,
            'Oxyethira_sp': 0.6,
            'Paraleptophlebia_sp': 0.7,
            'Philopotamus_montanus': 0.7,
            'Pisidium_sp': 1.5,
            'Plectrocnemia_conspersa': 0.7,
            'Polycentropus_flavomaculatus': 0.8,
            'Polycentropus_irroratus': 1.5,
            'Protonemura_sp': 0.6,
            'Psychodiidae': 0.6,
            'Rhyacophila_fasciata_obtilerata': 1.2,
            'Rhyacophila_nubila': 0.9,
            'Sericostoma_personatum': 0.9,
            'Sialis_lutaria': 1.0,
            'Silo_pallipes': 1.5,
            'Simuliidae': 0.6,
            'Siphonoperla_burmeisteri': 0.8,
            'Taeniopteryx_nebulosa': 0.6
        }
        """

		dataset = TiffDatasetLoader(
			indices=indices,
			img_data=img_data,
			img_labels=img_labels,
			data_stats=data_stats,
			img_res=self.img_res,
			weights=self.weights,
			augmentation_params=self.augmentation_mapping,
			rare_classes=class_p_scale,
		)

		return DataLoader(dataset, batch_size=1, shuffle=False)

	def save_confusion_matrix(self, preds, labels):
		# Ensure preds and labels are converted to tensors
		preds = torch.cat(preds) if isinstance(preds, list) else preds
		labels = torch.cat(labels) if isinstance(labels, list) else labels

		# Determine the correct device
		device = self.device  # Should be consistent with the model's device

		# Move tensors **before** initializing confusion matrix
		preds = preds.to(device)
		labels = labels.to(device)

		# Ensure confusion matrix itself is on the correct device
		cm = MulticlassConfusionMatrix(num_classes=self.model.num_classes).to(device)

		# Update confusion matrix with predictions
		cm.update(preds, labels)

		# Compute confusion matrix and move it to CPU before converting to NumPy
		cm_matrix = cm.compute().to("cpu").numpy()

		# Convert to % per true-class row
		cm_percent = (cm_matrix.astype(np.float32) / (cm_matrix.sum(axis=1, keepdims=True) + 1e-6)) * 100

		# Plot
		fig_size = max(12, self.num_classes * 0.25)
		plt.figure(figsize=(fig_size, fig_size))
		plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
		plt.colorbar(label="Percent")

		ticks = np.arange(self.num_classes)
		plt.xticks(ticks, ticks, rotation=90, fontsize=6)
		plt.yticks(ticks, ticks, fontsize=6)
		plt.xlabel("Predicted Label", fontsize=10)
		plt.ylabel("True Label", fontsize=10)

		plt.tight_layout()
		out_path = os.path.join(self.run_dir, "confusion_matrix_testing_set.png")
		plt.savefig(out_path, dpi=300)
		plt.close()
		print(f"Confusion matrix saved to {out_path}")

	def predict(self):
		"""
		Runs image‐level classification on every sample in the inference set,
		computes CrossEntropyLoss(logits, label) per sample, and prints the average loss.

		We do NOT unpack img_path from __getitem__. Instead, we use dataset.indices[i] to
		recover (class_idx, sample_idx) and then reconstruct the file path from dataset.img_data.
		"""
		# 1) Build DataLoader in inference mode
		dataloader = self.load_dataset(os.path.join(self.run_dir, "test_indices.txt"))
		dataset = dataloader.dataset
		print(f"Total test samples: {len(dataset)}")

		# Track predictions and labels for confusion matrix
		final_preds = []
		final_labels = []

		# 2) Prepare model & loss
		self.model.eval()
		device = self.device
		loss_fn = nn.CrossEntropyLoss()
		running_loss = 0.0
		total_samples = 0

		with torch.no_grad():
			for i, (img_resized, label) in enumerate(tqdm(dataloader, desc="Evaluating")):
				# Move image and label to device
				img_tensor = img_resized.to(device)  # Already shape [1, 3, H, W]
				label_tensor = torch.tensor([label], dtype=torch.long).to(device)

				logits = self.model(img_tensor)  # shape [1, num_classes]
				pred_class = logits.argmax(dim=1)  # for logging/debug only

				# CORRECT loss computation:
				loss = loss_fn(logits, label_tensor)

				# Store predictions for confusion matrix
				final_preds.append(pred_class.cpu())
				final_labels.append(label_tensor.cpu())

				running_loss += loss.item()
				total_samples += 1

		avg_test_loss = running_loss / total_samples

		if self.confusion_matrix:
			self.save_confusion_matrix(final_preds, final_labels)

		print(f"Average CrossEntropy Test Loss: {avg_test_loss:.4f}")
		return avg_test_loss
