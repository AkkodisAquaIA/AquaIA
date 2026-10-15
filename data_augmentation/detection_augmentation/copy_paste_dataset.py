"""Dataset PyTorch copy-paste généré directement en mémoire.

Les images ne sont jamais enregistrées en TIFF ou en PNG. Le fond et les cutouts
sont redimensionnés à la taille d'entrée du modèle, puis assemblés dans __getitem__.

Pour comparer deux modèles avec les mêmes données :
1. utiliser la même ``seed`` et les mêmes paramètres ;
2. appeler ``set_dataloader_epoch(train_loader, epoch)`` à chaque époque.

Le train change alors d'une époque à l'autre, tandis que val et test restent fixes.
"""

import json
import random
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from data_augmentation.detection_augmentation.copy_and_paste_augmentation import (
    LOSSLESS_EXTENSIONS,
    box_iou,
    boxes_are_nested,
    list_images,
)


# -----------------------------------------------------------------------------
# Paramètres principaux à modifier
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUTOUT_DIR = PROJECT_ROOT / "datasets/LPL_dataset_cutouts_single"
BACKGROUND_DIR = PROJECT_ROOT / "datasets/Images_fond_blanc_&_bruits"

MODEL_INPUT_SIZE = 1024
TRAIN_IMAGES_PER_EPOCH = 700
VAL_IMAGES = 100
TEST_IMAGES = 200
USE_NOISY_BACKGROUNDS = True
BACKGROUND_KIND = "all" if USE_NOISY_BACKGROUNDS else "clean"

MIN_MACROINVERTEBRATES = 4
MAX_MACROINVERTEBRATES = 10
PADDING = 0.08
MAX_ROTATION = 180
CUTOUT_SCALE_RANGE = (0.8, 1.2)
ALLOW_OVERLAP = True
IOU_MAX = 1.0
MULTI_CLASS = False
CLASS_MODE = "multi" if MULTI_CLASS else "single"
TEST_MANIFEST_PATH = PROJECT_ROOT / (
    f"datasets/copy_paste_test_manifest_{CUTOUT_DIR.name}_{CLASS_MODE}_"
    f"{BACKGROUND_KIND}_{MIN_MACROINVERTEBRATES}_{MAX_MACROINVERTEBRATES}.json"
)

BATCH_SIZE = 2
NUM_WORKERS = 2
SEED = 42

# Mettre à True pour lancer directement un entraînement YOLO avec main().
RUN_YOLO_TRAINING = False
YOLO_MODEL_ID = "yolo26n.pt"
YOLO_EPOCHS = 100
YOLO_OUTPUT_DIR = PROJECT_ROOT / "results/detection/copy_paste_yolo"

IMAGE_EXTENSIONS = LOSSLESS_EXTENSIONS
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}


class CopyPasteDataset(Dataset):
    """Génère des images de détection copy-paste sans les écrire sur le disque.

    Les boîtes retournées dans ``target["boxes"]`` sont au format YOLO normalisé
    ``(centre_x, centre_y, largeur, hauteur)``.
    """

    def __init__(
        self,
        cutout_dir,
        background_dir,
        length,
        image_size=1024,
        split="train",
        seed=42,
        min_macroinvertebrates=4,
        max_macroinvertebrates=10,
        padding=0.08,
        max_rotation=180,
        cutout_scale_range=(0.8, 1.2),
        allow_overlap=True,
        iou_max=1.0,
        multi_class=False,
        normalization_mean=None,
        normalization_std=None,
        manifest_path=None,
        cache_backgrounds=True,
        background_cache=None,
        use_noisy_backgrounds=True,
    ):
        if split not in SPLIT_SEED_OFFSETS:
            raise ValueError("split doit être 'train', 'val' ou 'test'.")
        if length <= 0:
            raise ValueError("length doit être supérieur à zéro.")
        if not 0 <= padding < 0.5:
            raise ValueError("padding doit être compris entre 0 et 0.5.")
        if not 0 <= iou_max <= 1:
            raise ValueError("iou_max doit être compris entre 0 et 1.")
        if min_macroinvertebrates < 0 or max_macroinvertebrates < min_macroinvertebrates:
            raise ValueError("Le minimum doit être positif ou nul et inférieur ou égal au maximum.")
        if len(cutout_scale_range) != 2 or cutout_scale_range[0] <= 0 or cutout_scale_range[1] < cutout_scale_range[0]:
            raise ValueError("cutout_scale_range doit contenir deux valeurs positives croissantes.")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size doit être un entier ou un couple (largeur, hauteur).")

        self.cutout_dir = Path(cutout_dir)
        self.background_dir = Path(background_dir)
        self.length = int(length)
        self.image_size = tuple(map(int, image_size))
        self.split = split
        self.seed = int(seed)
        self.min_macroinvertebrates = int(min_macroinvertebrates)
        self.max_macroinvertebrates = int(max_macroinvertebrates)
        self.padding = float(padding)
        self.max_rotation = float(max_rotation)
        self.cutout_scale_range = tuple(map(float, cutout_scale_range))
        self.allow_overlap = bool(allow_overlap)
        self.iou_max = float(iou_max)
        self.multi_class = bool(multi_class)
        self.use_noisy_backgrounds = bool(use_noisy_backgrounds)

        all_backgrounds = list_images(self.background_dir, IMAGE_EXTENSIONS)
        if len(all_backgrounds) < 8:
            raise ValueError("Le dossier doit contenir les 4 fonds sans bruit puis les 4 fonds avec bruit.")
        # False : uniquement les 4 fonds propres. True : les 8 fonds disponibles.
        self.backgrounds = all_backgrounds[:8] if self.use_noisy_backgrounds else all_backgrounds[:4]
        self.cutouts = list_images(self.cutout_dir, {".png"})
        if not self.backgrounds:
            raise FileNotFoundError("Aucun fond PNG ou TIFF trouvé.")
        if not self.cutouts:
            raise FileNotFoundError("Aucun cutout PNG trouvé.")

        self.manifest_path = Path(manifest_path) if manifest_path else None
        if self.manifest_path and self.split != "test":
            raise ValueError("Le manifest est réservé au split test.")
        if self.manifest_path and self.manifest_path.exists():
            self._load_manifest()

        self.epoch = Value("i", 0)
        if background_cache is not None:
            self.background_cache = background_cache
        elif cache_backgrounds:
            self.background_cache = self._prepare_backgrounds()
        else:
            self.background_cache = {}

        self.class_names = sorted({path.parent.name for path in self.cutouts}) if self.multi_class else ["macroinvertebrates"]
        self.class_to_id = {name: index for index, name in enumerate(self.class_names)}

        self.mean = self._normalization_tensor(normalization_mean)
        self.std = self._normalization_tensor(normalization_std)
        if (self.mean is None) != (self.std is None):
            raise ValueError("normalization_mean et normalization_std doivent être fournis ensemble.")
        if self.manifest_path and not self.manifest_path.exists():
            self._save_manifest()

    @staticmethod
    def _normalization_tensor(values):
        if values is None:
            return None
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() != 3:
            raise ValueError("La normalisation doit contenir trois valeurs RGB.")
        return tensor.view(3, 1, 1)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        """Change l'époque, y compris dans les workers persistants."""

        with self.epoch.get_lock():
            self.epoch.value = int(epoch)

    def _sample_seed(self, index, epoch):
        # val et test restent identiques, même si la boucle appelle set_epoch().
        if self.split != "train":
            epoch = 0
        return self.seed + SPLIT_SEED_OFFSETS[self.split] + epoch * self.length + index

    def _manifest_parameters(self):
        return {
            "image_size": list(self.image_size),
            "length": self.length,
            "seed": self.seed,
            "min_macroinvertebrates": self.min_macroinvertebrates,
            "max_macroinvertebrates": self.max_macroinvertebrates,
            "padding": self.padding,
            "max_rotation": self.max_rotation,
            "cutout_scale_range": list(self.cutout_scale_range),
            "allow_overlap": self.allow_overlap,
            "iou_max": self.iou_max,
            "multi_class": self.multi_class,
            "use_noisy_backgrounds": self.use_noisy_backgrounds,
        }

    def _save_manifest(self):
        """Conserve la configuration, les sources ordonnées et la seed de chaque image."""

        manifest = {
            "version": 1,
            "split": "test",
            "parameters": self._manifest_parameters(),
            "backgrounds": [str(path.relative_to(self.background_dir)) for path in self.backgrounds],
            "cutouts": [str(path.relative_to(self.cutout_dir)) for path in self.cutouts],
            "samples": [{"index": index, "seed": self._sample_seed(index, 0)} for index in range(self.length)],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Manifest du test enregistré dans : {self.manifest_path.resolve()}")

    def _load_manifest(self):
        """Recharge les listes ordonnées afin qu'un fichier ajouté ne change pas le test."""

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("split") != "test":
            raise ValueError(f"Manifest de test invalide : {self.manifest_path}")
        if manifest.get("parameters") != self._manifest_parameters():
            raise ValueError(
                "Les paramètres demandés diffèrent du manifest de test. "
                "Réutiliser les mêmes paramètres ou choisir un nouveau manifest."
            )

        self.backgrounds = [self.background_dir / relative_path for relative_path in manifest["backgrounds"]]
        self.cutouts = [self.cutout_dir / relative_path for relative_path in manifest["cutouts"]]
        missing = [path for path in self.backgrounds + self.cutouts if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Source du manifest introuvable : {missing[0]}")

    def _prepare_backgrounds(self):
        """Décode et redimensionne les fonds une seule fois."""

        cache = {}
        background_kind = "avec et sans bruit" if self.use_noisy_backgrounds else "sans bruit"
        print(f"Préparation de {len(self.backgrounds)} fond(s) {background_kind} en {self.image_size}...")
        for path in self.backgrounds:
            with Image.open(path) as image:
                background = image.convert("RGB")
            original_size = background.size
            resized = ImageOps.fit(background, self.image_size, method=Image.Resampling.LANCZOS)
            ratio = min(self.image_size[0] / original_size[0], self.image_size[1] / original_size[1])
            cache[path] = (resized, ratio)
        return cache

    def _load_background(self, rng):
        path = rng.choice(self.backgrounds)
        if path in self.background_cache:
            background, resize_ratio = self.background_cache[path]
            return background.copy(), resize_ratio

        with Image.open(path) as image:
            background = image.convert("RGB")
        original_size = background.size
        background = ImageOps.fit(background, self.image_size, method=Image.Resampling.LANCZOS)
        resize_ratio = min(self.image_size[0] / original_size[0], self.image_size[1] / original_size[1])
        return background, resize_ratio

    def _load_cutout(self, rng, resize_ratio):
        cutout_path = rng.choice(self.cutouts)
        with Image.open(cutout_path) as image:
            cutout = image.convert("RGBA")

        scale = resize_ratio * rng.uniform(*self.cutout_scale_range)
        width = max(1, round(cutout.width * scale))
        height = max(1, round(cutout.height * scale))
        cutout = cutout.resize((width, height), Image.Resampling.LANCZOS)
        cutout = cutout.rotate(
            rng.uniform(-self.max_rotation, self.max_rotation),
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )
        alpha_box = cutout.getchannel("A").getbbox()
        if alpha_box is None:
            return None, None

        class_id = self.class_to_id[cutout_path.parent.name] if self.multi_class else 0
        return cutout.crop(alpha_box), class_id

    def _generate(self, rng):
        background, resize_ratio = self._load_background(rng)
        padding_x = round(background.width * self.padding)
        padding_y = round(background.height * self.padding)
        target_count = rng.randint(self.min_macroinvertebrates, self.max_macroinvertebrates)
        annotations = []

        for _ in range(max(50, target_count * 20)):
            if len(annotations) == target_count:
                break

            cutout, class_id = self._load_cutout(rng, resize_ratio)
            if cutout is None:
                continue

            max_x = background.width - padding_x - cutout.width
            max_y = background.height - padding_y - cutout.height
            if max_x < padding_x or max_y < padding_y:
                continue

            x = rng.randint(padding_x, max_x)
            y = rng.randint(padding_y, max_y)
            box = (x, y, x + cutout.width, y + cutout.height)
            overlap_limit = self.iou_max if self.allow_overlap else 0.0
            invalid_position = any(
                boxes_are_nested(box, other_box) or box_iou(box, other_box) > overlap_limit
                for _, other_box in annotations
            )
            if invalid_position:
                continue

            background.paste(cutout, (x, y), cutout)
            annotations.append((class_id, box))

        return background, annotations

    def __getitem__(self, index):
        # Le sampler du train transmet (époque, index). Un entier reste accepté.
        if isinstance(index, tuple):
            epoch, index = index
        else:
            epoch = self.epoch.value
        index = int(index)

        rng = random.Random(self._sample_seed(index, int(epoch)))
        for _ in range(30):
            image, annotations = self._generate(rng)
            if len(annotations) >= self.min_macroinvertebrates:
                break
        else:
            raise RuntimeError("Impossible de placer assez d'insectes. Réduire leur taille, le padding ou les contraintes de superposition.")

        labels = torch.tensor([class_id for class_id, _ in annotations], dtype=torch.int64)
        boxes = []
        for _, (left, top, right, bottom) in annotations:
            boxes.append(
                [
                    (left + right) / 2 / image.width,
                    (top + bottom) / 2 / image.height,
                    (right - left) / image.width,
                    (bottom - top) / image.height,
                ]
            )

        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        model_input = image_tensor
        if self.mean is not None:
            model_input = (image_tensor - self.mean) / self.std

        return {
            "image": image_tensor,
            "input": model_input,
            "target": {
                "labels": labels,
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            },
            "img_path": f"synthetic://{self.split}/{index:06d}",
            "sample_seed": self._sample_seed(index, int(epoch)),
        }


class SeededEpochSampler(Sampler):
    """Mélange reproductible qui transmet l'époque au Dataset."""

    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for index in torch.randperm(len(self.dataset), generator=generator).tolist():
            yield self.epoch, index

    def __len__(self):
        return len(self.dataset)


def copy_paste_collate_fn(batch):
    """Regroupe les images et concatène les annotations de taille variable."""

    counts = [len(sample["target"]["labels"]) for sample in batch]
    return {
        "inputs": torch.stack([sample["input"] for sample in batch]),
        "images": torch.stack([sample["image"] for sample in batch]),
        "targets": {
            "labels": torch.cat([sample["target"]["labels"] for sample in batch]),
            "boxes": torch.cat([sample["target"]["boxes"] for sample in batch]),
            "counts": counts,
        },
        "img_paths": [sample["img_path"] for sample in batch],
        "sample_seeds": [sample["sample_seed"] for sample in batch],
    }


def seed_worker(worker_id):
    """Initialise aussi les bibliothèques qui utiliseraient un aléatoire global."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_copy_paste_dataloaders(
    cutout_dir,
    background_dir,
    image_size,
    train_length,
    val_length,
    test_length,
    batch_size=2,
    num_workers=2,
    seed=42,
    pin_memory=True,
    persistent_workers=True,
    test_manifest_path=None,
    cache_backgrounds=True,
    **dataset_parameters,
):
    """Crée les trois Datasets et DataLoaders avec les mêmes paramètres."""

    lengths = {"train": train_length, "val": val_length, "test": test_length}
    datasets = {}
    background_cache = None
    for split, length in lengths.items():
        datasets[split] = CopyPasteDataset(
            cutout_dir=cutout_dir,
            background_dir=background_dir,
            length=length,
            image_size=image_size,
            split=split,
            seed=seed,
            manifest_path=test_manifest_path if split == "test" else None,
            cache_backgrounds=cache_backgrounds,
            background_cache=background_cache,
            **dataset_parameters,
        )
        background_cache = datasets[split].background_cache

    train_sampler = SeededEpochSampler(datasets["train"], seed=seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": copy_paste_collate_fn,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = persistent_workers
        loader_options["prefetch_factor"] = 1

    loaders = {
        "train": DataLoader(datasets["train"], sampler=train_sampler, **loader_options),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }
    return datasets, loaders


def set_dataloader_epoch(train_loader, epoch):
    """À appeler au début de chaque époque d'entraînement."""

    if not hasattr(train_loader.sampler, "set_epoch"):
        raise TypeError("Le DataLoader ne possède pas de SeededEpochSampler.")
    train_loader.sampler.set_epoch(epoch)
    train_loader.dataset.set_epoch(epoch)


# -----------------------------------------------------------------------------
# Raccord avec Ultralytics YOLO
# -----------------------------------------------------------------------------


class YOLOCopyPasteDataset(CopyPasteDataset):
    """Même générateur, avec les noms et types de tenseurs attendus par YOLO."""

    collate_fn = staticmethod(YOLODataset.collate_fn)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        labels = sample["target"]["labels"]
        height, width = sample["image"].shape[1:]

        return {
            # Le Trainer YOLO effectue lui-même la division par 255.
            "img": (sample["image"] * 255).round().to(torch.uint8),
            "cls": labels.float().view(-1, 1),
            "bboxes": sample["target"]["boxes"],
            "batch_idx": torch.zeros(len(labels), dtype=torch.long),
            "im_file": sample["img_path"],
            "ori_shape": (height, width),
            "resized_shape": (height, width),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
        }


class CopyPasteYOLOTrainer(DetectionTrainer):
    """Trainer YOLO qui remplace les fichiers du dataset par notre générateur."""

    configured_dataset = None

    @classmethod
    def configure(cls, parameters):
        cls.configured_dataset = dict(parameters)

    def __init__(self, *args, **kwargs):
        if self.configured_dataset is None:
            raise RuntimeError("Appeler CopyPasteYOLOTrainer.configure() avant l'entraînement.")
        self.copy_paste_parameters = dict(self.configured_dataset)
        self.shared_background_cache = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._set_copy_paste_epoch)

    def get_dataset(self):
        cutouts = list_images(self.copy_paste_parameters["cutout_dir"], {".png"})
        if self.copy_paste_parameters["multi_class"]:
            names = sorted({path.parent.name for path in cutouts})
        else:
            names = ["macroinvertebrates"]
        return {
            "train": "copy_paste_train",
            "val": "copy_paste_val",
            "test": "copy_paste_test",
            "names": {index: name for index, name in enumerate(names)},
            "nc": len(names),
            "channels": 3,
        }

    def _create_dataset(self, split):
        parameters = dict(self.copy_paste_parameters)
        lengths = parameters.pop("lengths")
        test_manifest_path = parameters.pop("test_manifest_path")
        dataset = YOLOCopyPasteDataset(
            length=lengths[split],
            image_size=self.args.imgsz,
            split=split,
            manifest_path=test_manifest_path if split == "test" else None,
            background_cache=self.shared_background_cache,
            **parameters,
        )
        self.shared_background_cache = dataset.background_cache
        return dataset

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        split = "train" if mode == "train" else "test" if dataset_path == "copy_paste_test" else "val"
        dataset = self._create_dataset(split)

        # Le test n'est pas parcouru pendant train(), mais son manifest est créé.
        if split == "val" and self.copy_paste_parameters["test_manifest_path"]:
            self.test_dataset = self._create_dataset("test")

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=mode == "train",
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            device=self.device,
        )

    @staticmethod
    def _set_copy_paste_epoch(trainer):
        trainer.train_loader.dataset.set_epoch(trainer.epoch)

    def plot_training_labels(self):
        # Les labels sont créés à la demande et n'existent pas tous au démarrage.
        pass

    def get_validator(self):
        validator = super().get_validator()
        # La validation finale de best.pt exige un YAML, même avec un DataLoader déjà créé.
        data_file = self.save_dir / "copy_paste_dataset.yaml"
        data_file.write_text(
            json.dumps(
                {
                    "path": str(self.save_dir.resolve()),
                    "train": ".",
                    "val": ".",
                    "test": ".",
                    "names": self.data["names"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        validator.args.data = str(data_file)
        return validator


class CopyPasteYOLOValidator(DetectionValidator):
    """Validator YOLO utilisant le split test généré en mémoire."""

    configured_dataset = None

    @classmethod
    def configure(cls, parameters):
        cls.configured_dataset = dict(parameters)

    def get_dataloader(self, dataset_path, batch_size):
        del dataset_path
        parameters = dict(self.configured_dataset)
        test_length = parameters.pop("test_length")
        test_manifest_path = parameters.pop("test_manifest_path")
        dataset = YOLOCopyPasteDataset(
            length=test_length,
            image_size=self.args.imgsz,
            split="test",
            manifest_path=test_manifest_path,
            **parameters,
        )
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=False,
            rank=-1,
            device=self.device,
        )

    def get_stats(self):
        """Conserve les prédictions brutes pour les compter à un seuil choisi."""

        self.metrics.raw_stats = {
            key: np.concatenate(values, axis=0) if values else np.array([])
            for key, values in self.metrics.stats.items()
        }
        return super().get_stats()


def train_yolo_copy_paste(
    model_id,
    cutout_dir,
    background_dir,
    image_size=1024,
    train_length=700,
    val_length=100,
    test_length=200,
    test_manifest_path=None,
    epochs=100,
    batch_size=2,
    workers=2,
    seed=42,
    project=None,
    training_parameters=None,
    **parameters,
):
    """Configure le Dataset, entraîne YOLO et renvoie le modèle et les métriques."""

    dataset_parameters = {
        "cutout_dir": Path(cutout_dir),
        "background_dir": Path(background_dir),
        "lengths": {"train": train_length, "val": val_length, "test": test_length},
        "seed": seed,
        "test_manifest_path": test_manifest_path,
        "multi_class": parameters.pop("multi_class", False),
        "cache_backgrounds": parameters.pop("cache_backgrounds", True),
        **parameters,
    }
    CopyPasteYOLOTrainer.configure(dataset_parameters)

    model = YOLO(model_id)
    train_arguments = dict(training_parameters or {})
    for augmentation in ("mosaic", "mixup", "cutmix", "copy_paste", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v"):
        train_arguments.setdefault(augmentation, 0.0)
    train_arguments.update(
        {
            "data": "copy_paste",
            "imgsz": image_size,
            "epochs": epochs,
            "batch": batch_size,
            "workers": workers,
            "seed": seed,
            "deterministic": True,
            "close_mosaic": 0,
            "project": str(project) if project else None,
        }
    )
    metrics = model.train(trainer=CopyPasteYOLOTrainer, **train_arguments)
    return model, metrics


def predict_yolo_image(model_or_weights, image_path, output_dir, confidence=0.25, image_size=1024):
    """Lance YOLO sur une image réelle et enregistre l'image avec les boîtes."""

    model = model_or_weights if isinstance(model_or_weights, YOLO) else YOLO(model_or_weights)
    return model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=image_size,
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
    )


def evaluate_yolo_test(
    model_or_weights,
    cutout_dir,
    background_dir,
    output_dir,
    test_manifest_path,
    image_size=1024,
    test_length=200,
    batch_size=2,
    workers=2,
    seed=42,
    device=None,
    **parameters,
):
    """Calcule précision, rappel et mAP sur le test synthétique reproductible."""

    model = model_or_weights if isinstance(model_or_weights, YOLO) else YOLO(model_or_weights)
    multi_class = parameters.pop("multi_class", False)
    cutouts = list_images(cutout_dir, {".png"})
    names = sorted({path.parent.name for path in cutouts}) if multi_class else ["macroinvertebrates"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "copy_paste_test_dataset.yaml"
    data_file.write_text(
        json.dumps(
            {
                "path": str(output_dir.resolve()),
                "train": ".",
                "val": ".",
                "test": ".",
                "names": {index: name for index, name in enumerate(names)},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    CopyPasteYOLOValidator.configure(
        {
            "cutout_dir": Path(cutout_dir),
            "background_dir": Path(background_dir),
            "test_length": test_length,
            "test_manifest_path": test_manifest_path,
            "seed": seed,
            "multi_class": multi_class,
            **parameters,
        }
    )
    return model.val(
        validator=CopyPasteYOLOValidator,
        data=str(data_file),
        split="test",
        imgsz=image_size,
        batch=batch_size,
        workers=workers,
        device=device,
        plots=True,
        project=str(output_dir),
        name="test_metrics",
        exist_ok=True,
    )


def main():
    if RUN_YOLO_TRAINING:
        train_yolo_copy_paste(
            model_id=YOLO_MODEL_ID,
            cutout_dir=CUTOUT_DIR,
            background_dir=BACKGROUND_DIR,
            image_size=MODEL_INPUT_SIZE,
            train_length=TRAIN_IMAGES_PER_EPOCH,
            val_length=VAL_IMAGES,
            test_length=TEST_IMAGES,
            test_manifest_path=TEST_MANIFEST_PATH,
            epochs=YOLO_EPOCHS,
            batch_size=BATCH_SIZE,
            workers=NUM_WORKERS,
            seed=SEED,
            project=YOLO_OUTPUT_DIR,
            min_macroinvertebrates=MIN_MACROINVERTEBRATES,
            max_macroinvertebrates=MAX_MACROINVERTEBRATES,
            padding=PADDING,
            max_rotation=MAX_ROTATION,
            cutout_scale_range=CUTOUT_SCALE_RANGE,
            allow_overlap=ALLOW_OVERLAP,
            iou_max=IOU_MAX,
            multi_class=MULTI_CLASS,
            use_noisy_backgrounds=USE_NOISY_BACKGROUNDS,
        )
        return

    datasets, loaders = create_copy_paste_dataloaders(
        cutout_dir=CUTOUT_DIR,
        background_dir=BACKGROUND_DIR,
        image_size=MODEL_INPUT_SIZE,
        train_length=TRAIN_IMAGES_PER_EPOCH,
        val_length=VAL_IMAGES,
        test_length=TEST_IMAGES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        test_manifest_path=TEST_MANIFEST_PATH,
        min_macroinvertebrates=MIN_MACROINVERTEBRATES,
        max_macroinvertebrates=MAX_MACROINVERTEBRATES,
        padding=PADDING,
        max_rotation=MAX_ROTATION,
        cutout_scale_range=CUTOUT_SCALE_RANGE,
        allow_overlap=ALLOW_OVERLAP,
        iou_max=IOU_MAX,
        multi_class=MULTI_CLASS,
        use_noisy_backgrounds=USE_NOISY_BACKGROUNDS,
    )

    set_dataloader_epoch(loaders["train"], 0)
    batch = next(iter(loaders["train"]))
    print(f"Classes : {datasets['train'].class_names}")
    print(f"Images : {tuple(batch['inputs'].shape)}")
    print(f"Boîtes : {tuple(batch['targets']['boxes'].shape)}")
    print(f"Seeds  : {batch['sample_seeds']}")


if __name__ == "__main__":
    main()
