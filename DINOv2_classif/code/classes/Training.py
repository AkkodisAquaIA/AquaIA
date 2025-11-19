import sys
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import json
import matplotlib
matplotlib.use('Agg') 
from tqdm import tqdm
from classes.TiffDatasetLoader import TiffDatasetLoader
from classes.ParamConverter import ParamConverter
from classes.TrainingLogger import TrainingLogger
from classes.model_registry import model_mapping
from datetime import datetime
from torch.optim import Adagrad, Adam, AdamW, NAdam, RMSprop, RAdam, SGD
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex, MulticlassF1Score, BinaryF1Score, BinaryAccuracy, MulticlassAccuracy, BinaryAveragePrecision, MulticlassAveragePrecision, BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryPrecision, MulticlassPrecision, BinaryRecall, MulticlassRecall
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, NLLLoss
from torch.utils.data import DataLoader
from early_stopping_pytorch import EarlyStopping
from collections import defaultdict
from classes.losses import FocalLoss, LDAMLoss

from collections import Counter
from torch.utils.data import WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Training:

    def __repr__(self):
        """
        Returns a string representation of the Training class.

        Returns:
            str: A string indicating the class name.
        """
        return 'Training'

    def __init__(self, **kwargs):
        """
        Initializes the Training class with given parameters.

        Args:
            **kwargs: Keyword arguments containing configuration parameters such as:
                - subfolders (list): List of subfolder names containing the dataset.
                - data_dir (str): Directory containing the dataset.
                - run_dir (str): Directory for saving model outputs.
                - hyperparameters (Hyperparameters): An object containing model and training parameters.

        Raises:
            Exception: If pathLogDir is not provided.
        """
        self.optimizer_mapping = {
            'Adagrad' : Adagrad, 
            'Adam' : Adam, 
            'AdamW' : AdamW, 
            'NAdam' : NAdam, 
            'RMSprop' : RMSprop, 
            'RAdam' : RAdam, 
            'SGD' : SGD
        }

        self.loss_mapping = {
            'CrossEntropyLoss': nn.CrossEntropyLoss,
            'FocalLoss': FocalLoss,
            'LDAMLoss': LDAMLoss,
        }

        self.scheduler_mapping = {
            'LRScheduler': LRScheduler,
            'LambdaLR': LambdaLR,
            'MultiplicativeLR': MultiplicativeLR,
            'StepLR': StepLR,
            'MultiStepLR': MultiStepLR,
            'ConstantLR': ConstantLR,
            'LinearLR': LinearLR,
            'ExponentialLR': ExponentialLR,
            'PolynomialLR': PolynomialLR,
            'CosineAnnealingLR': CosineAnnealingLR,
            'SequentialLR': SequentialLR,
            'ReduceLROnPlateau': ReduceLROnPlateau,
            'CyclicLR': CyclicLR,
            'OneCycleLR': OneCycleLR,
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts
        }
        
        self.param_converter = ParamConverter()  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.subfolders = kwargs.get('subfolders')
        self.data_dir = kwargs.get('data_dir')
        self.run_dir = kwargs.get('run_dir')
        self.hyperparameters = kwargs.get('hyperparameters')
       
        #Model parameters
        self.model_params = {k: v for k, v in self.hyperparameters.get_parameters()['Model'].items()}
        # Sort to ensure consistent ordering across runs
        self.subfolders = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        # Create mapping from class name to integer index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.subfolders)}
        self.num_classes = len(self.subfolders)
        print("num classes = ", self.num_classes)
        self.model_mapping = model_mapping
        self.model = self.initialize_model()
       
        #Optimizer parameters
        self.optimizer_params = {k: v for k, v in self.hyperparameters.get_parameters()['Optimizer'].items()}
        self.optimizer = self.initialize_optimizer()
        
        #Scheduler parameters
        self.scheduler_params = {k: v for k, v in self.hyperparameters.get_parameters()['Scheduler'].items()}
        self.scheduler = self.initialize_scheduler(optimizer=self.optimizer)

        #Loss parameters
        self.loss_params = {k: v for k, v in self.hyperparameters.get_parameters()['Loss'].items()}
        self.weights = self.param_converter._convert_param(self.loss_params.get('weights', "False"))
        self.ignore_index = -100
        self.drw_start_epoch =  self.param_converter._convert_param(self.loss_params.get('drw_start_epoch', 160))
        self.beta_values = [self.param_converter._convert_param(self.loss_params.get('beta_0', 0.0)),
                            self.param_converter._convert_param(self.loss_params.get('beta_1', 0.9999))]
            
        # Training parameters
        self.training_params = {k: v for k, v in self.hyperparameters.get_parameters()['Training'].items()}
        self.batch_size = self.param_converter._convert_param(self.training_params.get('batch_size', 8))
        self.val_split = self.param_converter._convert_param(self.training_params.get('val_split', 0.8))
        self.epochs = self.param_converter._convert_param(self.training_params.get('epochs', 10))
        self.early_stopping = self.param_converter._convert_param(self.training_params.get('early_stopping', "False"))
        self.metrics_str = self.param_converter._convert_param(self.training_params.get('metrics', ''))        
       
        # Data parameters
        self.data = {k: v for k, v in self.hyperparameters.get_parameters()['Data'].items()}
        self.img_res = self.param_converter._convert_param(self.data.get('img_res', 560))
        self.num_samples = self.param_converter._convert_param(self.data.get('num_samples', 500))

   
        if 'Data_augmentation' in self.hyperparameters.get_parameters():
            self.data_aug_settings = {k: v for k, v in self.hyperparameters.get_parameters()['Data_augmentation'].items()}
            self.augmentation_mapping = {
            'brightness': self.param_converter._convert_param(self.data_aug_settings.get('brightness', 0)),
            'contrast': self.param_converter._convert_param(self.data_aug_settings.get('contrast', [1.0, 1.0])),
            'angle': self.param_converter._convert_param(self.data_aug_settings.get('angle', [0, 0])),
            'translate': self.param_converter._convert_param(self.data_aug_settings.get('translate', [0, 0])),
            'scale': self.param_converter._convert_param(self.data_aug_settings.get('scale', [0.0, 0.0])),
            'shear': self.param_converter._convert_param(self.data_aug_settings.get('shear', [0, 0])),
            'random_resized_crop_scale': self.param_converter._convert_param(self.data_aug_settings.get('random_resized_crop_scale', [0.85, 1.0])),
            'random_resized_crop_ratio': self.param_converter._convert_param(self.data_aug_settings.get('random_resized_crop_ratio', [0.9, 1.1])),
            'elastic_alpha': self.param_converter._convert_param(self.data_aug_settings.get('elastic_alpha', 30)),
            'elastic_sigma': self.param_converter._convert_param(self.data_aug_settings.get('elastic_sigma', 5)),
            'elastic_affine': self.param_converter._convert_param(self.data_aug_settings.get('elastic_affine', 5)),
            'color_jitter': self.param_converter._convert_param(self.data_aug_settings.get('color_jitter', [0.2, 0.2, 0.15, 0.05])),
            'gauss_noise': self.param_converter._convert_param(self.data_aug_settings.get('gauss_noise', [5.0, 20.0])),
            'coarse_dropout': self.param_converter._convert_param(self.data_aug_settings.get('coarse_dropout', [3, 6])),
            'blur': self.param_converter._convert_param(self.data_aug_settings.get('blur', 3)),
            'horizontal_flip_p': self.param_converter._convert_param(self.data_aug_settings.get('horizontal_flip_p', 0.5)),
            'vertical_flip_p': self.param_converter._convert_param(self.data_aug_settings.get('vertical_flip_p', 0.2))
    }
        else:
            self.augmentation_mapping = False
     
        self.training_time = datetime.now().strftime("%d-%m-%y-%H-%M-%S")

        if self.early_stopping:
            patience = int(self.epochs*0.2)
            if patience > 1:
                self.early_stopping_instance = EarlyStopping(patience=patience, verbose=True)
            else:
                self.early_stopping=False
                print("Early stopping has been automatically disabled because the patience value is too low.")
                print("Training will begin as normal.")

        self.save_directory = self.create_unique_folder()

                
        self.logger = TrainingLogger(save_directory=self.save_directory,
                                            num_classes=self.num_classes,
                                            model_params=self.model_params,
                                            optimizer_params=self.optimizer_params,
                                            scheduler_params=self.scheduler_params,
                                            loss_params=self.loss_params,
                                            training_params=self.training_params,
                                            data=self.data,
                                            augmentation_mapping=self.augmentation_mapping)
        
                                        
                                    
    def create_metric(self, binary_metric, multiclass_metric):
        name = multiclass_metric.__name__
        if self.num_classes == 1:
            return binary_metric(ignore_index=self.ignore_index).to(self.device)

        # If it’s a confusion matrix, don’t pass `average`
        if name == "MulticlassConfusionMatrix":
            return multiclass_metric(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index
            ).to(self.device)

        # Otherwise, include average='weighted'
        return multiclass_metric(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            #average="weighted"
        ).to(self.device)


    def initialize_metrics(self):
        """
        Initializes the specified metrics for evaluation.
        
        Returns:
            list: A list of metric instances corresponding to the specified names.
        
        Raises:
            ValueError: If a specified metric is not recognized.
        """
        self.metrics_mapping = {
            "Jaccard": self.create_metric(BinaryJaccardIndex, MulticlassJaccardIndex),
            "F1": self.create_metric(BinaryF1Score, MulticlassF1Score),
            "Accuracy": self.create_metric(BinaryAccuracy, MulticlassAccuracy),
            "AveragePrecision": self.create_metric(BinaryAveragePrecision, MulticlassAveragePrecision),
            "ConfusionMatrix": self.create_metric(BinaryConfusionMatrix, MulticlassConfusionMatrix),
            "Precision": self.create_metric(BinaryPrecision, MulticlassPrecision),
            "Recall": self.create_metric(BinaryRecall, MulticlassRecall),
        }
    
        # Parse metrics from string input or use default
        self.metrics = [metric.strip() for metric in self.metrics_str.split(',')] if self.metrics_str else ["Jaccard"]
    
        # Retrieve metric instances
        selected_metrics = []
        for metric in self.metrics:
            if metric in self.metrics_mapping:
                selected_metrics.append(self.metrics_mapping[metric])
            else:
                raise ValueError(f"Metric '{metric}' not recognized. Please check the name.")
    
        return selected_metrics

    def initialize_optimizer(self):
        optimizer_name = self.optimizer_params.get('optimizer', 'Adam')
        optimizer_class = self.optimizer_mapping.get(optimizer_name)

        if not optimizer_class:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Check your 'optimizer_mapping'.")

        converted_params = {k: self.param_converter._convert_param(v) for k, v in self.optimizer_params.items() if k != 'optimizer'}

        return optimizer_class(self.model.parameters(), **converted_params)

    def initialize_scheduler(self, optimizer):
        scheduler_name = self.scheduler_params.get('scheduler', 'ConstantLR')
        scheduler_class = self.scheduler_mapping.get(scheduler_name)

        if not scheduler_class:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported. Check your 'scheduler_mapping'.")

        converted_params = {k: self.param_converter._convert_param(v) for k, v in self.scheduler_params.items() if k != 'scheduler'}

        if not converted_params:
            return scheduler_class(optimizer)
        else:
            return scheduler_class(optimizer, **converted_params)

    def initialize_loss(self, **dynamic_params):
        loss_name = self.loss_params.get('loss', 'CrossEntropyLoss')
        loss_class = self.loss_mapping.get(loss_name)
    
        if not loss_class:
            raise ValueError(f"Loss '{loss_name}' is not supported. Check your 'loss_mapping'.")
    
        # Exclude control-only INI params
        exclude_keys = {'loss', 'ignore_background', 'weights', 'drw_start_epoch', 'beta_0', 'beta_1'}
    
        # Convert static parameters from config
        converted_params = {
            k: self.param_converter._convert_param(v)
            for k, v in self.loss_params.items()
            if k not in exclude_keys
        }
    
        # Merge static + dynamic, but strip excluded from dynamic as well
        filtered_dynamic = {k: v for k, v in dynamic_params.items() if k not in exclude_keys}
        final_params = {**converted_params, **filtered_dynamic}
    
        if self.num_classes > 1 and loss_name != "LDAMLoss":
            final_params['ignore_index'] = self.ignore_index
        else:
            final_params.pop('ignore_index', None)
    
        return loss_class(**final_params)

    def initialize_model(self) -> nn.Module:
        model_name = self.model_params.get('model_type', 'UnetVanilla')

        if model_name not in self.model_mapping:
            raise ValueError(f"Model '{model_name}' is not supported. Check your 'model_mapping'.")

        model_class = self.model_mapping[model_name]
        self.model_params['num_classes'] = self.num_classes

        required_params = {
            k: self.param_converter._convert_param(v) for k, v in self.model_params.items() if k in model_class.REQUIRED_PARAMS
        }
        optional_params = {
            k: self.param_converter._convert_param(v) for k, v in self.model_params.items() if k in model_class.OPTIONAL_PARAMS
        }

        required_params.pop('model_type', None)
        optional_params.pop('model_type', None)

        try:
            typed_required_params = {
                k: model_class.REQUIRED_PARAMS[k](v) 
                for k, v in required_params.items()
            }

            typed_optional_params = {
                k: model_class.OPTIONAL_PARAMS[k](v) 
                for k, v in optional_params.items()
            }
        except ValueError as e:
            raise ValueError(f"Error converting parameters for model '{model_name}': {e}")

        return model_class(**typed_required_params, **typed_optional_params).to(self.device)

    def create_unique_folder(self):
        """
        Creates a unique folder for saving model weights and logs based on the current training parameters.

        Returns:
            str: The path to the created directory.
        """
        filename = f"{self.model_params.get('model_type', 'UnetVanilla')}__" \
            f"{self.training_time}"

        save_directory = os.path.join(self.run_dir, filename)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        return save_directory
        
    def load_segmentation_data(self):
        import os, json, glob
        import numpy as np
        from collections import Counter
        from torch.utils.data import DataLoader
    
        def load_data_stats(data_dir):
            neutral_stats = [np.array([0.5] * 3), np.array([0.5] * 3)]
            stats_file = os.path.join(data_dir, 'data_stats.json')
            if not os.path.exists(stats_file):
                print(f"⚠️ {stats_file} not found. Using default normalization stats.")
                return {"default": neutral_stats}
            try:
                with open(stats_file, 'r') as f:
                    raw = json.load(f)
                return {
                    k: [np.array(v[0]), np.array(v[1])] for k, v in raw.items()
                    if isinstance(v, list) and len(v) == 2
                }
            except Exception as e:
                print(f"⚠️ Failed to load data stats: {e}. Using default.")
                return {"default": neutral_stats}
    
        def stratified_split_forced(subfolders, img_data, class_to_idx, repeat_factors, val_split=0.3, seed=42, max_total=10000):
            rng = np.random.RandomState(seed)
            
            train, val, test = [], [], []
            common_candidates = []
            
            for cls in subfolders:
                idx = class_to_idx[cls]
                paths = img_data[cls]
                n = len(paths)
                factor = repeat_factors.get(cls, 1)
            
                if n < 3:
                    print(f"⚠️ Class '{cls}' skipped (n = {n} < 3)")
                    continue
            
                indices = list(range(n))
                rng.shuffle(indices)
            
                if n == 3:
                    train.append((idx, indices[0]))
                    val.append((idx, indices[1]))
                    test.append((idx, indices[2]))
                elif factor >= 6:
                    n_test = max(1, int(round(0.3 * n)))
                    n_val = max(1, int(round(0.3 * n)))
                    n_train = n - n_val - n_test
            
                    train += [(idx, i) for i in indices[:n_train]]
                    val += [(idx, i) for i in indices[n_train:n_train + n_val]]
                    test += [(idx, i) for i in indices[n_train + n_val:]]
                else:
                    # Common classes: defer actual split after sampling up to max_total
                    for i in indices:
                        common_candidates.append((cls, idx, i))
            
            # Shuffle and truncate common_candidates globally to fit remaining slots
            rng.shuffle(common_candidates)
            remaining_slots = max_total - (len(train) + len(val))
            sampled_common = common_candidates[:remaining_slots]
            
            # Group common samples by class for split
            class_groups = {}
            for cls, idx, i in sampled_common:
                class_groups.setdefault(cls, []).append((idx, i))
            
            for cls, items in class_groups.items():
                idx = items[0][0]
                indices = [i for (_, i) in items]
                rng.shuffle(indices)
                n = len(indices)
            
                if n == 3:
                    train.append((idx, indices[0]))
                    val.append((idx, indices[1]))
                    test.append((idx, indices[2]))
                elif n > 3:
                    n_test = max(1, int(round(0.2 * n)))
                    remaining = n - n_test
                    n_val = max(1, int(round(val_split * remaining)))
                    n_train = remaining - n_val
            
                    train += [(idx, i) for i in indices[:n_train]]
                    val += [(idx, i) for i in indices[n_train:n_train + n_val]]
                    test += [(idx, i) for i in indices[n_train + n_val:]]
            
            rng.shuffle(train)
            rng.shuffle(val)
            rng.shuffle(test)
            
            print(f"✅ stratified_split_forced complete: {len(train)} train / {len(val)} val / {len(test)} test")
            return train, val, test

    
        def oversample_indices(indices, tag, repeat_factors):
            class_counts = Counter([c for (c, _) in indices])
            oversampled = []
            for cls, count in class_counts.items():
                samples = [(c, i) for (c, i) in indices if c == cls]
                cls_name = [k for k, v in self.class_to_idx.items() if v == cls][0]
                factor = repeat_factors.get(cls_name, 1)
                oversampled.extend(samples * factor)
            print(f"✅ Oversampled {tag}: {len(indices)} → {len(oversampled)}")
            return oversampled
    
        def print_class_distribution(indices, tag):
            counts = Counter([c for (c, _) in indices])
            print(f"\n📊 {tag.upper()} class distribution:")
            for cls_idx in sorted(counts):
                cls_name = [k for k, v in self.class_to_idx.items() if v == cls_idx][0]
                print(f"  [{cls_idx:02d}] {cls_name:30s} | samples: {counts[cls_idx]:5d}")
    
        # --- Load image data ---
        img_data = {}
        img_labels = {}
        num_samples_per_class = {}
        data_stats = load_data_stats(self.data_dir)
    
        for cls in self.subfolders:
            paths = sorted(glob.glob(os.path.join(self.data_dir, cls, "*")))
            img_data[cls] = paths
            label_idx = self.class_to_idx[cls]
            img_labels[cls] = [label_idx] * len(paths)
            num_samples_per_class[cls] = len(paths)
    
        self.cls_num_list = [num_samples_per_class.get(cls, 0) for cls in self.class_to_idx]
    
        # --- Compute per-class repeat factor ---
        max_count = max(num_samples_per_class.values())
        median_size = np.median(list(num_samples_per_class.values()))

        class_repeat_factor = {}
        for cls, count in num_samples_per_class.items():
            if count == 0:
                factor = 1
            else:
                rarity_ratio = median_size / count
                factor = int(np.ceil(rarity_ratio * 2))  # Tunable sensitivity
                factor = min(max(factor, 1), 10)        # Clamp between 1 and 10
            class_repeat_factor[cls] = factor
            
        print("\n📈 Oversampling factors:")
        for cls in sorted(class_repeat_factor, key=lambda c: self.class_to_idx[c]):
            idx = self.class_to_idx[cls]
            print(f"  [{idx:02d}] {cls:30s} | imgs: {num_samples_per_class[cls]:4d} | factor: {class_repeat_factor[cls]:2d}")
    
        # --- Albumentations p_scale ---
        max_factor = max(class_repeat_factor.values())
        class_p_scale = {
            cls: round(min(2.0, 0.5 + (factor / max_factor)), 2)
            for cls, factor in class_repeat_factor.items()
        }
    
        # --- Stratified sampling with forced rare classes ---
        train_idx, val_idx, test_idx = stratified_split_forced(
            self.subfolders, img_data, self.class_to_idx,
            repeat_factors=class_repeat_factor,
            val_split=1 - self.val_split,
            seed=42, max_total=self.num_samples
        )
    
        print_class_distribution(train_idx, "train (before)")
        print_class_distribution(val_idx, "val (before)")
        print_class_distribution(test_idx, "test (before)")
    
        train_idx = oversample_indices(train_idx, "train", class_repeat_factor)
        val_idx   = oversample_indices(val_idx, "val", class_repeat_factor)
    
        print_class_distribution(train_idx, "train (after)")
        print_class_distribution(val_idx, "val (after)")
    
        train_dataset = TiffDatasetLoader(
            indices=train_idx,
            img_data=img_data,
            img_labels=img_labels,
            data_stats=data_stats,
            img_res=self.img_res,
            weights=self.weights,
            augmentation_params=self.augmentation_mapping,
            rare_classes=class_p_scale
        )
    
        val_dataset = TiffDatasetLoader(
            indices=val_idx,
            img_data=img_data,
            img_labels=img_labels,
            data_stats=data_stats,
            img_res=self.img_res,
            weights=self.weights,
            augmentation_params=self.augmentation_mapping,
            rare_classes=class_p_scale
        )
    
        test_dataset = TiffDatasetLoader(
            indices=test_idx,
            img_data=img_data,
            img_labels=img_labels,
            data_stats=data_stats,
            img_res=self.img_res,
            weights=self.weights,
            augmentation_params=None,
            rare_classes=None
        )
    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)
        test_loader  = DataLoader(test_dataset,  batch_size=10,              shuffle=False, num_workers=2, drop_last=True)
    
        self.logger.save_indices_to_file([train_idx, val_idx, test_idx])
        self.logger.save_class_p_scale(class_p_scale)

        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'indices': [train_idx, val_idx, test_idx]
        }
    

    def training_loop(self, optimizer, scheduler):

        def print_epoch_box(epoch, total_epochs):
            epoch_str = f" Epoch {epoch}/{total_epochs} "
            box_width = len(epoch_str) + 4
            print(f"╔{'═' * (box_width - 2)}╗")
            print(f"║{epoch_str.center(box_width - 2)}║")
            print(f"╚{'═' * (box_width - 2)}╝")
    
        # AMP & cuDNN
        scaler = None
        if self.device == "cuda":
            scaler = torch.amp.GradScaler()
            torch.backends.cudnn.benchmark = True
    
        metrics = self.initialize_metrics()  # e.g. accuracy, f1, etc.
        loss_dict = {"train": {}, "val": {}}
        display_metrics = [m for m in self.metrics if m != "ConfusionMatrix"]
        metrics_dict = {phase: {m: [] for m in display_metrics} for phase in ["train", "val"]}
        best_val_loss = float("inf")
        best_val_metrics = {m: 0.0 for m in display_metrics}
        
        for epoch in range(1, self.epochs + 1):
            print_epoch_box(epoch, self.epochs)
        
        
            if self.loss_params.get('loss') == "LDAMLoss" and self.weights:
                idx = 0 if epoch <  self.drw_start_epoch else 1
                beta =  self.beta_values[idx]
                effective_num = 1.0 - np.power(beta, self.cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
                class_weights = torch.FloatTensor(per_cls_weights).to(self.device)
    
                loss_fn = self.initialize_loss(weight=class_weights, cls_num_list=self.cls_num_list)
                print("ok")
            elif self.weights:
                train_dataset = self.dataloaders["train"].dataset  
                class_weights = train_dataset.class_weights.to(self.device)
                loss_fn = self.initialize_loss(weight=class_weights)
            else:
                loss_fn = self.initialize_loss()

            for phase in ["train", "val"]:
                is_train = (phase == "train")
                self.model.train() if is_train else self.model.eval()

                running_loss = 0.0
                running_metrics = {m: 0.0 for m in display_metrics}
                total_samples = 0

                with tqdm(total=len(self.dataloaders[phase]), unit="batch") as pbar:
                    for inputs, labels in self.dataloaders[phase]:
                        # move to device
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device).long()

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(is_train):
                            # always run forward under autocast
                            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                                outputs = self.model(inputs)  # (B, num_classes)

                                # Visualize first sample in the batch
                                if epoch == 1 and total_samples < self.batch_size:  # Only once, for clarity
                                    import matplotlib.pyplot as plt
                                
                                    input_img = inputs[0].detach().cpu()  # C x H x W
                                    label_val = labels[0].item()
                                    pred_val = outputs.argmax(dim=1)[0].item()
                                
                                    # Convert C x H x W → H x W x C and to [0,1] range
                                    img_np = input_img.permute(1, 2, 0).numpy()
                                    img_np = np.clip(img_np, 0, 1)
                                
                                    plt.imshow(img_np)
                                    plt.title(f"Label: {label_val} | Pred: {pred_val}")
                                    plt.axis("off")
                                    plt.show()

                            loss = loss_fn(outputs.float(), labels)

                            if is_train:
                                if scaler:
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    loss.backward()
                                    optimizer.step()

                                # step scheduler
                                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    scheduler.step(loss)
                                else:
                                    scheduler.step()

                        batch_size = labels.size(0)
                        running_loss += loss.item() * batch_size
                        total_samples  += batch_size

                        # metrics
                        with torch.no_grad():
                            preds = outputs.argmax(dim=1)
                            for name, fn in zip(self.metrics, metrics):
                                if name in running_metrics:
                                    running_metrics[name] += fn(preds, labels).item() * batch_size

                        pbar.set_postfix(
                            loss=running_loss / total_samples,
                            **{m: running_metrics[m] / total_samples for m in display_metrics}
                        )
                        pbar.update(1)

                # end of phase
                epoch_loss = running_loss / total_samples
                epoch_metrics = {m: running_metrics[m] / total_samples for m in display_metrics}
                loss_dict[phase][epoch] = epoch_loss
                for m, val in epoch_metrics.items():
                    metrics_dict[phase][m].append(val)

                # print summary
                print(f"{phase.title()} Loss: {epoch_loss:.4f}", end=" | ")
                for m, val in epoch_metrics.items():
                    print(f"{phase.title()} {m}: {val:.4f}", end=" | ")
                print()

                # checkpoint on best val
                if phase == "val":
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        torch.save(self.model.state_dict(),
                                os.path.join(self.save_directory, "model_best_loss.pth"))
                    for m, val in epoch_metrics.items():
                        if val > best_val_metrics[m]:
                            best_val_metrics[m] = val
                            torch.save(self.model.state_dict(),
                                    os.path.join(self.save_directory, f"model_best_{m}.pth"))

            # early stopping
            if self.early_stopping:
                self.early_stopping_instance(epoch_loss, self.model)
                if self.early_stopping_instance.early_stop:
                    print("Early stopping triggered")
                    break

        print(f"Best Validation Metrics: {best_val_metrics}")
        return loss_dict, metrics_dict, metrics


    def train(self):
            loss_dict, metrics_dict, metrics = self.training_loop(optimizer=self.optimizer, 
                                                                scheduler=self.scheduler)
            
            #plot and metric saving
            self.logger.save_best_metrics(loss_dict=loss_dict, 
                                        metrics_dict=metrics_dict)
            self.logger.plot_learning_curves(loss_dict=loss_dict, 
                                            metrics_dict=metrics_dict)
            self.logger.save_hyperparameters()
            self.logger.save_data_stats(self.dataloaders["train"].dataset.data_stats)

            if "ConfusionMatrix" in self.metrics:
                self.logger.save_confusion_matrix(conf_metric=metrics[self.metrics.index("ConfusionMatrix")], 
                                                model=self.model, 
                                                val_dataloader=self.dataloaders["val"], 
                                                device=self.device)
