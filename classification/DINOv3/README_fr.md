
# Pipeline DINOv3

Ce projet implémente un pipeline complet d'entraînement, de validation et d'inférence pour un classifieur d'images basé sur **DINOv3** (Hugging Face Transformers) et **PyTorch**.

## Structure du projet

```text
.
├── train_dinov3.py
├── infer_dinov3.py
├── common_dinov3.py
├── losses.py
└── README.md
```

## Description des scripts

### `train_dinov3.py`

Script principal d'entraînement.

Fonctionnalités :

- Chargement des jeux `train/` et `val/`
- Prétraitement avec `AutoImageProcessor`
- Création du modèle DINOv3
- Gel du backbone puis dégel des *N* derniers blocs
- Entraînement avec **CrossEntropy** ou **Focal Loss**
- Scheduler (CosineAnnealingLR ou ReduceLROnPlateau)
- Early Stopping
- Sauvegarde des checkpoints
- Logs TensorBoard

### `common_dinov3.py`

Contient les fonctions communes :

- `DinoV3Classifier`
- gestion des checkpoints
- `freeze_all()`
- `unfreeze_last_n_blocks()`
- gestion des seeds
- informations système
- écriture des fichiers JSON

### `losses.py`

Implémentation d'une **Focal Loss** multi-classes compatible avec :

- alpha scalaire
- alpha par classe
- `ignore_index`
- différentes réductions (`mean`, `sum`, `none`)

### `infer_dinov3.py`

Script d'inférence.

Deux modes sont disponibles :

- **dataset labellisé** (`LABELED_BY_SUBFOLDER=True`)
- **dataset non labellisé** (`False`)

Le script recharge automatiquement :

- le checkpoint (`best.pt` ou `last.pt`)
- le modèle DINOv3
- les classes
- la configuration d'entraînement

## Structure du dataset

```text
dataset/
├── train/
│   ├── classe1/
│   ├── classe2/
│   └── ...
├── val/
│   ├── classe1/
│   ├── classe2/
│   └── ...
└── test/
    ├── classe1/
    ├── classe2/
    └── ...
```

Les classes doivent être identiques entre **train** et **val**.

## Variables d'environnement

```bash
export HF_TOKEN="xxxxxxxx"
export DATA_DIR="/path/to/dataset"
export RUN_DIR="/path/to/results"
export EPOCHS=30
export UNFREEZE=1
export EARLY_PATIENCE=15
```

## Entraînement

```bash
python train_dinov3.py
```

Le pipeline :

1. charge le dataset ;
2. construit le modèle DINOv3 ;
3. gèle le backbone ;
4. dégèle les derniers blocs ;
5. entraîne le modèle ;
6. valide à chaque époque ;
7. sauvegarde `best.pt` et `last.pt`.

## Fonctions de perte

Deux pertes sont disponibles :

```python
loss_name = "ce"
```

ou

```python
loss_name = "focal"
```

La Focal Loss peut utiliser un **alpha calculé automatiquement** à partir de la distribution des classes.

## Checkpoints générés

```text
RUN_DIR/
└── YYYYMMDD-HHMMSS/
    ├── config.json
    ├── class_to_idx.json
    ├── results_trainval.json
    ├── tb/
    └── checkpoints/
        ├── best.pt
        └── last.pt
```

## Inférence

```bash
python infer_dinov3.py
```

Le script produit :

- `metrics.json`
- `predictions.csv`
- `classification_report.txt`
- `confusion_matrix.npy`
- `confusion_matrix.png`
- logs TensorBoard

## TensorBoard

Entraînement :

```bash
tensorboard --logdir <RUN_DIR>
```

Inférence :

```bash
tensorboard --logdir <RUN_DIR>/inference_outputs/tb
```

## Remarques

- `HF_TOKEN` est obligatoire pour charger DINOv3.
- `best.pt` correspond au modèle ayant obtenu la meilleure **validation loss**.
- La Focal Loss améliore la gestion des jeux de données déséquilibrés.
- Le mapping `class_to_idx` est sauvegardé dans chaque checkpoint afin de garantir la cohérence entre entraînement et inférence.
