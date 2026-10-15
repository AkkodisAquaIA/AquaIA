# Guide du Dataset copy-paste en mémoire

Ce guide explique comment les images sont générées pendant l'entraînement, comment
la reproductibilité est assurée et comment entraîner puis tester un modèle YOLO.

Pour une utilisation interactive, `yolo_validation.ipynb` regroupe les paramètres,
la prévisualisation des données, l'entraînement, les métriques du test et les
prédictions dans un notebook.

La cellule de comparaison aléatoire utilise `RANDOM_GENERATED_IMAGES` pour afficher
autant de nouvelles compositions que souhaité, avec vérité terrain et prédictions
côte à côte. La relancer suffit pour obtenir une nouvelle seed et de nouvelles images.

## Principe général

`copy_paste_dataset.py` ne crée pas un dossier contenant des centaines de grandes
images. Il crée chaque image en mémoire au moment où PyTorch la demande :

```text
fond + cutouts PNG
        ↓
rotation, redimensionnement et placement
        ↓
image à la taille d'entrée du modèle + boîtes YOLO
        ↓
batch PyTorch
        ↓
modèle
```

Les PNG transparents des insectes restent enregistrés à leur résolution maximale.
La composition finale est directement créée à la résolution du modèle.

Avec `MODEL_INPUT_SIZE = 1024`, aucune composition intermédiaire de
`6336 × 6336` n'est créée.

## Images virtuelles et époques

```python
TRAIN_IMAGES_PER_EPOCH = 700
VAL_IMAGES = 100
TEST_IMAGES = 200
```

Ces valeurs ne sont pas des nombres de fichiers. Elles définissent combien
d'échantillons le Dataset sait produire :

- le train produit 700 compositions par époque ;
- la validation contient 100 compositions fixes ;
- le test contient 200 compositions fixes.

À chaque appel de `__getitem__`, le Dataset choisit un fond et plusieurs insectes,
les tourne, les place puis calcule les boîtes.

## Reproductibilité

Chaque image possède sa propre seed, calculée à partir de :

```text
seed globale + split + époque + index
```

Pour le train, l'époque est prise en compte. L'image change donc entre les époques,
mais deux entraînements utilisant la même seed reçoivent les mêmes compositions.

Pour val et test, l'époque reste toujours égale à zéro. Les images restent fixes.

Pour comparer deux modèles, conserver les mêmes valeurs pour :

- `SEED` ;
- `MODEL_INPUT_SIZE` ;
- le nombre d'images par split ;
- le nombre et la taille des insectes ;
- le padding et les règles de superposition ;
- le batch size si l'on souhaite également conserver les mêmes batches.

YOLO est lancé avec `deterministic=True`. Les augmentations natives de YOLO sont
désactivées, car la génération copy-paste constitue déjà notre augmentation et doit
rester strictement identique entre les modèles.

## Cache des fonds

Le booléen suivant choisit le type de fond :

```python
USE_NOISY_BACKGROUNDS = True
```

- `True` utilise les huit fonds : les quatre sans bruit et les quatre avec bruit ;
- `False` utilise les quatre premiers fonds sans bruit.

Le notebook crée un manifest de test différent pour chaque choix :

```text
copy_paste_test_manifest_LPL_dataset_cutouts_multi_all_0_15.json
copy_paste_test_manifest_LPL_dataset_cutouts_single_single_clean_0_15.json
```

Le dataset de cutouts, le mode mono ou multiclasse, le type de fond et les nombres
minimum et maximum font partie du nom. Les résultats peuvent ainsi être comparés
sans écraser le test d'une autre configuration.

Les TIFF originaux sont lourds à décoder. Avec `cache_backgrounds=True`, le script :

1. ouvre chaque fond une seule fois ;
2. le réduit à `MODEL_INPUT_SIZE` ;
3. conserve cette petite version en RAM ;
4. partage le même cache entre train, val et test.

À `1024 × 1024`, un fond RGB non compressé représente environ 3 Mo en mémoire. Un
ensemble de huit fonds représente donc environ 24 Mo, au lieu de relire plusieurs
fois les TIFF d'environ 120 Mo.

Mettre `cache_backgrounds=False` uniquement si le nombre de fonds devient trop grand
pour la mémoire disponible.

## Manifest JSON du test

Le premier lancement crée :

```text
datasets/copy_paste_test_manifest.json
```

Il contient :

- tous les paramètres de génération ;
- la liste ordonnée des fonds ;
- la liste ordonnée des cutouts ;
- la seed de chaque image du test.

Le test peut ainsi être régénéré sans enregistrer ses pixels. Ajouter un nouveau
cutout ne modifie pas le test existant, car l'ancienne liste est relue depuis le JSON.

Si les paramètres du test changent, utiliser un nouveau nom de manifest. On peut
aussi supprimer volontairement l'ancien JSON pour créer un nouveau test.

## Format PyTorch générique

La fonction suivante crée les trois DataLoaders :

```python
datasets, loaders = create_copy_paste_dataloaders(
    cutout_dir="datasets/LPL_dataset_cutouts_single",
    background_dir="datasets/Images_fond_blanc_&_bruits",
    image_size=1024,
    train_length=700,
    val_length=100,
    test_length=200,
    batch_size=2,
    num_workers=2,
    seed=42,
    test_manifest_path="datasets/copy_paste_test_manifest.json",
)
```

Dans une boucle PyTorch classique :

```python
for epoch in range(100):
    set_dataloader_epoch(loaders["train"], epoch)

    for batch in loaders["train"]:
        images = batch["inputs"]
        boxes = batch["targets"]["boxes"]
        labels = batch["targets"]["labels"]
```

Les boîtes utilisent le format YOLO normalisé :

```text
centre_x, centre_y, largeur, hauteur
```

## Raccord avec YOLO

YOLO utilise des noms de champs particuliers : `img`, `cls`, `bboxes` et
`batch_idx`. `YOLOCopyPasteDataset` transforme la sortie générique dans ce format.

`CopyPasteYOLOTrainer` remplace ensuite le DataLoader classique d'Ultralytics par
notre DataLoader. La loss, l'optimiseur, la validation, les checkpoints et les
graphiques restent ceux d'Ultralytics.

La taille est récupérée depuis `imgsz`. En utilisant `imgsz=1024`, le générateur
produit directement des images `1024 × 1024`.

## Lancer un entraînement YOLO

### Méthode la plus simple

Dans `copy_paste_dataset.py`, vérifier les paramètres globaux :

```python
CUTOUT_DIR = PROJECT_ROOT / "datasets/LPL_dataset_cutouts_single"
BACKGROUND_DIR = PROJECT_ROOT / "datasets/Images_fond_blanc_&_bruits"

MODEL_INPUT_SIZE = 1024
TRAIN_IMAGES_PER_EPOCH = 700
VAL_IMAGES = 100
TEST_IMAGES = 200

BATCH_SIZE = 2
NUM_WORKERS = 2
SEED = 42

YOLO_MODEL_ID = "yolo26n.pt"
YOLO_EPOCHS = 100
RUN_YOLO_TRAINING = True
```

Puis lancer depuis la racine du projet :

```bash
source .venv/bin/activate
python3 data_augmentation/detection_augmentation/copy_paste_dataset.py
```

Commencer avec `BATCH_SIZE = 2` pour une carte graphique de 6 Go. Si la mémoire GPU
est suffisante, essayer ensuite `4`.

### Depuis un autre script

```python
from data_augmentation.detection_augmentation.copy_paste_dataset import (
    train_yolo_copy_paste,
)

model, metrics = train_yolo_copy_paste(
    model_id="yolo26n.pt",
    cutout_dir="datasets/LPL_dataset_cutouts_single",
    background_dir="datasets/Images_fond_blanc_&_bruits",
    image_size=1024,
    train_length=700,
    val_length=100,
    test_length=200,
    test_manifest_path="datasets/copy_paste_test_manifest.json",
    epochs=100,
    batch_size=2,
    workers=2,
    seed=42,
    project="results/detection/copy_paste_yolo",
    min_macroinvertebrates=4,
    max_macroinvertebrates=10,
    allow_overlap=True,
    iou_max=0.5,
    training_parameters={
        "device": 0,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "plots": True,
    },
)
```

Les poids sont enregistrés par Ultralytics dans :

```text
results/detection/copy_paste_yolo/train/weights/best.pt
results/detection/copy_paste_yolo/train/weights/last.pt
```

Si `train` existe déjà, Ultralytics peut créer `train2`, `train3`, etc.

## Vérifier visuellement pendant l'entraînement

Avec `plots=True`, le dossier du run contient notamment des images telles que :

```text
train_batch0.jpg
val_batch0_labels.jpg
val_batch0_pred.jpg
```

Elles permettent de vérifier les images synthétiques, les vraies boîtes et les
prédictions du modèle.

## Tester `best.pt` sur une image réelle

```python
from data_augmentation.detection_augmentation.copy_paste_dataset import (
    predict_yolo_image,
)

results = predict_yolo_image(
    model_or_weights="results/detection/copy_paste_yolo/train/weights/best.pt",
    image_path="datasets/DATA/LPL_Photos_31072026/mon_image.tif",
    output_dir="results/detection/copy_paste_predictions",
    confidence=0.25,
    image_size=1024,
)
```

L'image avec les boîtes est enregistrée dans :

```text
results/detection/copy_paste_predictions/predictions/
```

Le même appel accepte aussi un dossier complet comme `image_path`.

Dans un notebook, on peut afficher directement la première prédiction :

```python
from PIL import Image

display(Image.fromarray(results[0].plot()[..., ::-1]))
```

`plot()` renvoie une image BGR, d'où `[..., ::-1]` pour revenir en RGB.

## Remarques importantes

- Le test synthétique vérifie les performances sur la distribution du générateur.
- Il faut aussi tester le modèle sur de vraies images mises de côté.
- Les cutouts restent en résolution maximale ; seule leur copie utilisée pour une
  composition est réduite.
- `imgsz` doit rester identique entre l'entraînement et la prédiction pour une
  comparaison cohérente.
- Un batch plus grand accélère parfois l'entraînement, mais augmente fortement la
  mémoire GPU utilisée.
