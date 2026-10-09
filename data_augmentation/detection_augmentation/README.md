# Copy-paste augmentation pour la détection

Ce dossier contient un pipeline simple pour créer un dataset YOLO de détection à
partir de macroinvertébrés détourés. Les insectes sont tournés et placés aléatoirement
sur des fonds réels, puis les bounding boxes YOLO sont générées automatiquement.

Le script principal est `copy_and_paste_augmentation.py`.

Le fichier `copy_paste_dataset.py` propose aussi une génération directement en
mémoire avec un `Dataset` et des `DataLoader` PyTorch. Cette version évite de stocker
les grandes images augmentées sur le disque.

## Génération en mémoire avec PyTorch

Dans `copy_paste_dataset.py`, modifier notamment :

```python
MODEL_INPUT_SIZE = 1024
TRAIN_IMAGES_PER_EPOCH = 700
VAL_IMAGES = 100
TEST_IMAGES = 200
BATCH_SIZE = 4
NUM_WORKERS = 2
SEED = 42
```

`MODEL_INPUT_SIZE` doit correspondre à la taille d'entrée du modèle. Les fonds et
les insectes sont redimensionnés avant le collage : aucune image augmentée de
`6336 x 6336` n'est créée si le modèle travaille en `1024 x 1024`.

Utilisation depuis un entraînement :

```python
from data_augmentation.detection_augmentation.copy_paste_dataset import (
    create_copy_paste_dataloaders,
    set_dataloader_epoch,
)

datasets, loaders = create_copy_paste_dataloaders(
    cutout_dir="datasets/LPL_dataset_cutouts_single",
    background_dir="datasets/Images_fond_blanc_&_bruits",
    image_size=1024,
    train_length=700,
    val_length=100,
    test_length=200,
    batch_size=4,
    num_workers=2,
    seed=42,
    test_manifest_path="datasets/copy_paste_test_manifest.json",
)

for epoch in range(100):
    set_dataloader_epoch(loaders["train"], epoch)
    for batch in loaders["train"]:
        images = batch["inputs"]
        targets = batch["targets"]
```

Avec les mêmes paramètres et la même seed, deux modèles reçoivent les mêmes images
pour une époque donnée. Le train change entre les époques. Val et test restent fixes.

Le manifest JSON du test contient les paramètres, la liste ordonnée des fonds et
cutouts et la seed de chaque échantillon. S'il existe déjà, il est relu : ajouter un
nouveau cutout dans le dossier ne modifie donc pas silencieusement le test. Pour créer
un autre test, utiliser un autre chemin de manifest ou supprimer volontairement
l'ancien fichier.

Les images retournées sont des tenseurs RGB `float32` compris entre 0 et 1. Les
boîtes sont au format YOLO normalisé `(centre_x, centre_y, largeur, hauteur)`. Une
normalisation supplémentaire peut être demandée avec `normalization_mean` et
`normalization_std`.

Cette classe fournit des DataLoaders PyTorch génériques. L'API haut niveau
Ultralytics `model.train(data="dataset.yaml")` attend, elle, un dataset présent sur
le disque : son Trainer devra être adapté séparément pour consommer ce DataLoader.

## Les trois fonctions principales

### 1. `generate_augmented_dataset`

C'est la fonction utilisée par défaut. Elle prend :

- un dossier de cutouts PNG transparents ;
- un dossier de fonds PNG ou TIFF sans perte ;
- un dossier dans lequel enregistrer le dataset final.

Elle crée les images augmentées, les annotations YOLO, les fichiers de classes, les
YAML et quelques aperçus avec les bounding boxes.

### 2. `create_cutouts_sam3`

Cette fonction est prévue pour les grandes images contenant plusieurs insectes.
SAM3 détecte les différentes instances, refait une segmentation avec du contexte,
puis GrabCut affine les contours sur l'image en résolution native.

Les PNG transparents sont enregistrés dans un sous-dossier de classe nommé
`macroinvertebrates` par défaut.

SAM3 nécessite un accès autorisé au modèle `facebook/sam3` sur Hugging Face et une
connexion préalable avec `huggingface-cli login`.

### 3. `create_cutouts_sam2_otsu`

Cette fonction est prévue pour des images déjà cropées, avec normalement un seul
insecte par image. Otsu donne un premier masque qui sert de prompt à SAM2.

L'arborescence des classes du dossier source est conservée dans le dossier de sortie.

## Utilisation rapide

Modifier les paramètres globaux au début du script, notamment :

```python
CUTOUT_DIR = PROJECT_ROOT / "datasets/LPL_dataset_cutouts_single"
BACKGROUND_DIR = PROJECT_ROOT / "datasets/Images_fond_blanc_&_bruits"
OUTPUT_DIR = PROJECT_ROOT / "datasets/LPL_datasets_augment"

NUMBER_OF_IMAGES = 1000
MIN_MACROINVERTEBRATES = 4
MAX_MACROINVERTEBRATES = 10
ALLOW_OVERLAP = True
IOU_MAX = 1.0

CREATE_SPLITS = True
SPLIT_RATIOS = [70, 10, 20]
MULTI_CLASS = False
CREATE_VALIDATION = True
OUTPUT_IMAGE_FORMAT = "png"
```

Puis lancer depuis la racine du projet :

```bash
python3 data_augmentation/detection_augmentation/copy_and_paste_augmentation.py
```

Par défaut, seule la génération du dataset augmenté est lancée. Les deux appels de
segmentation sont laissés en commentaire dans `main()`.

## Utilisation depuis un autre script

```python
from data_augmentation.detection_augmentation.copy_and_paste_augmentation import (
    generate_augmented_dataset,
)

dataset_dir = generate_augmented_dataset(
    cutout_dir="datasets/LPL_dataset_cutouts_single",
    background_dir="datasets/Images_fond_blanc_&_bruits",
    output_dir="datasets/LPL_datasets_augment",
    number_of_images=1000,
    min_macroinvertebrates=4,
    max_macroinvertebrates=10,
    allow_overlap=True,
    iou_max=0.5,
    create_splits=True,
    split_ratios=[70, 10, 20],
    multi_class=False,
    create_validation=True,
    output_image_format="png",
)
```

## Structure du dataset produit

Avec `create_splits=True` :

```text
dataset_YYYYMMDD_HHMMSS/
├── train/
│   ├── images/
│   ├── labels/
│   └── labels_multi/
├── val/
│   ├── images/
│   ├── labels/
│   └── labels_multi/
├── test/
│   ├── images/
│   ├── labels/
│   └── labels_multi/
├── validation/
├── classes.json
├── classes_multi.json
├── dataset.yaml
└── dataset_1class.yaml ou dataset_multi.yaml
```

Avec `create_splits=False`, les dossiers `images`, `labels` et `labels_multi` sont
placés directement à la racine du dataset.

Quand `create_validation=True`, le dossier `validation` est créé séparément du dataset
YOLO. Il contient au maximum dix copies réduites d'images avec les boîtes dessinées.
Ces fichiers servent seulement au contrôle visuel et ne possèdent volontairement pas
de labels associés. Avec `create_validation=False`, ce dossier n'est pas créé.

## Classes et labels

YOLO cherche automatiquement un dossier standard nommé `labels`. Ce dossier contient
donc toujours les annotations actives :

- `multi_class=False` : toutes les annotations utilisent la classe `0`, nommée
  `macroinvertebrates` ;
- `multi_class=True` : les identifiants correspondent aux noms des sous-dossiers du
  dossier de cutouts.

Le dossier `labels_multi` conserve toujours les identifiants des classes d'origine.
Cela permet de garder l'information taxonomique même lorsqu'on entraîne un modèle à
une seule classe.

`dataset.yaml` est le fichier à fournir directement à YOLO. Le second YAML porte un
nom explicite correspondant au mode actif.

## Répartition train, val et test

`split_ratios` reçoit trois valeurs dans l'ordre suivant :

```python
[train, val, test]
```

Avec `number_of_images=1000` et `[70, 10, 20]`, le résultat contient :

- 700 images de train ;
- 100 images de validation ;
- 200 images de test.

Le nombre total est toujours exactement égal à `number_of_images`, même lorsque les
pourcentages donnent des valeurs non entières.

## Paramètres de génération

| Paramètre | Description | Valeur par défaut |
|---|---|---:|
| `cutout_dir` | Dossier contenant les PNG transparents, rangés par classe | obligatoire |
| `background_dir` | Dossier contenant les fonds PNG ou TIFF | obligatoire |
| `output_dir` | Dossier parent des datasets générés | obligatoire |
| `number_of_images` | Nombre total d'images à générer | `1000` |
| `min_macroinvertebrates` | Nombre minimum d'insectes par image | `4` |
| `max_macroinvertebrates` | Nombre maximum d'insectes par image | `10` |
| `padding` | Marge interdite autour de l'image, en proportion | `0.08` |
| `max_rotation` | Rotation aléatoire maximale en degrés | `180` |
| `allow_overlap` | Autorise les superpositions partielles | `True` |
| `iou_max` | IoU maximale autorisée entre deux boîtes | `1.0` |
| `create_splits` | Crée les splits train, val et test | `True` |
| `split_ratios` | Répartition `[train, val, test]` | `[70, 10, 20]` |
| `multi_class` | Utilise les classes des sous-dossiers dans `labels` | `False` |
| `create_validation` | Crée les aperçus permettant de contrôler les boîtes | `True` |
| `validation_count` | Nombre demandé d'aperçus, plafonné à 10 | `10` |
| `validation_max_size` | Dimension maximale des aperçus | `1600` |
| `png_compress_level` | Compression PNG, sans perte dans tous les cas | `0` |
| `output_image_format` | Format sans perte `png` ou `tiff` | `png` |
| `random_seed` | Graine permettant de reproduire un tirage | `None` |

Quand `allow_overlap=False`, aucune intersection entre deux bounding boxes n'est
acceptée. Quand il vaut `True`, `iou_max` fixe la limite. Par exemple, `iou_max=0.5`
autorise une superposition modérée. Une boîte entièrement contenue dans une autre est
toujours refusée, car l'insecte intérieur risquerait de ne plus être détectable.

## Paramètres SAM3 importants

| Paramètre | Description |
|---|---|
| `source_dir` | Dossier contenant les images globales |
| `output_dir` | Racine du futur dossier de cutouts |
| `prompt` | Concept recherché, par exemple `insect` |
| `threshold` | Score minimal de la détection globale |
| `detection_max_size` | Taille maximale de l'image utilisée pour la détection globale |
| `context_ratio` | Quantité de contexte ajoutée autour de chaque boîte |
| `refine_threshold` | Score minimal lors du second passage SAM3 |
| `refine_max_size` | Résolution de travail du raffinement SAM3 |
| `full_resolution` | Active GrabCut sur les pixels en résolution native |

Exemple :

```python
create_cutouts_sam3(
    source_dir="datasets/DATA/LPL_Photos_31072026",
    output_dir="datasets/LPL_dataset_cutouts_single",
    prompt="insect",
    threshold=0.7,
    context_ratio=3,
    full_resolution=True,
)
```

## Formats et qualité

Les fonds JPEG sont ignorés pour éviter de réutiliser une image déjà compressée. Les
images générées et les cutouts sont enregistrés en PNG sans perte. Le niveau de
compression PNG change uniquement la vitesse et la taille du fichier, jamais les
pixels ni la résolution.

La rotation bicubique rééchantillonne nécessairement l'insecte tourné, mais aucune
compression avec pertes n'est appliquée ensuite.

## Vitesse de génération

Les fonds actuels mesurent `6336 x 6336`, soit plus de 40 millions de pixels chacun.
L'écriture de ces grandes images représente la majorité du temps de génération.

Pour accélérer sans perdre de qualité :

- utiliser `create_validation=False` si les aperçus ne sont pas nécessaires ;
- garder `png_compress_level=0` pour l'écriture PNG la plus rapide ;
- utiliser `output_image_format="tiff"` pour une écriture brute nettement plus rapide.

Le TIFF brut et le PNG sont tous les deux sans perte. Le TIFF est plus rapide à écrire
mais produit de gros fichiers. La version d'Ultralytics utilisée dans le projet accepte
les extensions `.tif` et `.tiff`.
