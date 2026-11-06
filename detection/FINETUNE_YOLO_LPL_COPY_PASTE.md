# Fine-tuning YOLO sur LPL avec le copy-paste en mémoire

## Séparation des données

Le dossier historique `datasets/LPL_dataset_cutouts_single` mélangeait des
cutouts issus des images train, val et test. Il ne doit plus être utilisé
directement pour entraîner un modèle.

Les cutouts audités sont maintenant rangés dans :

```text
datasets/LPL_dataset_cutouts_sam3_filtered_by_split/
├── train/   # seules sources autorisées pour créer le train synthétique
├── val/     # isolé, non utilisé pendant l'entraînement
├── test/    # isolé, strictement interdit pendant l'entraînement
├── filter_report.json
├── manifest.csv
└── rejected_cutouts.csv
```

Pour reconstruire ce dossier à partir des splits LPL :

```bash
python data_processing/generate_lpl_sam3_cutouts.py --overwrite
python data_processing/filter_sam3_cutouts_with_lpl_annotations.py --fill-unmatched --overwrite
```

La première commande relance SAM3 sur chaque image des trois splits. La seconde
ne conserve que les découpes qui correspondent à une bounding box LPL et crée
aussi `datasets/LPL_dataset_cutouts_sam3_filtered_by_taxon`, organisé par nom
taxonomique complet puis par split. L'option `--fill-unmatched` utilise chaque
bounding box encore sans découpe comme prompt SAM3, ce qui permet notamment de
traiter les mollusques que le prompt texte initial `insect` ne détecte pas.

## Crops de classification par taxon complet

Pour recréer les crops rectangulaires directement depuis les annotations LPL :

```bash
python data_processing/crop_detection_to_classification.py \
  --images-dir datasets/LPL_dataset_final/dataset_LPL_no_split/TIF_cropped \
  --labels-dir datasets/LPL_dataset_final/dataset_LPL_no_split/Annotations/labels/train \
  --classes-yaml datasets/LPL_dataset_final/dataset_LPL_no_split/Annotations/data.yaml \
  --output-dir datasets/LPL_dataset_final/dataset_LPL_no_split_crop_full_taxonomy
```

La conversion développe chaque préfixe taxonomique connu indépendamment. Un
ordre déjà écrit en toutes lettres, comme `Plectoptera`, ne bloque donc plus la
conversion de `Arthr_Insec` en `Arthropoda_Insecta`.

## Composition d'une époque

Avec `synthetic_fraction: 0.5`, une époque contient :

- les 22 vraies images de `images/train` ;
- 22 compositions copy-paste générées en mémoire à partir des cutouts du train ;
- les 9 vraies images de `images/val` pour la validation uniquement.

Les compositions changent à chaque époque grâce à la seed dépendante de
l'époque. La validation ne contient aucune image synthétique et reste fixe.

Les augmentations YOLO configurables (mosaic, HSV, rotations, flips, etc.) sont
réglées à zéro afin d'isoler l'effet du copy-paste personnalisé. Ultralytics peut
encore appliquer ses opérations Albumentations internes à très faible probabilité
(Blur, Gray ou CLAHE). Pour augmenter ou réduire la part synthétique, modifier la
même valeur dans les quatre fichiers :

```yaml
copy_paste_data:
  synthetic_fraction: 0.5
```

Par exemple, `0.33` donne environ un tiers d'images synthétiques et `0.67`
environ deux tiers.

## Lancer les quatre entraînements

Depuis la racine du dépôt :

```bash
source .venv/bin/activate
```

```bash
python main.py train --config detection/finetune_copy_paste_configs/lpl_from_roboflow_300_aug.yaml
```

```bash
python main.py train --config detection/finetune_copy_paste_configs/lpl_from_roboflow_1000_aug.yaml
```

```bash
python main.py train --config detection/finetune_copy_paste_configs/lpl_from_roboflow_no_aug.yaml
```

```bash
python main.py train --config detection/finetune_copy_paste_configs/lpl_from_coco_pretrained.yaml
```

Ne pas utiliser `--resume` : chaque commande initialise un nouveau fine-tuning
LPL depuis les poids configurés.

## TensorBoard

```bash
tensorboard --logdir results/detect_lpl_copy_paste --port 6006
```

Ouvrir ensuite <http://localhost:6006>. Le nouveau `weights/best.pt` de chaque
expérience est choisi avec la mAP50-95 calculée exclusivement sur les vraies
images LPL de validation.
