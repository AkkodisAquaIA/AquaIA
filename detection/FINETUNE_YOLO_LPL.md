# Fine-tuning YOLO11n sur le dataset LPL

Ces quatre expériences utilisent le même dataset, les mêmes hyperparamètres et
la même graine. Seule l'initialisation du modèle change. Cette règle est
importante pour pouvoir comparer les résultats.

## Expériences disponibles

| Configuration | Initialisation |
| --- | --- |
| `finetune_configs/lpl_from_roboflow_300_aug.yaml` | `best.pt` du run Roboflow 300 epochs avec augmentations |
| `finetune_configs/lpl_from_roboflow_1000_aug.yaml` | `best.pt` du run Roboflow 1000 epochs avec augmentations |
| `finetune_configs/lpl_from_roboflow_no_aug.yaml` | `best.pt` du run Roboflow arrêté vers 425 epochs, sans augmentations YOLO |
| `finetune_configs/lpl_from_coco_pretrained.yaml` | poids officiels COCO `yolo11n.pt` |

Les trois premières configurations utilisent `model.init: checkpoint`. La
dernière utilise simplement `model.init: pretrained`.

## Avant de lancer

Depuis la racine du dépôt, vérifier l'environnement et les chemins :

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
test -f datasets/LPL_dataset_final/dataset/dataset.yaml
test -f results/detect/yolo11_n_pretrained/20261015_125247/weights/best.pt
test -f results/detect/yolo11_n_pretrained/20261015_153825/weights/best.pt
test -f results/detect/yolo11_n_pretrained/20261016_103935/weights/best.pt
```

Le fichier `dataset.yaml` fournit à Ultralytics :

- `images/train` pour mettre à jour les poids ;
- `images/val` pour calculer les métriques à chaque epoch et choisir le nouveau
  `best.pt` ;
- `images/test`, qui n'est pas utilisé pendant le fine-tuning.

## Paramètres à choisir

Les quatre fichiers sont réglés par défaut sur 300 epochs, sans early stopping :

```yaml
epochs: 300
patience: 0
```

Pour une première comparaison, modifier ces valeurs de la même manière dans les
quatre fichiers. `patience: 0` permet d'observer toutes les courbes ; le meilleur
checkpoint de validation est tout de même conservé sous `weights/best.pt`.

Les quatre expériences appliquent également les mêmes augmentations au nouveau
dataset LPL. Elles sont indépendantes des augmentations qui avaient été
utilisées, ou non, lors de l'ancien entraînement Roboflow.

## Lancer les entraînements

Lancer une expérience à la fois sur un même GPU :

```bash
python main.py train --config detection/finetune_configs/lpl_from_roboflow_300_aug.yaml
```

```bash
python main.py train --config detection/finetune_configs/lpl_from_roboflow_1000_aug.yaml
```

```bash
python main.py train --config detection/finetune_configs/lpl_from_roboflow_no_aug.yaml
```

```bash
python main.py train --config detection/finetune_configs/lpl_from_coco_pretrained.yaml
```

Il ne faut pas utiliser `--resume` ici. Chaque commande démarre un nouvel
entraînement LPL à l'epoch 1 avec les poids indiqués dans `model`.

## Suivre et comparer dans TensorBoard

Dans un autre terminal :

```bash
source .venv/bin/activate
tensorboard --logdir results/detect_lpl --port 6006
```

Ouvrir ensuite <http://localhost:6006> et comparer en particulier :

- `metrics/precision(B)` ;
- `metrics/recall(B)` ;
- `metrics/mAP50(B)` ;
- `metrics/mAP50-95(B)` ;
- les trois losses `train/*` et `val/*`.

Chaque lancement crée un sous-dossier horodaté dans le projet correspondant.
Le checkpoint final à comparer se trouve dans `weights/best.pt`; il maximise la
`mAP50-95` sur la validation LPL.

## Évaluation finale

Utiliser le split `val` et TensorBoard pour choisir les paramètres et le modèle.
Une fois le choix terminé, lancer l'inférence sur `test` avec le nouveau
`weights/best.pt`. Éviter de choisir les hyperparamètres à partir du test afin de
conserver une évaluation finale indépendante.
