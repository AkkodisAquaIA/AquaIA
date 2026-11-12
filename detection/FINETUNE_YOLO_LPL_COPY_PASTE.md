# Fine-tuning LPL avec Copy & Paste

Le mode mixte ajoute à chaque époque des compositions Copy & Paste générées en
mémoire aux vraies images du train. La validation reste le split LPL réel : elle
ne contient aucune composition synthétique et conserve son rôle dans la sélection
de `best.pt`.

## Configuration

Ajouter cette section à un YAML d'entraînement YOLO :

```yaml
copy_paste_data:
  enabled: true
  cutout_dir: datasets/LPL_dataset_cutouts_sam3_filtered_by_split/train
  background_dir: datasets/Images_fond_blanc_&_bruits
  synthetic_fraction: 0.5
  min_macroinvertebrates: 4
  max_macroinvertebrates: 10
  padding: 0.08
  max_rotation: 180
  cutout_scale_range: [0.8, 1.2]
  allow_overlap: true
  iou_max: 1.0
  multi_class: false
  use_noisy_backgrounds: true
  cache_backgrounds: true
  apply_yolo_augmentations_to_synthetic: true
```

Paramètres principaux :

| Paramètre | Rôle |
| --- | --- |
| `cutout_dir` | Cutouts utilisés pour créer les objets synthétiques. Pointer explicitement vers le sous-dossier `train`. |
| `background_dir` | Fonds propres et bruités disponibles. |
| `synthetic_fraction` | Part synthétique visée dans une époque. `0.5` crée autant d'images synthétiques que réelles. |
| `min/max_macroinvertebrates` | Nombre d'objets collés par composition. |
| `padding` | Marge conservée autour de l'image. |
| `max_rotation` | Rotation aléatoire maximale des cutouts. |
| `cutout_scale_range` | Intervalle de redimensionnement des objets. |
| `allow_overlap` et `iou_max` | Autorisation et limite de superposition des objets. |
| `use_noisy_backgrounds` | Autorise les fonds bruités en plus des fonds propres. |
| `apply_yolo_augmentations_to_synthetic` | Applique aussi les transformations YOLO natives aux compositions synthétiques. |

Avec `apply_yolo_augmentations_to_synthetic: false`, les augmentations YOLO
natives restent actives sur les vraies images, tandis que les compositions Copy
& Paste sont utilisées telles qu'elles ont été générées.

## Fonctionnement du mélange

Le trainer construit d'abord le dataset YOLO réel. Pour le train uniquement, il
ajoute un dataset synthétique dont la taille est calculée à partir de
`synthetic_fraction`. Les deux sources utilisent le même `collate_fn`. Les
compositions sont renouvelées à chaque époque grâce à la graine dépendante de
l'époque.

Le chemin des cutouts est explicite : le code ne déduit pas automatiquement leur
split. Pour éviter toute fuite de données, seuls les cutouts issus des images
`train` doivent être fournis. Les cutouts de validation et de test ne doivent
jamais être utilisés pour l'entraînement.

Lancement et suivi :

```bash
python main.py train --config path/to/copy_paste_config.yaml
tensorboard --logdir results/detect_lpl_copy_paste --port 6006
```

Le fichier `detection/benchmark_pipeline.yaml` fournit des exemples de profils
Copy & Paste seuls et Copy & Paste avec Mosaic sur les images réelles.
