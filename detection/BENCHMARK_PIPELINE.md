# Pipeline automatisé des expériences de détection

Le pipeline permet de décrire dans un seul fichier YAML une suite complète d'expériences : initialisation des modèles, entraînements, inférences, collecte des métriques et classement final.

Les deux fichiers principaux sont :

- `detection/benchmark_pipeline.yaml` : configuration de la campagne ;
- `detection/benchmark.py` : orchestration des expériences.

Le pipeline appelle les commandes d'entraînement et d'inférence déjà utilisées dans le projet. Il génère les YAML individuels, puis lance successivement `python main.py train` et `python main.py infer`.

## 1. Commandes principales

Toutes les commandes sont à lancer depuis la racine du dépôt, avec l'environnement virtuel activé.

### Vérifier la configuration sans entraîner

```bash
python main.py benchmark \
  --config detection/benchmark_pipeline.yaml \
  --dry-run
```

Cette commande charge et valide le grand YAML, résout les dépendances et génère les YAML individuels sans lancer de calcul. Les checkpoints indiqués dans une section `reuse` doivent toutefois exister.

### Lancer toute la campagne

```bash
python main.py benchmark --config detection/benchmark_pipeline.yaml
```

Les expériences actives sont exécutées dans l'ordre où elles apparaissent dans le YAML.

### Lancer seulement certaines expériences

```bash
python main.py benchmark \
  --config detection/benchmark_pipeline.yaml \
  --only M1 M12 M13
```

Les dépendances sont ajoutées automatiquement. Les parents doivent néanmoins apparaître avant leurs enfants dans la liste `experiments`, car le pipeline conserve l'ordre du YAML.

### Rejouer une expérience terminée

```bash
python main.py benchmark \
  --config detection/benchmark_pipeline.yaml \
  --only M6 \
  --force
```

Sans `--force`, une expérience terminée et inchangée est réutilisée. Avec `--force`, un nouvel entraînement et une nouvelle inférence sont lancés.

## 2. Organisation générale du YAML

Le fichier comporte sept sections obligatoires :

```yaml
benchmark: {}
model: {}
datasets: {}
training_defaults: {}
inference_defaults: {}
augmentation_profiles: {}
experiments: []
```

La section `copy_paste_defaults` du fichier actuel est un bloc réutilisable appelé par les profils Copy & Paste. Elle n'est pas imposée par le validateur.

### Campagne actuellement configurée

| Modèle | Initialisation | Entraînement ou rôle | Résolution | Test |
|---|---|---|---:|---|
| M0 | COCO | Aucun, modèle de fondation | 640 | LPL |
| M1 | M0 | LPL sans augmentation, baseline | 640 | LPL |
| M2 | M0 | Aquatic Insects sans augmentation, checkpoint réutilisé | 640 | Aquatic Insects |
| M3 | M0 | Aquatic Insects avec augmentation YOLO, checkpoint réutilisé | 640 | Aquatic Insects |
| M4 | M2 | LPL sans augmentation | 640 | LPL |
| M5 | M3 | LPL sans augmentation | 640 | LPL |
| M6 | M0 | LPL avec augmentations YOLO fortes | 640 | LPL |
| M7 | M0 | LPL avec couleurs légères et flips, sans Mosaic | 640 | LPL |
| M8 | M0 | LPL avec Mosaic uniquement | 640 | LPL |
| M9 | M0 | LPL avec couleurs légères, flips et Mosaic | 640 | LPL |
| M10 | M0 | LPL avec Copy & Paste | 640 | LPL |
| M11 | M0 | LPL avec Copy & Paste et Mosaic sur les images réelles | 640 | LPL |
| M12 | M0 | LPL sans augmentation | 1 024 | LPL |
| M13 | M0 | LPL sans augmentation | 2 048 | LPL |

M2 et M3 ne sont pas réentraînés. Leurs checkpoints existants sont évalués puis transmis respectivement à M4 et M5. Toutes les autres phases d'entraînement de cette campagne utilisent uniquement le dataset LPL.

## 3. Section `benchmark`

```yaml
benchmark:
  name: "detection_standardized_benchmark"
  output_dir: "results/benchmarks/detection_standardized_benchmark"
  continue_on_error: false
  auto_compute_stats: true
  ranking_dataset: "lpl"
```

| Paramètre | Rôle |
|---|---|
| `name` | Nom descriptif de la campagne. Il n'est actuellement pas utilisé pour construire les chemins. |
| `output_dir` | Dossier contenant tous les entraînements, inférences, états et tableaux. Utiliser un nouveau dossier pour une campagne indépendante. |
| `continue_on_error` | Si `false`, arrêt à la première erreur. Si `true`, le pipeline tente les expériences indépendantes suivantes. Une expérience dépendante d'un modèle en échec ne pourra pas s'exécuter. |
| `auto_compute_stats` | Génère automatiquement le fichier `stats_<imgsz>.npy` manquant avant l'inférence. |
| `ranking_dataset` | Limite le classement final aux expériences évaluées sur ce dataset. Mettre `null` pour inclure tous les jeux de test, ce qui est généralement déconseillé. |

Changer `output_dir` est la méthode la plus simple pour démarrer une campagne propre sans réutiliser l'état précédent.

## 4. Section `model`

```yaml
model:
  family: "yolo11"
  size: "n"
  pretrained_checkpoint: "yolo11n.pt"
```

| Paramètre | Rôle |
|---|---|
| `family` | Famille YOLO reportée dans les configurations générées, par exemple `yolo11`. |
| `size` | Variante du modèle : `n`, `s`, `m`, `l` ou `x`. |
| `pretrained_checkpoint` | Checkpoint utilisé par `init.type: pretrained`. Il peut s'agir d'un poids de fondation ou d'un fichier local. |

Le fichier doit exister au moment de l'exécution. Le pipeline n'effectue pas lui-même son téléchargement.

## 5. Section `datasets`

Chaque dataset reçoit un identifiant libre, ensuite utilisé dans `train.dataset`, `evaluate.dataset` et `benchmark.ranking_dataset`.

```yaml
datasets:
  lpl:
    root: "datasets/LPL_dataset_final/dataset"
    img_format: "TIF"
  aquatic_insects:
    root: "datasets/aquatic_insects_v7_1classe"
    img_format: "jpg"
```

| Paramètre | Rôle |
|---|---|
| `root` | Racine du dataset. L'entraînement attend aussi `<root>/<nom_du_dossier>.yaml`. |
| `img_format` | Extension utilisée par le dataset d'inférence, sans point initial. La casse doit correspondre aux fichiers. |

Structure attendue :

```text
dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Le pipeline ne crée pas les splits et ne vérifie pas leur étanchéité.

## 6. Section `training_defaults`

Cette section définit le protocole commun à tous les entraînements.

```yaml
training_defaults:
  task: "detect"
  device: "cuda"
  imgsz: 640
  batch: 2
  workers: 3
  epochs: 500
  patience: 25
  optimizer: "AdamW"
  lr0: 0.0001
  lrf: 0.05
  cos_lr: true
  weight_decay: 0.00005
  cls: 4.0
  box: 5.0
  dfl: 1.5
  conf: 0.001
  iou: 0.5
  val: true
  seed: 42
  deterministic: true
  compile: true
```

### Exécution

| Paramètre | Rôle |
|---|---|
| `task` | Tâche Ultralytics, ici `detect`. |
| `device` | `auto` sélectionne CUDA si disponible, sinon le CPU. |
| `imgsz` | Taille carrée donnée au modèle. Elle peut être remplacée par expérience. |
| `batch` | Nombre d'images par lot. Un batch identique facilite les comparaisons, mais doit tenir dans la mémoire GPU. |
| `workers` | Nombre de processus de chargement des données. |
| `epochs` | Nombre maximal d'époques. |
| `patience` | Nombre d'époques sans amélioration avant early stopping. `0` le désactive. |
| `compile` | Active `torch.compile`. Ce paramètre peut modifier le temps de calcul et doit rester identique dans un benchmark de vitesse. |

### Optimisation

| Paramètre | Rôle |
|---|---|
| `optimizer` | Optimiseur utilisé. |
| `lr0` | Taux d'apprentissage initial. |
| `lrf` | Facteur final. Avec `lr0: 0.0001` et `lrf: 0.05`, le taux cible final est proche de `0.000005`. |
| `cos_lr` | Active une décroissance cosinus du taux d'apprentissage. |
| `weight_decay` | Régularisation appliquée aux poids. |
| `cls`, `box`, `dfl` | Pondérations des composantes de la loss YOLO. |

### Validation et reproductibilité

| Paramètre | Rôle |
|---|---|
| `conf` | Seuil minimal utilisé par la validation Ultralytics pour construire ses courbes. `0.001` conserve presque toutes les prédictions. |
| `iou` | Seuil IoU utilisé notamment pour la suppression non maximale. |
| `val` | Active la validation à chaque époque. |
| `seed` | Graine pseudo-aléatoire. |
| `deterministic` | Demande un comportement aussi déterministe que possible. Des différences peuvent subsister selon le matériel et CUDA. |

Les paramètres d'augmentation YOLO sont apportés par le profil sélectionné dans chaque expérience.

## 7. Section `inference_defaults`

```yaml
inference_defaults:
  imgsz: 640
  batch: 2
  workers: 3
  conf_thresh: 0.001
  pr_conf_thresh: 0.05
  conf: 0.10
  iou: 0.5
  num_samples: 20
  seed: 0
```

| Paramètre | Rôle |
|---|---|
| `imgsz` | Résolution d'évaluation. Elle doit normalement être identique à celle de l'entraînement. |
| `batch` | Taille des lots d'inférence. Elle agit sur la mémoire et la vitesse, pas sur les métriques. |
| `workers` | Nombre de processus de chargement. |
| `conf_thresh` | Seuil minimal avant le calcul de la mAP. `0.001` conserve presque toute la courbe précision-rappel. |
| `pr_conf_thresh` | Seuil fixe utilisé pour la précision, le rappel et le F1 du tableau. |
| `conf` | Seuil réservé aux images de prédiction enregistrées. Il n'agit pas sur les métriques. |
| `iou` | Seuil de NMS utilisé pendant la prédiction. |
| `num_samples` | Nombre d'images de prédiction sauvegardées. |
| `seed` | Graine de sélection des exemples visuels. |

La précision et le rappel dépendent de `pr_conf_thresh`. La mAP résume une courbe calculée sur plusieurs seuils de confiance, après élimination des prédictions inférieures à `conf_thresh`.

## 8. Configuration du Copy & Paste

Le bloc actuel utilise une ancre YAML pour éviter les répétitions :

```yaml
copy_paste_defaults: &copy_paste_defaults
  enabled: true
  cutout_dir: "datasets/LPL_dataset_cutouts_sam3_filtered_by_split/train"
  background_dir: "datasets/Images_fond_blanc_&_bruits"
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
```

| Paramètre | Rôle |
|---|---|
| `enabled` | Active le trainer personnalisé Copy & Paste. |
| `cutout_dir` | Organismes détourés. Pour éviter une fuite, utiliser uniquement des objets issus du train. |
| `background_dir` | Fonds propres et bruités. L'implémentation actuelle attend au moins huit images : quatre propres puis quatre bruitées. |
| `synthetic_fraction` | Part d'images synthétiques. `0.5` produit autant d'images synthétiques que de vraies images. |
| `min_macroinvertebrates`, `max_macroinvertebrates` | Nombre d'objets synthétiques par composition. |
| `padding` | Marge autour des bords. Valeur entre 0 et 0,5. |
| `max_rotation` | Amplitude maximale de rotation, en degrés. |
| `cutout_scale_range` | Intervalle multiplicatif de redimensionnement des cutouts. |
| `allow_overlap` | Autorise ou interdit la superposition des objets. |
| `iou_max` | IoU maximale autorisée lorsque les superpositions sont contrôlées. |
| `multi_class` | `false` réduit toutes les espèces à une classe. |
| `use_noisy_backgrounds` | `true` utilise fonds propres et bruités ; `false` conserve les quatre premiers fonds propres. |
| `cache_backgrounds` | Met les fonds redimensionnés en RAM pour accélérer la génération. |

La syntaxe suivante recopie le bloc dans un profil :

```yaml
copy_paste_data: *copy_paste_defaults
```

`&copy_paste_defaults` définit l'ancre et `*copy_paste_defaults` la réutilise. Il s'agit d'une fonctionnalité YAML, pas d'une instruction Python.

## 9. Profils d'augmentation

Chaque profil possède un nom libre utilisé par `train.augmentation`.

```yaml
augmentation_profiles:
  none:
    training:
      hsv_h: 0
      degrees: 0
      mosaic: 0

  copy_paste:
    training:
      hsv_h: 0
      degrees: 0
      mosaic: 0
    copy_paste_data: *copy_paste_defaults
```

Un profil peut contenir :

- `training` : paramètres transmis à Ultralytics ;
- `copy_paste_data` : paramètres du dataset synthétique personnalisé.

Le profil `copy_paste` désactive les augmentations YOLO pour isoler le Copy & Paste. Le profil `copy_paste_mosaic_real_only` applique Mosaic uniquement aux vraies images LPL ; les compositions synthétiques ne reçoivent pas d'augmentation YOLO supplémentaire.

Exemple de profil personnalisé :

```yaml
augmentation_profiles:
  yolo_custom:
    training:
      hsv_h: 0.005
      hsv_s: 0.10
      hsv_v: 0.10
      degrees: 10
      translate: 0.03
      scale: 0.10
      flipud: 0.5
      fliplr: 0.5
      mosaic: 0
      mixup: 0
      cutmix: 0
      copy_paste: 0
      close_mosaic: 0
```

## 10. Définition d'une expérience

Exemple complet :

```yaml
- id: M12
  enabled: true
  initialization_label: "M0"
  training_label: "LPL sans augmentation"
  imgsz: 1024
  init:
    type: experiment
    experiment: M0
  train:
    dataset: lpl
    augmentation: none
    overrides:
      imgsz: 1024
      batch: 2
  evaluate:
    dataset: lpl
    split: test
    overrides:
      imgsz: 1024
      batch: 2
```

| Champ | Rôle |
|---|---|
| `id` | Identifiant unique utilisé pour les dépendances, dossiers, états et tableaux. |
| `enabled` | Facultatif, `true` par défaut. Une expérience à `false` est ignorée. |
| `initialization_label` | Texte du rapport. Il ne choisit pas le checkpoint. |
| `training_label` | Texte de la colonne Entraînement. |
| `imgsz` | Résolution du rapport et valeur de repli de l'inférence. Pour l'entraînement, définir aussi `train.overrides.imgsz`. |
| `init` | Définit réellement le checkpoint de départ. |
| `train` | Décrit l'entraînement. Son absence crée une référence sans entraînement. |
| `reuse` | Réutilise un entraînement existant au lieu de lancer `train`. |
| `evaluate` | Décrit le dataset et le split d'évaluation. |

### Ordre de fusion

Pour l'entraînement :

1. `training_defaults` ;
2. `augmentation_profiles.<profil>.training` ;
3. `train.overrides`.

La dernière valeur gagne. `train.overrides.batch: 4` remplace donc le batch global uniquement pour cette expérience.

Pour l'inférence :

1. `inference_defaults` ;
2. `evaluate.overrides`.

Les dictionnaires sont fusionnés récursivement. Une liste ou une valeur simple est entièrement remplacée.

## 11. Types d'initialisation

### Modèle préentraîné global

```yaml
init:
  type: pretrained
```

Le checkpoint vient de `model.pretrained_checkpoint`. Une expérience sans `train`, comme M0, évalue directement ce checkpoint et peut le transmettre à ses dépendances.

### Checkpoint explicite

```yaml
init:
  type: checkpoint
  checkpoint: "results/mon_modele/weights/best.pt"
```

### Résultat d'une autre expérience

```yaml
init:
  type: experiment
  experiment: M3
```

Le pipeline récupère le checkpoint enregistré pour M3. Si M3 a été entraîné, il s'agit de son `best.pt`. Si M3 utilise `reuse`, il s'agit du checkpoint réutilisé.

Le champ `initialization_label` ne crée pas cette dépendance : seule la section `init` est fonctionnelle.

## 12. Réutiliser un entraînement existant

```yaml
reuse:
  checkpoint: "results/detect/yolo11_n_pretrained/<run>/weights/best.pt"
  run_dir: "results/detect/yolo11_n_pretrained/<run>"
```

Le pipeline vérifie le checkpoint et `run_dir/results.csv`. Il lit dans ce fichier les métriques de validation et le temps, puis exécute une nouvelle inférence selon `evaluate`. Le checkpoint peut ensuite servir de parent.

Si `run_dir` est omis, le pipeline utilise le dossier situé deux niveaux au-dessus du checkpoint, conformément à `<run>/weights/best.pt`.

Supprimer `reuse` réactive l'entraînement décrit dans `train`. `reuse` ne reprend pas un entraînement interrompu : le run est considéré comme terminé.

## 13. Blocs `train` et `evaluate`

```yaml
train:
  dataset: lpl
  augmentation: copy_paste
  overrides:
    imgsz: 1024
    batch: 2
    epochs: 500
    patience: 25
```

- `dataset` doit être une clé de `datasets` ;
- `augmentation` doit être une clé de `augmentation_profiles` ;
- `overrides` remplace les paramètres globaux pour cette expérience.

```yaml
evaluate:
  dataset: lpl
  split: test
  overrides:
    imgsz: 1024
    batch: 2
    single_class: true
```

- `dataset` choisit le jeu d'évaluation ;
- `split` vaut généralement `test` ;
- `overrides` remplace les paramètres d'inférence ;
- `single_class: true` réduit les classes prédites à une seule classe, notamment pour évaluer le modèle COCO brut sur le LPL mono-classe.

## 14. Ajouter des expériences

### Nouvelle résolution

```yaml
- id: M13
  initialization_label: "M0"
  training_label: "LPL sans augmentation"
  imgsz: 1536
  init: {type: experiment, experiment: M0}
  train:
    dataset: lpl
    augmentation: none
    overrides: {imgsz: 1536, batch: 2}
  evaluate:
    dataset: lpl
    split: test
    overrides: {imgsz: 1536, batch: 2}
```

### Fine-tuning en deux étapes

```yaml
- id: M14
  initialization_label: "M3"
  training_label: "LPL sans augmentation"
  imgsz: 640
  init: {type: experiment, experiment: M3}
  train:
    dataset: lpl
    augmentation: none
    overrides: {imgsz: 640}
  evaluate: {dataset: lpl, split: test}
```

### Désactiver une expérience

```yaml
- id: M12
  enabled: false
  # reste de la configuration
```

Une expérience désactivée ne peut pas servir de parent à une expérience active.

## 15. Fonctionnement interne du script

Lors de `python main.py benchmark`, le déroulement est le suivant :

1. `main.py` reconnaît la sous-commande et transmet les arguments à `handle_benchmark`.
2. `load_yaml` lit le fichier avec `yaml.safe_load` et crée des dictionnaires et listes Python.
3. `validate_suite` contrôle les sections, identifiants, datasets, profils et dépendances.
4. `dependency_ids` ajoute récursivement les parents demandés avec `--only`.
5. Une empreinte SHA-256 est calculée pour chaque expérience à partir du modèle, des datasets, des paramètres globaux, des profils et de l'expérience.
6. Si l'état précédent est réutilisable, ses chemins sont rechargés et aucun calcul n'est lancé.
7. `experiment_checkpoint` résout le modèle initial.
8. Avec `reuse`, le run existant est utilisé. Sinon, `build_train_config` fusionne les paramètres et écrit `generated_configs/<id>_train.yaml`.
9. L'entraînement est lancé avec `python main.py train --config ...`.
10. Le script retrouve le nouveau `weights/best.pt` et construit le YAML d'inférence.
11. Le fichier `stats_<imgsz>.npy` est calculé si nécessaire.
12. L'inférence est lancée avec `python main.py infer --config ...` et son `inference_metrics.yaml` est retrouvé.
13. L'état et tous les tableaux sont régénérés après chaque expérience.

L'entraînement et l'inférence sont exécutés dans des processus Python distincts. Cela évite d'accumuler les modèles en mémoire GPU entre les expériences.

## 16. Reprise et empreinte

L'état est conservé dans `benchmark_state.yaml`. Une expérience est réutilisée uniquement si :

- son statut est `complete` ;
- son empreinte correspond à la configuration actuelle ;
- son checkpoint existe ;
- son dossier d'inférence contient `inference_metrics.yaml` ;
- `--force` n'est pas utilisé.

L'empreinte inclut les sections globales complètes. Modifier ou ajouter un profil ou un dataset peut donc invalider toutes les expériences, même celles qui n'utilisent pas directement l'élément modifié.

Une expérience interrompue ne reprend pas au milieu d'un entraînement. Au prochain lancement, elle redémarre dans un nouveau run, sauf si un entraînement terminé est déclaré avec `reuse`.

## 17. Sélection du meilleur checkpoint

`best.pt` est produit par Ultralytics. Pour reconstruire le tableau, le pipeline recherche aussi la meilleure ligne de `results.csv` avec la fitness :

```text
fitness = 0,1 × mAP50 + 0,9 × mAP50-95
```

La meilleure époque, la mAP50 et la mAP50-95 de validation viennent de cette même ligne. Le temps affiché correspond au temps total à la fin de l'entraînement, pas au temps cumulé jusqu'à la meilleure époque.

## 18. Résultats produits

```text
results/benchmarks/detection_standardized_benchmark/
├── benchmark_state.yaml
├── benchmark_results.csv
├── benchmark_results.json
├── benchmark_results.md
├── benchmark_ranking.csv
├── benchmark_ranking.json
├── benchmark_ranking.md
├── generated_configs/
├── references/
├── training/
└── evaluation/
```

Le tableau principal contient l'initialisation, l'entraînement, la résolution, le dataset de test, les métriques de validation et de test, la meilleure époque, le nombre d'époques, le temps total et le statut.

Le classement trie les expériences terminées par mAP50-95 de test décroissante. `benchmark.ranking_dataset` évite de comparer directement des modèles évalués sur des datasets différents.

## 19. TensorBoard

```bash
tensorboard \
  --logdir results/benchmarks/detection_standardized_benchmark/training \
  --port 6006
```

Ouvrir ensuite <http://localhost:6006>. Adapter le chemin si `benchmark.output_dir` change.

Les expériences avec `reuse` conservent leurs logs dans leur ancien `run_dir` et n'apparaissent donc pas nécessairement sous le dossier de la nouvelle campagne.

## 20. Checklist avant une campagne

1. Vérifier les images, annotations et splits.
2. Vérifier que `dataset.yaml` référence les bons dossiers.
3. Déclarer les datasets et leurs extensions.
4. Choisir un nouvel `output_dir`.
5. Définir le protocole commun.
6. Vérifier l'absence de fuite entre train, validation et test.
7. Pour le Copy & Paste, utiliser exclusivement les cutouts du train.
8. Vérifier les profils d'augmentation.
9. Placer chaque parent avant ses dépendances.
10. Lancer `--dry-run` et inspecter `generated_configs`.
11. Lancer un essai court si le dataset ou le modèle vient de changer.
12. Lancer la campagne et suivre TensorBoard.

## 21. Points d'attention

- `imgsz` au niveau d'une expérience sert au rapport et à l'inférence par défaut. Définir aussi `train.overrides.imgsz` pour l'entraînement.
- `initialization_label` et `training_label` sont uniquement descriptifs.
- `reuse` évite l'entraînement, mais relance l'inférence actuelle.
- `--force` rejoue aussi les entraînements coûteux sans `reuse`.
- Une comparaison de temps exige les mêmes batch, workers, compilation, précision mixte et matériel.
- Une comparaison de scores exige le même dataset, le même split et les mêmes seuils.
- Le classement sur un petit test ne mesure pas à lui seul la généralisation.
