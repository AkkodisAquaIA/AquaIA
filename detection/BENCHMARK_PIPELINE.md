# Benchmark automatisé de détection

Le benchmark décrit une campagne complète dans un seul YAML. Il génère les
configurations unitaires, lance les entraînements et les inférences, puis produit
un tableau comparatif et un classement par mAP50-95 de test.

Fichiers principaux :

- `detection/benchmark_pipeline.yaml` : exemple de campagne LPL ;
- `detection/benchmark.py` : orchestrateur ;
- `detection/FINETUNE_YOLO_LPL_COPY_PASTE.md` : configuration du Copy & Paste.

## Commandes

Depuis la racine du dépôt :

```bash
# Vérifier le YAML et générer les configurations sans calcul
python main.py benchmark --config detection/benchmark_pipeline.yaml --dry-run

# Exécuter toute la campagne
python main.py benchmark --config detection/benchmark_pipeline.yaml

# Exécuter une sélection et ses dépendances
python main.py benchmark --config detection/benchmark_pipeline.yaml --only M1 M10

# Rejouer les expériences, même si leur résultat est réutilisable
python main.py benchmark --config detection/benchmark_pipeline.yaml --only M10 --force
```

Sans `--force`, une expérience terminée est réutilisée si sa configuration n'a
pas changé et si ses artefacts existent encore. Si l'entraînement est terminé
mais que l'inférence a échoué, seule l'inférence est relancée.

## Déroulement

Pour chaque expérience active, le pipeline :

1. résout le checkpoint initial et les dépendances ;
2. fusionne les paramètres communs, le profil d'augmentation et les surcharges ;
3. écrit les YAML générés dans `generated_configs/` ;
4. lance `python main.py train` si un entraînement est demandé ;
5. lance `python main.py infer` sur le checkpoint `best.pt` ;
6. met à jour l'état, les tableaux et le classement.

L'état est sauvegardé après chaque étape. Les expériences doivent être déclarées
avant celles qui dépendent d'elles.

## Structure du YAML

```yaml
benchmark: {}
model: {}
datasets: {}
training_defaults: {}
inference_defaults: {}
augmentation_profiles: {}
experiments: []
```

### Campagne et modèle

```yaml
benchmark:
  name: "lpl_benchmark"
  output_dir: "results/benchmarks/lpl_benchmark"
  continue_on_error: false
  auto_compute_stats: true
  ranking_dataset: "lpl"

model:
  family: "yolo11"
  size: "n"
  pretrained_checkpoint: "yolo11n.pt"
```

- `output_dir` isole les résultats d'une campagne.
- `continue_on_error` autorise les expériences indépendantes à continuer après
  une erreur.
- `auto_compute_stats` crée `stats_<imgsz>.npy` avant une inférence si nécessaire.
- `ranking_dataset` évite de classer ensemble des modèles évalués sur des jeux
  différents. La valeur `null` désactive ce filtre.
- `pretrained_checkpoint` est le poids utilisé par `init.type: pretrained`.

### Datasets

```yaml
datasets:
  lpl:
    root: "datasets/LPL_dataset_final/dataset"
    img_format: "TIF"
```

La racine doit contenir le YAML attendu par le pipeline d'entraînement ainsi que
les dossiers `images/{train,val,test}` et `labels/{train,val,test}`. `img_format`
est utilisé par le chargeur d'inférence.

### Paramètres communs

```yaml
training_defaults:
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
  seed: 42
  deterministic: true
  compile: true

inference_defaults:
  imgsz: 640
  batch: 2
  workers: 3
  conf_thresh: 0.001
  pr_conf_thresh: 0.05
  conf: 0.10
  iou: 0.5
  num_samples: 16
  seed: 0
```

Les clés de `training_defaults` sont transmises à Ultralytics. Les paramètres
peuvent être remplacés dans un profil ou dans une expérience.

En inférence :

- `conf_thresh` est le seuil minimal des prédictions utilisées pour la mAP ;
- `pr_conf_thresh` est le seuil fixe utilisé pour précision, rappel et F1 ;
- `conf` est le seuil des images de prévisualisation ;
- `iou` règle la NMS ;
- `num_samples` règle le nombre de prévisualisations.

### Profils d'augmentation

Un profil regroupe les paramètres réutilisés par plusieurs expériences :

```yaml
augmentation_profiles:
  none:
    training:
      mosaic: 0
      fliplr: 0
      hsv_h: 0

  copy_paste:
    training:
      mosaic: 0
    copy_paste_data:
      enabled: true
      cutout_dir: "datasets/LPL_dataset_cutouts_sam3_filtered_by_split/train"
      background_dir: "datasets/Images_fond_blanc_&_bruits"
      synthetic_fraction: 0.5
```

`training` est fusionné avec `training_defaults`. `copy_paste_data` active le
dataset mixte réel et synthétique. Le fichier fourni utilise des ancres YAML pour
éviter de recopier les paramètres communs.

### Expériences

Exemple d'entraînement depuis le poids COCO :

```yaml
- id: M1
  initialization_label: "M0"
  training_label: "LPL sans augmentation"
  imgsz: 640
  init: {type: experiment, experiment: M0}
  train:
    dataset: lpl
    augmentation: none
    overrides: {imgsz: 640}
  evaluate: {dataset: lpl, split: test}
```

Les trois initialisations possibles sont :

```yaml
init: {type: pretrained}
init: {type: checkpoint, checkpoint: "path/to/model.pt"}
init: {type: experiment, experiment: M1}
```

- `pretrained` utilise `model.pretrained_checkpoint` ;
- `checkpoint` utilise un fichier explicite ;
- `experiment` utilise le checkpoint produit ou réutilisé par une expérience
  précédente.

`train` est facultatif. Sans cette section, le modèle est seulement évalué.
La clé `imgsz` de l'expérience est appliquée à l'entraînement et à l'inférence.
`train.overrides` remplace les paramètres en dernier et a donc priorité sur le
profil et les valeurs communes. `evaluate.overrides` joue le même rôle pour
l'inférence. `enabled: false` ignore une expérience.

### Réutiliser un entraînement existant

```yaml
- id: M2
  initialization_label: "M0"
  training_label: "Aquatic Insects, réutilisé"
  imgsz: 640
  init: {type: pretrained}
  reuse:
    checkpoint: "results/detect/run/weights/best.pt"
    run_dir: "results/detect/run"
  train:
    dataset: aquatic_insects
    augmentation: none
  evaluate: {dataset: aquatic_insects, split: test}
```

Avec `reuse`, le pipeline ne réentraîne pas le modèle. Il vérifie le checkpoint
et `run_dir/results.csv`, réalise l'évaluation, puis rend ce checkpoint disponible
aux expériences dépendantes. Les chemins de réutilisation du YAML d'exemple sont
propres à cette campagne et doivent être adaptés sur une autre machine.

## Résultats

Une campagne produit :

```text
results/benchmarks/<campagne>/
├── benchmark_state.yaml
├── generated_configs/
├── training/<id>/<timestamp>/
├── evaluation/<id>/
├── references/<id>/
├── benchmark_results.{csv,json,md}
└── benchmark_ranking.{csv,json,md}
```

Le tableau contient l'initialisation, le protocole, la résolution, les mAP de
validation et de test, la précision, le rappel, la meilleure époque et la durée.
La meilleure époque correspond au maximum de mAP50-95 de validation. Le
classement est trié par mAP50-95 de test sur `ranking_dataset`.

Les TensorBoard de la campagne se trouvent sous `training/` :

```bash
tensorboard --logdir results/benchmarks/<campagne>/training --port 6006
```

## Bonnes pratiques

- utiliser un nouvel `output_dir` pour une nouvelle campagne ;
- comparer sur un même split de test et avec les mêmes seuils ;
- ne modifier qu'un facteur à la fois pour une étude d'ablation ;
- lancer `--dry-run` avant un calcul long ;
- conserver le split test uniquement pour l'évaluation finale ;
- vérifier que les cutouts Copy & Paste proviennent exclusivement du train.
