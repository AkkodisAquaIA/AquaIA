# Fine-tuning YOLO sur LPL

Le pipeline accepte un poids préentraîné officiel ou un checkpoint local. Un
fine-tuning depuis un checkpoint se configure ainsi :

```yaml
model:
  family: yolo11
  size: n
  init: checkpoint
  checkpoint: results/path/to/weights/best.pt

data:
  dataset_yaml: datasets/LPL_dataset_final/dataset

training:
  task: detect
  device: cuda
  imgsz: 640
  batch: 16
  epochs: 300
  patience: 0

output:
  project: results/detect_lpl/my_experiment
  exist_ok: true
  plots: true
```

Lancement :

```bash
python main.py train --config path/to/train_config.yaml
```

Il ne faut pas utiliser `--resume` pour un nouveau fine-tuning : l'entraînement
repart à l'époque 1 avec les poids indiqués. `images/train` met à jour le modèle,
`images/val` sert à suivre les métriques et à sélectionner `weights/best.pt`, et
`images/test` reste réservé à l'évaluation finale.

Pour une comparaison valide, conserver les mêmes hyperparamètres et la même
graine entre les initialisations étudiées. Les campagnes multi-modèles doivent
de préférence être décrites avec `detection/benchmark_pipeline.yaml`.

TensorBoard :

```bash
tensorboard --logdir results/detect_lpl --port 6006
```
