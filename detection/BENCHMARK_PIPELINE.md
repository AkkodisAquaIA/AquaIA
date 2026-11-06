# Pipeline automatisé des expériences de détection

Le fichier `detection/benchmark_pipeline.yaml` décrit les datasets, les paramètres communs, les profils d'augmentation et les expériences M0 à M12.

## Vérifier sans entraîner

```bash
python main.py benchmark --config detection/benchmark_pipeline.yaml --dry-run
```

Cette commande valide le grand YAML et génère les configurations individuelles sans lancer de calcul GPU.

## Lancer toute la suite

```bash
python main.py benchmark --config detection/benchmark_pipeline.yaml
```

Les expériences sont exécutées dans l'ordre. Les dépendances sont résolues automatiquement : M4 utilise par exemple le `best.pt` produit par M3.

## Ne lancer qu'un sous-ensemble

Pour l'étude de la résolution de la baseline :

```bash
python main.py benchmark --config detection/benchmark_pipeline.yaml --only M1 M9 M10
```

M0 est inclus automatiquement, car ces modèles en dépendent.

Une expérience terminée est réutilisée si sa configuration n'a pas changé. L'option `--force` impose un nouveau calcul.

## Réutiliser un entraînement existant

Une expérience peut conserver toute sa description et sa place dans le graphe tout en utilisant un checkpoint déjà entraîné :

```yaml
reuse:
  checkpoint: "results/detect/yolo11_n_pretrained/<run>/weights/best.pt"
  run_dir: "results/detect/yolo11_n_pretrained/<run>"
```

Le pipeline ne relance alors pas l'entraînement. Il lit les métriques de validation dans le `results.csv` du run existant, réalise l'inférence configurée et transmet ce checkpoint aux expériences qui en dépendent. Supprimer la section `reuse` réactive l'entraînement normal.

## Résultats

Le dossier configuré dans `benchmark.output_dir` contient :

- `benchmark_results.csv` : tableau exploitable dans Excel ;
- `benchmark_results.md` : tableau lisible directement ;
- `benchmark_results.json` : données structurées ;
- `benchmark_ranking.csv`, `.md` et `.json` : classement décroissant par mAP50-95 de test ;
- `benchmark_state.yaml` : état permettant la reprise ;
- `generated_configs/` : YAML exacts utilisés pour chaque entraînement et inférence ;
- `training/` et `evaluation/` : runs et prédictions.

Pour utiliser de nouvelles données, modifier les chemins sous `datasets`. Le changement est détecté par l'empreinte de configuration et les expériences concernées sont recalculées. Les fichiers `stats_<résolution>.npy` manquants sont générés automatiquement à partir des dossiers `images/train`, `images/val` et `images/test`.

## Early stopping

Le protocole utilise 500 époques au maximum et une patience de 100. La patience élevée est adaptée à la petite validation LPL, dont les métriques peuvent fluctuer fortement. Tous les modèles comparés utilisent la même règle et le tableau conserve la meilleure époque, le nombre réel d'époques et le temps effectivement consommé.

La mAP est calculée à partir des prédictions conservées dès une confiance de 0,001. La précision et le rappel utilisent séparément le seuil `pr_conf_thresh`, fixé à 0,05 et modifiable dans le grand YAML.

Le classement est limité par défaut au jeu de test LPL grâce à `benchmark.ranking_dataset`. Cela évite de classer ensemble des scores produits sur des jeux de test différents.
