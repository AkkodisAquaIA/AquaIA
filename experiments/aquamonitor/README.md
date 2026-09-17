# AquaMonitor — espace d'expérimentation

Ce dossier isole l'exploration d'AquaMonitor du code applicatif. Le premier
objectif est de comprendre les métadonnées et les splits officiels avant tout
entraînement.

## Structure

```text
experiments/aquamonitor/
├── notebooks/
│   └── aquamonitor_eda.ipynb
├── data/                  # cache et données téléchargées, ignorés par Git
├── reports/               # tableaux et figures générés, ignorés par Git
├── requirements.txt       # dépendances propres à l'exploration
└── README.md
```

## 1. Préparer l'environnement

Depuis la racine du dépôt :

```bash
source .venv/bin/activate
python -m pip install -r experiments/aquamonitor/requirements.txt
```

Ces dépendances restent séparées des dépendances principales du projet : elles
servent à l'exploration, pas au fonctionnement de l'application AquaIA.

## 2. Lancer le notebook

Toujours depuis la racine du dépôt :

```bash
python -m jupyter lab experiments/aquamonitor/notebooks/aquamonitor_eda.ipynb
```

Dans Jupyter, choisir le kernel correspondant à `.venv`, puis exécuter les
cellules dans l'ordre avec `Shift+Enter`.

Le notebook commence par télécharger uniquement
`aquamonitor-jyu.parquet.gzip` (environ 1,4 Mo). Les images ne sont chargées que
si `LOAD_IMAGES = True` est activé dans la cellule de configuration.

## 3. Résultats attendus

Après exécution de la partie métadonnées :

```text
experiments/aquamonitor/reports/
├── dataset_summary.json
├── class_statistics.csv
├── class_distribution_individuals.png
└── camera_distribution_by_class.png
```

Ces fichiers constituent la base du futur rapport EDA. Le notebook vérifie
également qu'aucun `individual` ou `imaging_run` n'est partagé entre train,
validation et test.

## 4. Arrêter Jupyter

Dans le terminal où Jupyter fonctionne :

```text
Ctrl+C
```

Confirmer ensuite l'arrêt si Jupyter le demande.

## Étape suivante

Une fois le notebook compris et exécuté, la prochaine étape sera de préparer un
petit sous-ensemble `ImageFolder` groupé par individu, puis de lancer une
baseline DINOv3 en classification. Ne pas commencer par le dataset AquaMonitor
complet : AquaMonitor-JYU suffit pour valider toute la méthode.
