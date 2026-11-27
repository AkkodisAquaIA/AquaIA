# Crop detection to classification pipeline

Yang Benjamin — 01/10/2026

## Objectif

Ce pipeline transforme un dataset de détection au format YOLO en dataset de
classification. Chaque bounding box est découpée pour produire une image ne
contenant qu'un seul macroinvertébré, puis le crop est rangé dans le dossier de
sa classe.

## Utilisation dans un terminal

Les commandes suivantes sont à lancer depuis la racine du projet. Le dossier de
sortie doit être nouveau ou vide.

### Dataset sans splits

```bash
python3 data_processing/crop_detection_to_classification.py \
    --images-dir ./datasets/mon_dataset/images \
    --labels-dir ./datasets/mon_dataset/labels \
    --output-dir ./datasets/mon_dataset_classification \
    --classes-yaml ./datasets/mon_dataset/data.yaml
```

### Dataset avec des splits

La structure attendue est `images/train`, `images/val`, `labels/train`,
`labels/val`, etc.

```bash
python3 data_processing/crop_detection_to_classification.py \
    --dataset-dir ./datasets/mon_dataset \
    --splits train val \
    --output-dir ./datasets/mon_dataset_classification \
    --classes-yaml ./datasets/mon_dataset/data.yaml \
    --classes-json ./datasets/classes_reference.json
```

Sans l'option `--splits`, le script cherche par défaut les splits `train`, `val`
et `test`.

Lorsque `--classes-yaml` et `--classes-json` sont fournis ensemble, le YAML reste
la source des identifiants YOLO et le JSON sert de référentiel de comparaison.
Le JSON doit avoir la forme suivante :

```json
[
    {"name": "Macroinvertebrate"},
    {"name": "Arthropoda_Insecta_Diptera"}
]
```

Le terminal affiche pour chaque classe YAML :

- si elle est présente ou absente du JSON ;
- son nombre d'annotations et d'images ;
- la répartition des annotations par split ;
- les classes inutilisées et celles présentes uniquement dans le JSON.

## Utilisation dans un script Python

```python
from pathlib import Path

from data_processing.crop_detection_to_classification import crop_splits


crops = crop_splits(
    dataset_dir=Path("./datasets/mon_dataset"),
    output_dir=Path("./datasets/mon_dataset_classification"),
    taxonomy_path=Path("data_processing/resources/taxonomy_prefixes.json"),
    yaml_path=Path("./datasets/mon_dataset/data.yaml"),
    json_path=Path("./datasets/classes_reference.json"),
    splits=("train", "val"),
)

print(f"{len(crops)} crops créés")
```

Pour un dataset sans splits, utiliser la fonction `crop_directories()`.

## Fonctionnement

- Le script lit les classes dans le fichier YAML.
- Chaque fichier label est associé à l'image portant le même nom.
- Les coordonnées YOLO normalisées sont converties en pixels.
- Chaque bounding box est découpée et enregistrée dans le dossier de sa classe.
- Les noms taxonomiques connus sont développés. Les autres noms de classes sont
  conservés tels qu'ils sont écrits dans le YAML.

Le dossier de sortie contient également :

- `manifest_crops.csv` : provenance et coordonnées de chaque crop ;
- `table_conversion_classes.csv` : correspondance des noms de classes ;
- `annotations_ignorees.csv` : annotations qui n'ont pas pu être traitées.

## Table de conversion des classes

Le fichier `table_conversion_classes.csv` indique comment chaque classe du YAML
a été interprétée. Le nom indiqué dans la colonne `nom_standard` est celui utilisé
pour créer le dossier contenant les crops.

Il contient les colonnes suivantes :

- `class_id` : identifiant numérique utilisé dans les labels YOLO ;
- `nom_encode` : nom d'origine lu dans le YAML ;
- `nom_standard` : nom final utilisé dans le dossier de sortie ;
- `source_nom` : fichier depuis lequel le nom a été chargé ;
- `statut` : traitement appliqué au nom.

Les principaux statuts sont :

- `converti_par_prefixes` : les préfixes taxonomiques ont été développés, par
  exemple `Arthr_Insec_Dipte` devient `Arthropoda_Insecta_Diptera` ;
- `deja_complet` : le nom taxonomique était déjà écrit en entier ;
- `nom_conserve` : le nom ne suit pas le format taxonomique connu et reste
  inchangé, par exemple `Macroinvertebrate`.
