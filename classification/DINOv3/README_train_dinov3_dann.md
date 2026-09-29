# DINOv3 + DANN

Entraînement d'un modèle **DINOv3** avec **Domain Adversarial Neural Network (DANN)** pour la classification taxonomique d'images provenant de deux domaines (`source` et `target`).

## Structure des données

```text
DATA_DIR/
├── source/
│   ├── train/<taxon>/*
│   ├── val/<taxon>/*
│   └── test/<taxon>/*
└── target/
    ├── train/<taxon>/*
    ├── val/<taxon>/*
    └── test/<taxon>/*
```

Les taxons peuvent être différents entre les deux domaines. Le script construit automatiquement une taxonomie globale unique.

---

## Principe

Le modèle est composé de :

```text
Image
  ↓
DINOv3
  ↓
Features
 ├── Taxon Head  → prédiction taxon
 └── GRL + Domain Head → prédiction domaine
```

La **Gradient Reversal Layer (GRL)** force le backbone à apprendre des représentations indépendantes du domaine.

---

## Configuration principale

### Données

```python
data_dir
run_dir
```

Chemins du dataset et du dossier de résultats.

### Entraînement

```python
epochs = 50
batch_size = 32
num_workers = 4
use_amp = True
```

### Fine-tuning

```python
unfreeze_last_n_blocks = 1
```

Nombre de blocs DINOv3 dégelés, au moins 1 pour que DANN puisse avoir un effet

### Learning rates

```python
lr_backbone = 3e-5
lr_head = 1e-3
lr_domain = 1e-4
```

### Adaptation de domaine

```python
domain_loss_weight = 5.0
grl_gamma = 10.0
```

- `domain_loss_weight` contrôle la force du DANN.
- `grl_gamma` contrôle la progression du coefficient de la GRL.

### Utilisation des labels cible

```python
use_target_labels = False
```

- `False` : DANN non supervisé (recommandé), les labels du domaine cible ne participent pas à l'apprentissage.
- `True` : les labels du domaine cible participent également à l'apprentissage.

### Focal Loss

```python
focal_gamma = 0.5
focal_alpha_mode = "auto"
focal_max_weight = 5.0
```

### Early Stopping

```python
early_patience = 49
early_min_delta = 1e-5
```

Basé sur la **loss de validation du domaine cible**.

---

## Fonction de coût

```text
Ltotal =
ws × Lsource
+ wt × Ltarget
+ β × Ldomain
```

avec :

```python
source_taxon_weight = 0.5
target_taxon_weight = 0.5
domain_loss_weight = 5.0
```

Si :

```python
use_target_labels = False
```

alors seule la loss taxonomique source est utilisée.

---

## Sorties

Chaque exécution génère :

```text
results_dann/
└── YYYYMMDD-HHMMSS/
    ├── config.json
    ├── class_to_idx.json
    ├── classes_by_split.json
    ├── results_dann.json
    ├── checkpoints/
    │   ├── best.pt
    │   └── last.pt
    └── tb/
```

---

## TensorBoard

```bash
tensorboard --logdir <run_dir_parent>
```

Principales métriques :

```text
train/total
train/source_taxon
train/domain
train/domain_acc

val/source_acc
val/target_acc
```

---

## Interprétation de `domain_acc`

```text
≈ 100 % : domaines très différents
≈ 50 %  : adaptation de domaine réussie
< 50 %  : adaptation très forte
```

L'objectif est généralement :

```text
domain_acc ≈ 50 %
```

tout en maximisant :

```text
target_val_acc
target_test_acc
```