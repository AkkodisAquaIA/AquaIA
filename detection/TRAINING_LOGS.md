# Système de logs d'entraînement persistants — Architecture & Audit

## Contexte et problème

Les entraînements sont lancés directement dans un terminal SSH sur la VM (Tesla T4).
Si le terminal se ferme, la connexion tombe ou le process crashe, **tout est perdu** :
- les métriques epoch par epoch
- l'état de l'optimiseur / scheduler
- la trace des erreurs

Ce document décrit l'état actuel (audit), les lacunes identifiées, et les choix d'architecture
retenus pour y remédier.

---

## Audit de l'état actuel

### Comment un entraînement est lancé aujourd'hui

```
python -m detection.train --config detection/train_config.yaml
  └── train_from_config(config_path)
        └── train(config)
              ├── train_dino(config)   # si model.family = "dinov3"
              └── train_yolo(config)   # si model.family = "yolo*"
```

La commande est lancée **directement dans le terminal SSH**, sans tmux, screen, ni nohup.

### Structure des sorties actuelles (DINO)

```
{output.project}/
└── {YYYYMMDD_HHMMSS}/          ← run_dir créé au démarrage
    ├── weights/
    │   ├── best.pt              ← sauvegardé à chaque amélioration du val_loss ✅
    │   └── last.pt              ← sauvegardé UNIQUEMENT en fin d'entraînement ⚠️
    ├── last_training_state.pt   ← optimizer/scaler/scheduler — FIN seulement ⚠️
    ├── metrics.npy              ← toutes les métriques — FIN seulement ⚠️
    ├── resolved_config.yaml     ← config figée au démarrage ✅
    └── eval_predictions/        ← prédictions visuelles sur val — FIN seulement
```

### Structure des sorties actuelles (YOLO)

Délégué entièrement à Ultralytics `model.train()`.
Ultralytics gère ses propres logs dans `{run_dir}/` mais sans contrôle fin de notre côté.

### Lacunes identifiées

| # | Problème | Impact |
|---|----------|--------|
| 1 | `metrics.npy` écrit **seulement en fin de run** | Crash = zéro métrique récupérable |
| 2 | `last.pt` et `last_training_state.pt` écrits **seulement en fin de run** | Impossible de reprendre après un crash |
| 3 | Tout le logging passe par `print()` vers stdout | Fermeture du terminal = perte de tous les logs |
| 4 | Aucun fichier de log persistant | Impossible de tracer une erreur après le fait |
| 5 | Aucun registre de runs | Impossible de lister "qu'est-ce qui a tourné, quand, avec quel config" |
| 6 | Aucun indicateur "le process est vivant" | Impossible de savoir si un run est en cours ou mort |
| 7 | Lancement direct en SSH sans gestionnaire de session | Déconnexion SSH = arrêt immédiat du process |
| 8 | `save_training_state_checkpoint` en fin seulement | Reprise impossible depuis un epoch intermédiaire |

> Ces points sont partiellement reconnus dans le code : deux TODO explicites dans
> `detection/dino/training/run.py` lignes 23-24 et 231 :
> - `"# TODO : Add training state freq to tradeoff training speed for robustness"`
> - `"# /!\ ---- TODO : should be inside the training loop and remote"`

---

## Architecture cible

### Principe général

```
┌─────────────────────────────────────────────────────────┐
│  Session SSH (tmux / screen)                            │
│                                                         │
│  python -m detection.train --config train_config.yaml   │
│       │                                                 │
│       ▼                                                 │
│  TrainingLogger (nouveau)                               │
│       ├── train.jsonl    ← une ligne JSON par epoch     │
│       ├── train.log      ← log texte lisible humain     │
│       ├── heartbeat      ← timestamp mis à jour / batch │
│       └── run_meta.json  ← statut global du run         │
│                                                         │
│  CheckpointManager (amélioré)                           │
│       ├── best.pt                 (inchangé)            │
│       ├── last.pt                 (toutes les N epochs) │
│       └── last_training_state.pt  (toutes les N epochs) │
└─────────────────────────────────────────────────────────┘
         │
         ▼
  runs/registry.jsonl   ← registre global de tous les runs
```

### 1. Persistance de session — recommandation opérationnelle

**Avant toute implémentation code**, la première chose à faire est d'utiliser `tmux`
sur la VM pour que le process survive à la déconnexion SSH.

```bash
# Sur la VM, avant de lancer un entraînement
tmux new -s training

# Lancer l'entraînement dans la session tmux
python -m detection.train --config detection/train_config.yaml

# Détacher la session (le process continue même si SSH se ferme)
Ctrl+B, D

# Se reconnecter plus tard
tmux attach -t training
```

Alternative minimale sans tmux :
```bash
nohup python -m detection.train --config detection/train_config.yaml \
  > runs/current.log 2>&1 &
echo $! > runs/current.pid
```

### 2. TrainingLogger — logging persistant par epoch

Un objet `TrainingLogger` instancié au début du run, passé à la boucle d'entraînement.

**Fichiers écrits dans `{run_dir}/` :**

#### `train.jsonl` — une ligne JSON par epoch (format JSONL)
```json
{"epoch": 1, "train_loss": 2.341, "val_loss": 1.987, "lr": 0.0005, "timestamp": "2025-06-15T14:23:11", "elapsed_s": 312}
{"epoch": 2, "train_loss": 1.823, "val_loss": 1.654, "lr": 0.00048, "timestamp": "2025-06-15T14:28:45", "elapsed_s": 626}
```
- Écrit **flush immédiat** après chaque epoch → survit à un crash
- Lisible avec `tail -f train.jsonl` depuis un autre terminal
- Parseable facilement pour plots post-mortem

#### `train.log` — log texte pour humains
```
[2025-06-15 14:23:11] [INFO ] Run started — run_id=20250615_142200
[2025-06-15 14:23:11] [INFO ] Config: model=dinov3_small, epochs=50, batch=30, lr=0.0005
[2025-06-15 14:23:11] [INFO ] Dataset: datasets/coco_custom_match (train=1200, val=300)
[2025-06-15 14:23:11] [INFO ] Device: cuda (Tesla T4) | AMP: True
[2025-06-15 14:28:45] [EPOCH] 1/50 | train_loss=2.3410 | val_loss=1.9870 | lr=5.00e-4 | 5m12s
[2025-06-15 14:34:22] [EPOCH] 2/50 | train_loss=1.8230 | val_loss=1.6540 | lr=4.80e-4 | 5m37s
[2025-06-15 14:34:22] [BEST ] New best at epoch 2 — val_loss=1.6540 (was 1.9870)
```
- Utilise le module `logging` Python avec `FileHandler` + `StreamHandler`
- Remplace les `print()` actuels

#### `heartbeat` — fichier texte mis à jour toutes les N batches
```
2025-06-15T14:35:02 epoch=2 batch=45/120
```
- Écrase le fichier à chaque update (pas d'accumulation)
- Permet de détecter si un run est mort : si le timestamp est vieux de > 5 min, le process est probablement mort
- Script de monitoring : `watch -n 30 cat runs/20250615_142200/heartbeat`

#### `run_meta.json` — état global du run
```json
{
  "run_id": "20250615_142200",
  "status": "running",
  "started_at": "2025-06-15T14:22:00",
  "last_updated": "2025-06-15T14:35:02",
  "pid": 12345,
  "config": {"model": "dinov3_small", "epochs": 50, "batch": 30},
  "best_epoch": 2,
  "best_val_loss": 1.654,
  "current_epoch": 2,
  "total_epochs": 50
}
```
- Mis à jour après chaque epoch
- `status` : `"running"` | `"done"` | `"error"` | `"interrupted"`
- Écrit avec `json.dump` + flush

### 3. CheckpointManager — sauvegarde périodique intra-run

Remplace les appels directs à `save_model_checkpoint` / `save_training_state_checkpoint`.

**Paramètre `save_period`** (déjà dans `train_config.yaml`, actuellement ignoré par DINO) :
- `save_period: 0` → sauvegarder seulement le best (comportement actuel)
- `save_period: 5` → sauvegarder `last.pt` + `last_training_state.pt` toutes les 5 epochs

**Fichiers :**
```
weights/
├── best.pt                    ← inchangé (sauvegardé à chaque amélioration)
├── last.pt                    ← mis à jour toutes les save_period epochs
└── last_training_state.pt     ← optimizer + scaler + scheduler — même fréquence
```

**Reprise depuis un checkpoint (`--resume`) :**
- Détecter si `last_training_state.pt` existe dans le `run_dir`
- Charger l'état de l'optimizer, scaler, scheduler
- Reprendre depuis `epoch_saved + 1`
- Réécrire dans `run_meta.json` : `"status": "resumed"`

### 4. Registre global des runs — `runs/registry.jsonl`

Fichier à la racine du projet `detection/` (ou configurable).
Une ligne JSON par run, ajoutée au démarrage :

```json
{"run_id": "20250615_142200", "model": "dinov3_small", "dataset": "coco_custom_match", "epochs": 50, "status": "running", "run_dir": "runs/20250615_142200", "started_at": "2025-06-15T14:22:00", "pid": 12345}
{"run_id": "20250615_093100", "model": "yolo11n", "dataset": "coco_custom_match", "epochs": 100, "status": "done", "run_dir": "runs/20250615_093100", "started_at": "2025-06-15T09:31:00", "pid": 11234}
```

Mise à jour en fin de run (status → `done` ou `error`).

Script utilitaire `detection/list_runs.py` pour afficher le registre :
```
$ python -m detection.list_runs

run_id               model           dataset               epochs  status      started_at
20250615_142200      dinov3_small    coco_custom_match     50      running     2025-06-15 14:22
20250615_093100      yolo11n         coco_custom_match     100     done        2025-06-15 09:31
20250614_180022      dinov3_small    coco_custom_match     50      error       2025-06-14 18:00
```

### 5. Portée du système selon le modèle

| Fonctionnalité | DINO | YOLO |
|----------------|------|------|
| `TrainingLogger` (JSONL + log file) | Intégration directe dans la boucle | Via callback Ultralytics |
| Heartbeat | Intégration directe | Via callback Ultralytics |
| `run_meta.json` | Intégration directe | Wrapping autour de `model.train()` |
| `CheckpointManager` (save_period) | Intégration directe | Délégué à Ultralytics (`save_period`) |
| Registre global | Commun aux deux | Commun aux deux |
| Reprise (`--resume`) | À implémenter | Natif Ultralytics (`resume=True`) |

---

## Structure des fichiers à créer

```
detection/
├── logging/                        ← nouveau package
│   ├── __init__.py
│   ├── training_logger.py          ← TrainingLogger (JSONL + log file + heartbeat)
│   ├── run_registry.py             ← registre global des runs
│   └── checkpoint_manager.py       ← sauvegarde périodique + reprise
├── list_runs.py                    ← script CLI pour lister les runs
├── TRAINING_LOGS.md                ← ce fichier
└── dino/training/run.py            ← modifié pour intégrer le nouveau système
```

---

## Ce qui NE change PAS

- La structure du `run_dir` (compatible avec les runs existants)
- Le format de `best.pt` et `last.pt` (même format `torch.save`)
- Le `train_config.yaml` (on ajoute seulement `logging:` section)
- Le point d'entrée `python -m detection.train`

---

## Extension `train_config.yaml` prévue

```yaml
# Nouvelle section à ajouter
logging:
  # Fréquence de sauvegarde du checkpoint intermédiaire (0 = jamais, N = toutes les N epochs)
  save_period: 5
  # Fréquence de mise à jour du heartbeat (en batches)
  heartbeat_every_n_batches: 10
  # Répertoire du registre global des runs
  registry_path: "runs/registry.jsonl"
```

---

## Ordre d'implémentation prévu

1. **`TrainingLogger`** — logging JSONL + fichier texte (remplace les `print()`)
2. **Heartbeat** — fichier mis à jour toutes les N batches
3. **`CheckpointManager`** — save_period pour `last.pt` + training state
4. **`run_meta.json`** — statut global du run
5. **Registre global** — `runs/registry.jsonl` + `list_runs.py`
6. **Reprise (`--resume`)** — chargement depuis `last_training_state.pt`
7. **Intégration YOLO** — via callbacks Ultralytics

Chaque étape est indépendante et testable séparément avec un petit modèle en local.

---

## Tests

### En local (sans GPU, sans dataset)

Le script `detection/test_training_logs.py` simule des epochs sans torch ni dataset.

```bash
# Run normal (5 epochs complets)
python -m detection.test_training_logs

# Crash simulé à l'epoch 2
python -m detection.test_training_logs --crash

# Crash puis resume (scénario complet)
python -m detection.test_training_logs --crash --resume

# Avec affichage du registre
python -m detection.test_training_logs --crash --resume --list
```

Résultat attendu après `--crash --resume` :

| Fichier | Contenu attendu |
|---------|----------------|
| `train.jsonl` | 5 lignes — epoch 1 (avant crash) + epochs 2-5 (après resume) |
| `train.log` | Log continu sans rupture visible, ~30 lignes |
| `run_meta.json` | `status: done`, `current_epoch: 5` |
| `heartbeat` | `epoch=5 batch=10/10` |

### Sur la VM (Tesla T4, entraînement réel)

```bash
# Connexion
ssh -i /path/to/key.pem user@vm-ip

# Lancer dans tmux pour survivre à la déconnexion SSH
tmux new -s training
python main.py train --config detection/train_config.yaml

# Détacher (le process continue même si SSH se ferme)
Ctrl+B, D

# Suivre les métriques en temps réel depuis un autre terminal
tail -f runs/<run_id>/train.jsonl

# Vérifier que le process est vivant (timestamp récent = OK)
cat runs/<run_id>/heartbeat

# Lister tous les runs
python -m detection.list_runs

# Se reconnecter à la session tmux
tmux attach -t training
```

En cas de crash, reprendre depuis le dernier checkpoint :

```bash
python main.py train --config detection/train_config.yaml --resume runs/<run_id>
```

### Critères de validation

- [x] `train.jsonl` contient une ligne par epoch après crash simulé (Ctrl+C)
- [x] `train.log` lisible et complet
- [x] `heartbeat` mis à jour pendant l'entraînement
- [x] `run_meta.json` status = `"interrupted"` après Ctrl+C
- [x] `last.pt` sauvegardé toutes les `save_period` epochs, pas seulement en fin
- [x] `list_runs.py` affiche le run avec le bon statut
- [x] Reprise depuis `--resume` repart de l'epoch sauvegardée

---

## Notes opérationnelles

### Checkpoint bloquant

`torch.save()` est synchrone — la boucle d'entraînement est en pause pendant la sauvegarde.
Pour DINO sur T4 avec `save_period: 5`, l'impact est de 1-3 secondes toutes les 5 epochs,
négligeable face à la durée d'une epoch. Si ça devient un goulot, la solution est de déléguer
le `torch.save` à un thread avec clonage des tenseurs au préalable.
