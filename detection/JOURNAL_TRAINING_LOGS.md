# Journal de modifications — Système de logs d'entraînement persistants (AquaIA)

---

## 1) Objectif du travail

Mettre en place un système de logging robuste et persistant pour les entraînements de modèles
lancés sur la VM (Tesla T4), de façon à ce que :

- les métriques epoch par epoch survivent à un crash ou à une déconnexion SSH
- l'état de l'optimiseur/scheduler soit sauvegardé périodiquement (reprise possible)
- un fichier de log lisible reste sur disque après la fin ou l'échec du run
- on puisse savoir à tout moment si un run est en cours, mort ou terminé
- on puisse lister tous les runs passés avec leur statut

```bash
# Lancement cible (inchangé côté utilisateur)
python -m detection.train --config detection/train_config.yaml

# Reprise après crash
python -m detection.train --config detection/train_config.yaml --resume runs/20250615_142200

# Liste des runs
python -m detection.list_runs
```

---

## 2) Fichiers modifiés / créés

*(mis à jour au fil des implémentations)*

| Fichier | Statut | Description |
|---------|--------|-------------|
| `detection/TRAINING_LOGS.md` | ✅ Créé | Audit + architecture cible (document de référence) |
| `detection/JOURNAL_TRAINING_LOGS.md` | ✅ Créé | Ce journal (trace des changements) |
| `detection/logging/__init__.py` | ✅ Créé | Package logging |
| `detection/logging/training_logger.py` | ✅ Créé | `TrainingLogger` — JSONL + log texte + heartbeat |
| `detection/logging/run_registry.py` | ✅ Créé | Registre global des runs |
| `detection/logging/checkpoint_manager.py` | ✅ Créé | Sauvegarde périodique + reprise |
| `detection/list_runs.py` | ✅ Créé | Script CLI pour lister les runs |
| `detection/dino/training/run.py` | ✅ Modifié | Intégration du nouveau système + logique `--resume` |
| `detection/train.py` | ✅ Modifié | Propagation du paramètre `resume_dir` |
| `detection/checkpoint.py` | ✅ Modifié | Ajout `load_training_state_checkpoint` |
| `main.py` | ✅ Modifié | Ajout argument `--resume` au sous-commande `train` |
| `detection/yolo/training/run.py` | ⏳ À modifier | Intégration via callbacks Ultralytics |
| `detection/train_config.yaml` | ✅ Modifié | Ajout section `logging:` + `output.project` |

---

## 3) Audit de l'état initial

### 3.1 Problèmes identifiés dans le code existant

**Fichiers concernés :** `detection/dino/training/run.py`, `detection/checkpoint.py`

#### Problème

Deux TODOs explicites dans le code (sabeaussan / zzj-bj) reconnaissent le problème :

```python
# TODO :
# - Add training state freq to tradeoff training speed for robustness
# - Offload logging and checkpointing to a remote process
```

```python
# /!\ ---- TODO : should be inside the training loop and remote
# save last model
save_model_checkpoint(path=os.path.join(weights_dir, "last.pt"), model=model)
```

#### Lacunes recensées

| # | Problème | Fichier | Ligne | Impact |
|---|----------|---------|-------|--------|
| 1 | `metrics.npy` écrit seulement en fin de run | `dino/training/run.py` | 245 | Crash = zéro métrique |
| 2 | `last.pt` écrit seulement en fin de run | `dino/training/run.py` | 232 | Reprise impossible |
| 3 | `last_training_state.pt` écrit seulement en fin de run | `dino/training/run.py` | 234–241 | Reprise depuis epoch intermédiaire impossible |
| 4 | Tout le logging via `print()` vers stdout | `dino/training/run.py` | passim | Fermeture terminal = perte totale des logs |
| 5 | Aucun fichier de log persistant | — | — | Impossible de tracer une erreur post-crash |
| 6 | Aucun registre de runs | — | — | Impossible de lister ce qui a tourné |
| 7 | Aucun indicateur "process vivant" | — | — | Impossible de détecter un run mort |
| 8 | Lancement direct en SSH sans tmux/screen | — | — | Déconnexion SSH = arrêt immédiat |

#### Sorties actuelles (DINO)

```
{output.project}/
└── {YYYYMMDD_HHMMSS}/
    ├── weights/
    │   ├── best.pt              ← OK (sauvegardé à chaque amélioration val_loss)
    │   └── last.pt              ← ⚠️ FIN seulement
    ├── last_training_state.pt   ← ⚠️ FIN seulement
    ├── metrics.npy              ← ⚠️ FIN seulement
    ├── resolved_config.yaml     ← OK (créé au démarrage)
    └── eval_predictions/        ← FIN seulement (normal)
```

#### Impact

État du système au démarrage du travail :
- Un crash à l'epoch 48/50 → perte de toutes les métriques et de `last.pt`
- Reconnexion SSH impossible pour voir les logs en cours
- Aucun moyen de savoir si le run a crashé ou tourne encore

---

## 4) Modifications techniques détaillées

*(chaque section est remplie au moment de l'implémentation)*

---

### 4.1 [⏳ À implémenter] `TrainingLogger` — logging JSONL + fichier texte

**Fichier créé :** `detection/logging/training_logger.py`

#### Problème

Tous les logs vont vers `print()` → stdout. Si le terminal SSH se ferme, il ne reste
aucune trace de l'entraînement.

#### Solution prévue

Classe `TrainingLogger` instanciée au début du run, écrivant dans :

- **`{run_dir}/train.jsonl`** : une ligne JSON par epoch, flush immédiat
  ```json
  {"epoch": 1, "train_loss": 2.341, "val_loss": 1.987, "lr": 0.0005, "timestamp": "...", "elapsed_s": 312}
  ```
- **`{run_dir}/train.log`** : log texte lisible, module `logging` Python avec `FileHandler` + `StreamHandler`
- **`{run_dir}/heartbeat`** : fichier texte mis à jour toutes les N batches
  ```
  2025-06-15T14:35:02 epoch=2 batch=45/120
  ```
- **`{run_dir}/run_meta.json`** : état global du run
  ```json
  {"run_id": "...", "status": "running", "pid": 12345, "best_val_loss": 1.654, ...}
  ```

Remplace tous les `print()` de `dino/training/run.py`.

#### Impact attendu

Toute information produite pendant l'entraînement reste sur disque même après crash ou
déconnexion SSH.

---

### 4.2 [⏳ À implémenter] `CheckpointManager` — sauvegarde périodique intra-run

**Fichier créé :** `detection/logging/checkpoint_manager.py`  
**Fichier modifié :** `detection/dino/training/run.py`

#### Problème

`last.pt` et `last_training_state.pt` ne sont sauvegardés qu'en fin de run.
Un crash à epoch 48/50 → perte de l'état de l'optimiseur et du modèle last.

#### Solution prévue

`CheckpointManager` gérant la sauvegarde de `last.pt` + `last_training_state.pt`
toutes les `save_period` epochs (configurable dans `train_config.yaml`).

```yaml
# train_config.yaml — nouvelle section
logging:
  save_period: 5          # 0 = jamais, N = toutes les N epochs
  heartbeat_every_n_batches: 10
  registry_path: "runs/registry.jsonl"
```

`save_period: 0` → comportement actuel (best seulement), sans régression.

#### Impact attendu

Reprise possible depuis n'importe quelle epoch sauvegardée via `--resume`.

---

### 4.3 [⏳ À implémenter] Registre global des runs

**Fichier créé :** `detection/logging/run_registry.py`  
**Fichier créé :** `detection/list_runs.py`

#### Problème

Aucun moyen de savoir quels runs ont été lancés, quand, avec quel config, et quel est leur
statut final.

#### Solution prévue

Fichier `runs/registry.jsonl` (une ligne JSON par run) :
```json
{"run_id": "20250615_142200", "model": "dinov3_small", "epochs": 50, "status": "running", "pid": 12345, ...}
{"run_id": "20250614_180022", "model": "dinov3_small", "epochs": 50, "status": "error", ...}
```

Script CLI `list_runs.py` :
```
$ python -m detection.list_runs

run_id               model           dataset               epochs  status      started_at
20250615_142200      dinov3_small    coco_custom_match     50      running     2025-06-15 14:22
20250614_180022      dinov3_small    coco_custom_match     50      error       2025-06-14 18:00
```

#### Impact attendu

Traçabilité complète de tous les runs, lisible en une commande.

---

### 4.4 [✅ Implémenté] Reprise depuis checkpoint (`--resume`)

**Fichiers modifiés :** `main.py`, `detection/train.py`, `detection/checkpoint.py`, `detection/logging/training_logger.py`, `detection/dino/training/run.py`

#### Problème

Si un run crashe à l'epoch 30/50 et que `last_training_state.pt` a été sauvegardé,
il n'existe aucun mécanisme pour reprendre depuis cet epoch sans repartir de zéro.

#### Solution implémentée

```bash
python main.py train --config detection/train_config.yaml --resume runs/20250615_142200
```

Flux d'exécution :
1. `main.py` parse `--resume <run_dir>` et le transmet à `handle_train`
2. `train_from_config(config_path, resume_dir=...)` → `train_dino(config, resume_dir=...)`
3. Dans `train_dino` :
   - Réutilise le même `run_dir` et `run_id` (pas de nouveau dossier)
   - Charge `weights/last.pt` → poids du modèle (avant `torch.compile`)
   - Charge `last_training_state.pt` → optimizer, scaler, scheduler via `load_training_state_checkpoint`
   - Lit `run_meta.json` → récupère `best_val_loss` pour que `best.pt` reste cohérent
   - Lance la boucle depuis `start_epoch + 1` au lieu de 0
4. `TrainingLogger` en mode `resume=True` :
   - Charge le `run_meta.json` existant (status → `"resumed"`, pid mis à jour)
   - **Appende** à `train.log` et `train.jsonl` (mode `"a"`) — l'historique complet reste lisible
   - Pas de `register_run` (l'entrée dans le registre existe déjà)

#### Nouvelle fonction dans `checkpoint.py`

```python
def load_training_state_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device)
```

#### Impact

Un crash ne signifie plus repartir de zéro — on reprend depuis le dernier checkpoint périodique.

---

### 4.5 [⏳ À implémenter] Intégration YOLO

**Fichier modifié :** `detection/yolo/training/run.py`

#### Problème

YOLO délègue tout à Ultralytics `model.train()`. Pas de contrôle sur le logging,
pas de heartbeat, pas d'intégration avec le registre global.

#### Solution prévue

Utiliser les callbacks Ultralytics pour :
- Créer le `run_meta.json` au démarrage
- Mettre à jour le heartbeat dans le callback `on_train_batch_end`
- Écrire une ligne JSONL dans `on_train_epoch_end`
- Marquer `status: "done"` ou `"error"` dans `on_train_end`
- Enregistrer dans le registre global

Reprise : `model.train(..., resume=True)` — natif Ultralytics.

#### Impact attendu

Même niveau de traçabilité pour YOLO que pour DINO.

---

## 5) Tests de validation prévus

Test local avec un modèle minimal (sans GPU, avant de tester sur la VM) :

```yaml
# test_config_local.yaml
model:
  family: "yolo11"
  size: "n"
  init: "pretrained"
training:
  epochs: 3
  batch: 4
  device: "cpu"
logging:
  save_period: 1
  heartbeat_every_n_batches: 2
```

### Critères de validation

| # | Test | Critère |
|---|------|---------|
| 1 | Crash simulé (Ctrl+C à epoch 2/3) | `train.jsonl` contient 2 lignes (epochs 1 et 2) |
| 2 | Crash simulé | `train.log` lisible et complet jusqu'au crash |
| 3 | Heartbeat | `heartbeat` mis à jour pendant l'entraînement |
| 4 | Crash simulé | `run_meta.json` → `"status": "interrupted"` |
| 5 | Checkpoint périodique | `last.pt` présent après epoch 1 (save_period=1) |
| 6 | Registre | `list_runs.py` affiche le run avec le bon statut |
| 7 | Reprise | `--resume` repart de l'epoch 2, pas de l'epoch 0 |
| 8 | Run complet | `run_meta.json` → `"status": "done"` |

---

## 6) Recommandation opérationnelle immédiate

**Avant toute implémentation**, utiliser `tmux` sur la VM pour que le process survive
à la déconnexion SSH :

```bash
# Sur la VM
tmux new -s training
python -m detection.train --config detection/train_config.yaml

# Détacher (le process continue même si SSH se ferme)
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

---

## 7) Résultats observés

| Problème | Statut |
|----------|--------|
| Métriques perdues en cas de crash | ✅ Résolu — `train.jsonl` flush par epoch |
| `last.pt` perdu en cas de crash | ✅ Résolu — `CheckpointManager` sauvegarde toutes les N epochs |
| Aucun fichier de log persistant | ✅ Résolu — `train.log` via `logging.FileHandler` |
| Aucun registre de runs | ✅ Résolu — `runs/registry.jsonl` + `list_runs.py` |
| Aucun indicateur "process vivant" | ✅ Résolu — `heartbeat` mis à jour tous les N batches |
| Reprise depuis checkpoint impossible | ✅ Résolu — `--resume <run_dir>` recharge last.pt + last_training_state.pt |
| Intégration YOLO | ⏳ À implémenter (callbacks Ultralytics) |

### Smoke test local validé (sans GPU)

```
[2026-06-15 14:21:57] [INFO ] Run started — run_id=test_run | pid=59670
[2026-06-15 14:21:57] [INFO ] Model: dinov3_small (pretrained) | epochs=10 | batch=4 | lr=0.001
[2026-06-15 14:21:57] [INFO ] [EPOCH   1/10] train_loss=1.5000 | val_loss=1.2000 | lr=1.00e-03 | 0m0s
[2026-06-15 14:21:57] [INFO ] [BEST ] New best at epoch 1 — val_loss=1.2000
[2026-06-15 14:21:57] [INFO ] Training complete — total time: 0m0s
All assertions passed
  JSONL:      {"epoch": 1, "train_loss": 1.5, "val_loss": 1.2, ...}
  heartbeat:  2026-06-15T12:21:57 epoch=1 batch=2/10
  meta status: done | best_val_loss: 0.9
```

---

## 8) Récapitulatif des changements par fichier

| Fichier | Modification | Section |
|---------|-------------|---------|
| `detection/TRAINING_LOGS.md` | Audit initial + architecture cible | — |
| `detection/JOURNAL_TRAINING_LOGS.md` | Ce journal | — |
| `detection/logging/__init__.py` | Nouveau package — exports publics | 4.1 |
| `detection/logging/training_logger.py` | `TrainingLogger` — JSONL + log texte + heartbeat + run_meta | 4.1 |
| `detection/logging/checkpoint_manager.py` | `CheckpointManager` — best + last périodique | 4.2 |
| `detection/logging/run_registry.py` | Registre global — append + update par run_id | 4.3 |
| `detection/list_runs.py` | CLI `python -m detection.list_runs` | 4.3 |
| `detection/dino/training/run.py` | Intégration complète — print() → logger, try/except, heartbeat, checkpoint + logique resume | 4.1–4.4 |
| `detection/train_config.yaml` | Ajout `output.project` + section `logging:` | 4.1–4.2 |
| `main.py` | Argument `--resume <run_dir>` au sous-commande `train` | 4.4 |
| `detection/train.py` | Propagation `resume_dir` vers `train_dino` | 4.4 |
| `detection/checkpoint.py` | Ajout `load_training_state_checkpoint(path, device)` | 4.4 |
| `detection/logging/training_logger.py` | Mode `resume=True` — append logs, charge meta existant | 4.4 |
