# État des lieux — CI/CD AquaIA

## 1. Ce qui existe

- **Code Python**
  - Détection : [detection/](../detection/) — YOLO + DINOv3 + SAM
  - Classification : [classification/](../classification/)
  - Nettoyage de données : [data_cleaning/](../data_cleaning/)
  - Utils dataset : [dataset_utils/](../dataset_utils/)
  - Loader SharePoint : [sharepoint_dataloading/](../sharepoint_dataloading/)
- **Entrée unifiée** : [main.py](../main.py) avec sous-commandes `train` / `infer`
- **Requirements éclatés**
  - [requirements.txt](../requirements.txt) — full
  - [requirements-vm.txt](../requirements-vm.txt) — light VM
  - [requirements-gpu.txt](../requirements-gpu.txt)
  - [requirements-dev.txt](../requirements-dev.txt)
- **Lint** : [.pre-commit-config.yaml](../.pre-commit-config.yaml) + [pyproject.toml](../pyproject.toml) (ruff)
- **CI actuelle** : [.github/workflows/build.yml](../.github/workflows/build.yml) — `workflow_dispatch` uniquement (manuel), build et push DockerHub
- **Dockerfile** : [docker/build/Dockerfile.prod](../docker/build/Dockerfile.prod)
  - À noter : nommé `.prod` mais le workflow référence `Dockerfile` sans suffixe — incohérence
- **Déploiement actuel** : `scp` manuel local → VM (cf. `commande_vm_azure.txt`)
- **Repo** : `AkkodisAquaIA/AquaIA`, branche par défaut `development`

## 2. Problèmes / dettes détectés

À régler avant ou avec la mise en place de la CI/CD.

| # | Problème | Fichier | Impact |
|---|---|---|---|
| 1 | **Marqueurs de conflit non résolus** (`<<<<<<< HEAD … >>>>>>> development`) | [.gitignore:14-21](../.gitignore#L14-L21) | Casse `.gitignore`, bloquant |
| 2 | Workflow référence `Dockerfile` mais fichier = `Dockerfile.prod` | [.github/workflows/build.yml:28](../.github/workflows/build.yml#L28) | Build ne tourne pas |
| 3 | `COPY datasets/coco128` en dur | [docker/build/Dockerfile.prod:7](../docker/build/Dockerfile.prod#L7) | À supprimer (demande utilisateur) |
| 4 | `COPY .cache/` mais le dossier n'existe pas dans le repo | [docker/build/Dockerfile.prod:4](../docker/build/Dockerfile.prod#L4) | Build casse |
| 5 | `aquaia_archive.tar` (10 Ko) committé | [aquaia_archive.tar](../aquaia_archive.tar) | Binaire dans Git, à dégager |
| 6 | Aucun `.dockerignore` | racine | Image 2-3× trop grosse |
| 7 | Aucun test (`pytest`, smoke test) | — | CI ne peut rien valider de fonctionnel |
| 8 | Pas de `docker-compose.yml` côté VM | — | Déploiement non déclaratif |
| 9 | DockerHub (rate-limit, secrets en double) | workflow | Mieux : GHCR |
