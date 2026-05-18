# AquaIA — Computer vision for aquatic macro-invertebrate identification

[![ci](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/ci.yml)
[![cd-staging](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/cd-staging.yml/badge.svg?branch=development)](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/cd-staging.yml)

AquaIA is a research project of the **LPL · Akkodis · Scimabio-Interface** consortium (2025-2028). Its goal is to replace the destructive, time-consuming morphological identification of benthic macro-invertebrates — used today to score French rivers under the EU Water Framework Directive — with non-lethal AI-based identification from images and environmental DNA.

This repository contains **Axis 2** (computer vision), led by Akkodis Research: data preparation, model training, and inference for taxonomic identification of benthic macro-invertebrates using **YOLO + DINOv3 + SAM**.

Status: research / pre-operational. Operational deployment (Task 2.4 of the scientific dossier — REST API, end-user packaging) is not yet in scope.

---

## Table of contents

- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the CLI](#running-the-cli)
- [Dependency layout](#dependency-layout)
- [Testing](#testing)
- [Linting and formatting](#linting-and-formatting)
- [CI/CD pipeline](#cicd-pipeline)
  - [Overview](#overview)
  - [Workflows](#workflows)
  - [Container runtime model](#container-runtime-model)
  - [Versioning and release](#versioning-and-release)
  - [Rollback](#rollback)
  - [Architecture deep-dive](#architecture-deep-dive)
- [Branch model](#branch-model)
- [Deployment infrastructure](#deployment-infrastructure)
- [Contributing](#contributing)
- [License and consortium](#license-and-consortium)
- [References](#references)

---

## Repository structure

```
.
├── classification/         # DINOv2 / DINOv3 classification heads & training
├── data_cleaning/          # Embedding-based dataset cleaning (DINOv3, FiftyOne)
├── data_processing/        # Dataset preprocessing helpers (npy export, ...)
├── dataloading/            # PyTorch datasets and dataloaders
├── dataset_utils/          # One-off utilities: split, rename, class-distribution audit
├── detection/              # Detection backends — YOLO, DINOv3 (Plain-DETR), SAM
├── sharepoint_dataloading/ # SharePoint dataset downloader (MSAL + Graph)
├── IP/                     # Standalone scripts (CIFAR similarity experiments)
├── src/                    # Misc helpers (MNIST baseline)
├── docker/build/           # Multi-stage Dockerfile (base: pytorch:2.4 cuda12.1)
├── deploy/                 # docker-compose, bootstrap script, ops runbook
├── docs/                   # Architecture documentation
├── tests/                  # Pytest suite (smoke-level)
├── .github/workflows/      # GitHub Actions — ci, cd-staging, cd-prod
├── main.py                 # CLI entry point — `train` and `infer` subcommands
├── pyproject.toml          # ruff + pytest configuration
├── requirements.txt        # General-purpose dependencies
├── requirements-vm.txt     # Lighter set used in the production VM image
├── requirements-gpu.txt    # NVIDIA CUDA-specific extras
├── requirements-dev.txt    # Lint + tests + pre-commit
└── .pre-commit-config.yaml # ruff hooks
```

---

## Quick start

### Prerequisites

- **Python 3.12.10**.<br>
  Note: CI runs on Python 3.11, and the production Docker image is also 3.11 (PyTorch base image). The 3.11/3.12 discrepancy is tracked in the follow-up TODO list.
- A virtualenv is strongly recommended — see the [official `venv` documentation](https://docs.python.org/3/library/venv.html).
- **Docker 24+** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — only if you intend to run the actual vision models locally. Tests, lint, and CLI smoke work on CPU.
- **Git**.

### Installation

```bash
git clone https://github.com/AkkodisAquaIA/AquaIA.git
cd AquaIA
git checkout development           # default branch — see § Branch model
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

For the production-VM dependency set (no CUDA wheels included — torch comes from the image base) or the GPU dev set, see [Dependency layout](#dependency-layout).

### Running the CLI

The single entry point is `main.py`:

```bash
python main.py --help
python main.py train --help
python main.py infer --help
```

Two subcommands are exposed today:

| Subcommand | Purpose                                              | Default config                  |
| ---------- | ---------------------------------------------------- | ------------------------------- |
| `train`    | Train a detection model (YOLO or DINOv3 family)      | `detection/train_config.yaml`   |
| `infer`    | Run inference on train/test samples (latest run dir) | `detection/infer_config.yaml`   |

Each subcommand accepts a single `--config <path>` flag pointing at a YAML config. The model family (`dino*` / `yolo*`) is read from the config and dispatched to the matching backend.

---

## Dependency layout

| File                     | When to use                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| `requirements.txt`       | General-purpose / local development                                        |
| `requirements-vm.txt`    | Production VM image — torch and torchvision come from the PyTorch base image, so they are intentionally **not** listed here |
| `requirements-gpu.txt`   | Heavy GPU dev workloads — NVIDIA DALI, nvjpeg, nvtiff                      |
| `requirements-dev.txt`   | Lint, tests, hooks (ruff, pytest, pytest-cov, pre-commit)                  |

CI installs `requirements-vm.txt` + `requirements-dev.txt` after a separate CPU-torch install — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the exact sequence.

---

## Testing

```bash
pytest                          # full suite
pytest -v                       # verbose
pytest tests/test_imports.py    # quickest sanity check
pytest --collect-only           # discovery only, no execution
```

The test suite is intentionally minimal at this stage — smoke tests covering imports, CLI invocation, and a thin pure-Python pipeline check (`infer_output_project`). Heavier model-level evaluation is part of **Task 2.3** of the scientific dossier and will be added as the pipeline matures.

CI runs the same `pytest -q` command, so a green local run is a strong signal the PR will pass CI.

---

## Linting and formatting

```bash
ruff check .            # lint
ruff format .           # rewrite formatting
ruff format --check .   # CI-style check (no rewrite)
```

Both run automatically:

- **Locally**, on staged files via `pre-commit` (after `pre-commit install`).
- **In CI**, on every PR targeting `development` ([`ci.yml`](.github/workflows/ci.yml) → `lint` job).

Configuration lives in [`pyproject.toml`](pyproject.toml) (ruff section).

---

## CI/CD pipeline

### Overview

Every commit on `development` is automatically built, tested, and deployed to the staging environment on the project's Azure VM. Production deployments are explicit, version-tagged releases — promoted by tagging a commit with a `v*.*.*` semver tag and approving in the GitHub UI.

```
┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────────────┐
│ Pull req │────▶│   ci.yml     │     │                 │     │                    │
│  to dev  │     │ lint · test  │     │                 │     │                    │
└──────────┘     │ build (no    │     │                 │     │                    │
                 │ push)        │     │                 │     │                    │
                 └──────────────┘     │                 │     │                    │
                                      │                 │     │                    │
┌──────────┐     ┌──────────────┐     │   Docker Hub    │     │   Azure VM         │
│ Push to  │────▶│ cd-staging   │────▶│ <DOCKERHUB_NS>/ │────▶│ <STACK_PATH_       │
│   dev    │     │ build · push │     │ aquaia:dev-sha  │     │     STAGING>       │
└──────────┘     └──────────────┘     │                 │     │                    │
                                      │                 │     │                    │
┌──────────┐     ┌──────────────┐     │                 │     │                    │
│ Tag      │────▶│  cd-prod     │────▶│ <DOCKERHUB_NS>/ │────▶│ <STACK_PATH_PROD>  │
│ v*.*.*   │     │ build · push │     │ aquaia:vX.Y.Z   │     │ (after approval)   │
└──────────┘     │ approval gate│     │                 │     │                    │
                 └──────────────┘     └─────────────────┘     └────────────────────┘
```

### Workflows

| Workflow                                                              | Trigger                | Action                                            | Image tags pushed             | Secrets used                                                            |
| --------------------------------------------------------------------- | ---------------------- | ------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------- |
| [`ci.yml`](.github/workflows/ci.yml)                                  | PR to `development`    | Lint, test, Docker build (no push) + smoke test   | none (local to runner)        | none                                                                    |
| [`cd-staging.yml`](.github/workflows/cd-staging.yml)                  | Push to `development`  | Build, push to Docker Hub, deploy to VM staging   | `dev-<sha>` + `dev-latest`    | `DOCKERHUB_USER`, `DOCKERHUB_TOKEN`, `VM_HOST`, `VM_SSH_PRIVATE_KEY`    |
| [`cd-prod.yml`](.github/workflows/cd-prod.yml)                        | Tag `v*.*.*`           | Build, push, deploy to VM prod (manual approval)  | `vX.Y.Z` + `latest`           | Same set + GitHub Environment `production` with required reviewer       |

The floating `latest` tag only moves on prod tag pushes — `dev-latest` is its staging counterpart, so a developer pulling `latest` always gets the most recent reviewed release.

For the full secret-rotation procedure, the `production` environment setup, and the branch protection rules, see [`deploy/README.md`](deploy/README.md) §1 and §2.

### Container runtime model

The container does not start by running training: by default it stays up via `sleep infinity`. Workloads are launched on demand using **Compose profiles**. This pattern keeps the deployed stack idempotent, makes `docker exec` debugging trivial, and leaves room for an HTTP API later without restructuring.

| Profile  | Service          | Default command                | Behaviour                                               |
| -------- | ---------------- | ------------------------------ | ------------------------------------------------------- |
| `idle`   | `aquaia`         | `sleep infinity`               | Default, restart `unless-stopped`, healthchecked on GPU |
| `train`  | `aquaia-train`   | `python main.py train`         | One-shot, exits on completion, `restart: "no"`          |
| `infer`  | `aquaia-infer`   | `python main.py infer`         | One-shot, exits on completion, `restart: "no"`          |
| `api`    | `aquaia-api`     | (placeholder, exits 64)        | **Disabled** — reserved for Task 2.4 (FastAPI deploy)   |

Example commands once you are SSH'ed onto the VM:

```bash
ssh <DEPLOY_USER>@<VM_HOST>

# Run a training job on the staging stack
cd <STACK_PATH_STAGING>
docker compose -p aquaia-staging --profile train up

# exec into the idle container for ad-hoc work
docker compose -p aquaia-staging exec aquaia bash

# Tail logs
docker compose -p aquaia-staging logs -f --tail 200 aquaia
```

The corresponding prod stack lives in `<STACK_PATH_PROD>`, project name `aquaia-prod`. Container names are `aquaia-staging` / `aquaia-prod` for the idle service, `-train` / `-infer` for the one-shot ones. The placeholders `<VM_HOST>`, `<DEPLOY_USER>`, `<STACK_PATH_STAGING>`, and `<STACK_PATH_PROD>` are documented in the private Akkodis ops wiki.

### Versioning and release

Semver convention:

- **Patch** (`v0.1.1`) — bug fixes only, no model behaviour change.
- **Minor** (`v0.2.0`) — new features, backwards-compatible.
- **Major** (`v1.0.0`) — breaking changes (CLI flags, model interface, data format).

Cutting a release:

```bash
git checkout development
git pull --ff-only
git tag -a v0.1.0 -m "AquaIA 0.1.0 — initial production release"
git push origin v0.1.0
# cd-prod.yml triggers; manual approval is required in the GitHub UI
```

The semver tag becomes a stable reference for scientific reporting (Task 2.4 deliverables, consortium publications). 0 git tags posted to date — the first one will trigger the first prod deployment.

### Rollback

Every previously published image tag remains available on Docker Hub indefinitely (deploy jobs never prune remote tags, only local dangling images). To roll back:

```bash
ssh <DEPLOY_USER>@<VM_HOST>
cd <STACK_PATH_STAGING>        # or <STACK_PATH_PROD>
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=<previous-tag>/' .env
docker compose -p aquaia-staging pull
docker compose -p aquaia-staging --profile idle up -d
```

For the full runbook (prod-specific steps, post-rollback verification), see [`deploy/README.md`](deploy/README.md) §4.1.

### Architecture deep-dive

Pipeline architecture, image tagging strategy, branch transition plan, and future hardening (digest pinning, SBOM, auto-rollback) are documented in [`docs/cicd_architecture.md`](docs/cicd_architecture.md).

---

## Branch model

Current state (April 2026):

- **`development`** is the default branch and the source of truth. All PRs target it. CI/CD workflows trigger on it.
- **`main`** is currently obsolete. When the codebase is stable enough, `development` will be merged into `main` and the workflows will flip targets — a one-line change in `ci.yml` and `cd-staging.yml`. See [`docs/cicd_architecture.md`](docs/cicd_architecture.md) §6 for the migration plan.
- Feature branches: `feat/<short-name>`. Fixes: `fix/<short-name>`. Chores: `chore/<short-name>`.
- 0 git tags posted to date.

**Non-negotiable rules** (from the original repo charter):

- Never push directly to `main`. Always open a pull request.
- Never delete the `development` branch.
- Always branch off `development` for new work.
- No data files inside the repository — code only. Use SharePoint or another shared storage for datasets and model weights. This is also enforced by [`.gitignore`](.gitignore) and [`.dockerignore`](.dockerignore).

---

## Deployment infrastructure

- **Target**: Azure VM (Ubuntu 24.04 LTS, NVIDIA GPU).
- **Two stacks side by side**: `<STACK_PATH_STAGING>` and `<STACK_PATH_PROD>`. Each has its own `docker-compose.yml`, `.env`, and isolated volumes. Compose project names (`aquaia-staging` / `aquaia-prod`) provide additional isolation.
- **Volumes for `datasets/`, `models/`, `results/`, `cache/`** are bind-mounted from the host filesystem and **never baked into the image**. The image is reproducible across stacks; the data is not.
- **The VM only pulls** images from Docker Hub. CI never sends data to the VM beyond the compose file and a rendered `.env`.

For VM bootstrap, secrets, and operational procedures, see [`deploy/README.md`](deploy/README.md).

---

## Contributing

- Open an issue first for non-trivial changes.
- Branch (internal) or fork (external) from `development`.
- Naming: `feat/<short-name>`, `fix/<short-name>`, `chore/<short-name>`.
- Run `pytest` and `ruff check .` locally before pushing.
- Open a PR targeting `development`.
- All three CI status checks (`lint`, `test`, `build`) must be green to merge — soon to be enforced via branch protection (cf. [`deploy/README.md`](deploy/README.md) §2.3).
- At least one approval is expected on non-trivial PRs.

---

## License and consortium

**License**: TBD by the AquaIA consortium. Contact the maintainers before reuse outside the consortium scope.

**Consortium**: AquaIA is a research project of the **LPL · Akkodis · Scimabio-Interface** consortium (2025-2028). Akkodis Research leads Axis 2 (computer vision).

**Maintainer**: Samuel Beaussant (Akkodis Research). Other regular contributors: Zhijian Zhou, Sarah Laroui, Pierre Fa., Theo Dupont, Halim Bengana, Mehdi Mankai. See the [GitHub Contributors graph](https://github.com/AkkodisAquaIA/AquaIA/graphs/contributors) for the up-to-date list.

**Contact**: GitHub Issues for technical questions.

---

## References

- [`deploy/README.md`](deploy/README.md) — VM bootstrap, secrets, day-to-day ops, runbooks.
- [`docs/cicd_architecture.md`](docs/cicd_architecture.md) — Pipeline architecture, tagging strategy, future hardening.
- [`docs/cicd_etat_des_lieux.md`](docs/cicd_etat_des_lieux.md) — Original audit (French) that informed the current implementation. Kept for traceability.
- AquaIA scientific dossier (confidential, not linked) — context, scientific rationale, task breakdown.
