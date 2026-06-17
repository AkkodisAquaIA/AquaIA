# AquaIA — Computer vision for aquatic macro-invertebrate identification

[![ci](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/ci.yml)
[![cd](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/cd.yml/badge.svg?branch=development)](https://github.com/AkkodisAquaIA/AquaIA/actions/workflows/cd.yml)

Research project of the **LPL · Akkodis · Scimabio-Interface** consortium (2025-2028). Replaces destructive morphological identification of benthic macro-invertebrates with AI-based identification from images, using **YOLO + DINOv3 + SAM**.

This repository contains **Axis 2** (computer vision), led by Akkodis Research.

---

## Quick start

```bash
git clone https://github.com/AkkodisAquaIA/AquaIA.git
cd AquaIA
git checkout development
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

```bash
python main.py --help        # list subcommands
python main.py train --help
python main.py infer --help
```

The pre-trained models (YOLO, DINO and potentially other) may be located in different location depending on the system. For instance, by default HuggingFace will download models to ```~/.cache```. However, the VM does not have enough disk space inside ```~``` so large files (including pre-trained models) are located in ```aqua-ia-data``` instead. To specify where to find those models, you can set the ```AQUAIA_MODEL_ROOT_DIR```. On the linux VM:
```bash
export AQUAIA_MODEL_ROOT_DIR=/aqua-ia-data/models
```
In this case for DINO-based backbones, the software will search inside ```$AQUAIA_MODEL_ROOT_DIR/torch/hub``` instead of ```~/.cache/torch/hub```
---


## Repository structure

```
.
├── classification/         # DINOv2 / DINOv3 classification heads & training
├── data_cleaning/          # Embedding-based dataset cleaning
├── data_processing/        # Dataset preprocessing helpers
├── dataloading/            # PyTorch datasets and dataloaders
├── dataset_utils/          # Split, rename, class-distribution utilities
├── detection/              # Detection backends — YOLO, DINOv3, SAM
├── sharepoint_dataloading/ # SharePoint dataset downloader
├── docker/build/           # Multi-stage Dockerfile
├── deploy/                 # Compose stack, bootstrap script, ops runbook → deploy/README.md
├── docs/                   # CI/CD architecture reference → docs/cicd_architecture.md
├── tests/                  # Pytest suite → tests/README.md
├── .github/workflows/      # ci.yml + cd.yml
└── main.py                 # CLI entry point
```

---

## Documentation

| Topic | File |
|---|---|
| VM bootstrap, secrets, day-to-day ops, rollback | [`deploy/README.md`](deploy/README.md) |
| CI/CD architecture, tagging, future hardening | [`docs/cicd_architecture.md`](docs/cicd_architecture.md) |
| Test suite — what's covered and how to run | [`tests/README.md`](tests/README.md) |

---

## Local development — lint & tests

Install the dev tools once (virtual environment required on macOS/Homebrew):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Then use the `Makefile` shortcuts before every PR:

| Command | What it does |
|---------|-------------|
| `make lint` | Lint only the files changed vs `origin/development` — **identical scope to CI** |
| `make lint-fix` | Same, but auto-fix what ruff can |
| `make lint-all` | Lint every Python file in the repo |
| `make format` | Auto-format every Python file |
| `make test` | Run the pytest suite |
| `make ci` | `lint` + `test` — full local CI simulation |

> Run `make lint` before pushing to catch ruff errors locally instead of reading cryptic GitHub logs.

---

## Branch rules

- **`development`** is the default branch. All PRs target it.
- Never push directly — always open a pull request.
- Branch naming: `feat/<name>`, `fix/<name>`, `chore/<name>`, `docs/<name>`.
- No data files in the repo — datasets and weights go on SharePoint.

---

## License

TBD by the AquaIA consortium. Contact the maintainers before reuse outside consortium scope.
