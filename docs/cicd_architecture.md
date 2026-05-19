# Architecture CI/CD — AquaIA

Reference document for the AquaIA CI/CD pipeline. Paired with the operational runbook at [`deploy/README.md`](../deploy/README.md): this file describes **what** and **why**; `deploy/README.md` describes **how** to operate it.

> **Placeholders.** `<VM_HOST>`, `<DEPLOY_USER>`, `<DOCKERHUB_NAMESPACE>`, `<STACK_PATH_STAGING>` stand for values stored in GitHub Secrets / Variables and the private Akkodis ops wiki.

---

## 1. Overview

Two workflows, one image, one stack:

```
                        ┌──────────────────────────────────────────┐
                        │             GitHub Actions                │
                        │                                          │
PR  → development       │  ci.yml   lint · pytest · build no-push  │
push → development      │  cd.yml   build → Docker Hub → deploy    │
                        └─────────────────────┬────────────────────┘
                                              │ SSH (deploy@<VM_HOST>)
                                              ▼
                        ┌──────────────────────────────────────────┐
                        │           Azure VM (<VM_HOST>)            │
                        │  <STACK_PATH_STAGING>/                   │
                        │    ├─ docker-compose.yml                 │
                        │    ├─ .env  (IMAGE_TAG=dev-<sha>)        │
                        │    └─ {datasets,models,results,cache}/   │
                        └──────────────────────────────────────────┘
```

A single Dockerfile produces a single image: `<DOCKERHUB_NAMESPACE>/aquaia:<tag>`. The VM only **pulls**; CI never sends data beyond the compose file and a rendered `.env`.

---

## 2. Workflows

| Workflow | Trigger | Builds | Pushes to Docker Hub | Deploys to VM |
|---|---|---|---|---|
| `ci.yml` | PR to `development` | yes (local, no push) | no | no |
| `cd.yml` | push to `development` | yes | yes — `dev-<sha>` + `dev-latest` | `<STACK_PATH_STAGING>` |

---

## 3. GitHub secrets and variables

*Settings → Secrets and variables → Actions* on `AkkodisAquaIA/AquaIA`:

### Secrets (encrypted)

| Secret | Used by | Purpose |
|---|---|---|
| `DOCKERHUB_USERNAME` | `cd.yml` | Docker Hub username with **write** access to `<NAMESPACE>/aquaia` |
| `DOCKERHUB_TOKEN` | `cd.yml` | Docker Hub PAT — **Read & Write** scope |
| `VM_HOST` | `cd.yml` | Azure VM IP or DNS hostname |
| `VM_SSH_PRIVATE_KEY` | `cd.yml` | Private key for the `deploy` user on the VM |

### Repository Variables (not credentials)

| Variable | Used by | Purpose |
|---|---|---|
| `DOCKERHUB_NAMESPACE` | `cd.yml` | Docker Hub org or username owning `<NAMESPACE>/aquaia` |
| `STACK_DIR_STAGING` | `cd.yml` | Absolute path to the staging stack directory on the VM |

Project-internal conventions with no exploit value (`aquaia-staging`, image name `aquaia`, Linux user `deploy`) are hardcoded in the workflows and Compose file — they are part of the public contract the repo describes.

---

## 4. Image tagging strategy

| Origin | SHA-pinned tag | Floating tag | Lifetime |
|---|---|---|---|
| CD push | `dev-<7-char-sha>` | `dev-latest` | Retained on Docker Hub (used for rollback); pruned from VM after each deploy |
| CI on PR | `aquaia:ci-<full-sha>` (local only) | — | Discarded with the runner |

To pin a stable version on the VM, edit `.env` and set `IMAGE_TAG=dev-<sha>` — see [`deploy/README.md §4.2`](../deploy/README.md).

---

## 5. Runbook cross-reference

| Scenario | Where it lives |
|---|---|
| Bootstrap a fresh VM | `deploy/README.md` §1 |
| Configure GitHub secrets and variables | `deploy/README.md` §2 |
| Trigger a deploy | `deploy/README.md` §3.1 |
| Run a one-shot `train` or `infer` job | `deploy/README.md` §3.2 |
| Rollback to a previous image | `deploy/README.md` §4.1 |
| Pin a stable version | `deploy/README.md` §4.2 |
| Rotate the CI/CD SSH key | `deploy/README.md` §4.3 |
| Rotate the Docker Hub PAT | `deploy/README.md` §4.4 |
| Re-target `development` → `main` | `deploy/README.md` §4.6 + §6 below |

---

## 6. État transitoire `main` / `development`

- `development` is the GitHub default branch (active, all PRs target it).
- `main` is currently obsolete.

When `main` is merged up to date and made the default branch, two one-line edits suffice (see `deploy/README.md` §4.6). Nothing on the VM, in Docker Hub, or in the secrets needs to change.

---

## 7. Future hardening

- **`VM_SSH_KNOWN_HOSTS` secret** — pin the VM host key instead of TOFU via `ssh-keyscan` on every run.
- **Auto-rollback on failed smoke** — rewrite `IMAGE_TAG` to the previous value if the smoke test fails.
- **SBOM / vulnerability scan** — add `trivy` or `grype` to the `build-push` job.
- **Image digest pinning** — consume `${IMAGE_REF}@sha256:<digest>` instead of the human-readable tag.
- **API service** — flip `aquaia-api` from `__disabled__` to `api` once the FastAPI entry point lands (Task 2.4).

---

## 8. Design aside — cd-prod.yml

A `cd-prod.yml` workflow (tag-triggered, manual approval gate, semver push to `latest`) was designed and implemented but later removed. AquaIA is a research project with batch workloads and no 24/7 user-facing service; the staging/prod split added complexity without benefit.

The design is documented in `docs/AquaIA-CICD-etat-et-todo.md` for restoration if the project later exposes a public API (Task 2.4).
