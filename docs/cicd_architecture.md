# Architecture CI/CD — AquaIA

This document is the reference for the AquaIA CI/CD pipeline. It is paired
with the operational runbook at [`deploy/README.md`](../deploy/README.md):
this file describes **what** the pipeline is and **why** it is shaped this
way; `deploy/README.md` describes **how** to operate it from the VM side.

The companion file [`cicd_etat_des_lieux.md`](./cicd_etat_des_lieux.md)
captured the pre-pipeline state of the repository. It is intentionally kept
for traceability — do not delete it.

> **Note on placeholders.** This document uses placeholders like `<VM_HOST>`,
> `<DEPLOY_USER>`, `<DOCKERHUB_NAMESPACE>`, `<STACK_PATH_STAGING>`, and
> `<STACK_PATH_PROD>` for any value that should not appear in a public
> repository. The actual values are stored in the GitHub Repository
> Variables / Secrets used by the workflows (the structural names, not the
> values, are public) and in the private Akkodis ops wiki.

---

## 1. Overview

Three workflows, one image, two stacks:

```
                            ┌─────────────────────────────────────────────────────┐
                            │                  GitHub Actions                     │
                            │                                                     │
PR  → development           │   ci.yml          (lint · pytest · build no-push)   │
push → development          │   cd-staging.yml  (build → Docker Hub → deploy)     │
git tag v*.*.*              │   cd-prod.yml     (build → Docker Hub → ENV gate    │
                            │                                  → deploy)         │
                            └────────────────────────┬────────────────────────────┘
                                                     │ SSH (<DEPLOY_USER>@<VM_HOST>)
                                                     ▼
                            ┌─────────────────────────────────────────────────────┐
                            │              Azure VM (<VM_HOST>)                   │
                            │  <STACK_PATH_STAGING>   docker compose -p           │
                            │     ├─ docker-compose.yml   aquaia-staging          │
                            │     ├─ .env (IMAGE_TAG=dev-<sha>)                   │
                            │     └─ {datasets, models, results, cache}/          │
                            │  <STACK_PATH_PROD>                                  │
                            │     ├─ docker-compose.yml   aquaia-prod             │
                            │     ├─ .env (IMAGE_TAG=vX.Y.Z)                      │
                            │     └─ {datasets, models, results, cache}/          │
                            └─────────────────────────────────────────────────────┘
```

A single Dockerfile produces a single image, published as
`<DOCKERHUB_NAMESPACE>/aquaia:<tag>` on Docker Hub. The VM only **pulls**;
CI never sends data into the VM beyond the Compose file and a rendered
`.env`.

---

## 2. Triggers and outputs per workflow

| Workflow | Trigger | Builds | Pushes to Docker Hub | Deploys to VM | Approval gate |
|---|---|---|---|---|---|
| `ci.yml` | PR to `development` | yes (load locally) | no | no | none |
| `cd-staging.yml` | push to `development` | yes | yes — `dev-<sha>` + `dev-latest` | `<STACK_PATH_STAGING>` | none |
| `cd-prod.yml` | tag `v*.*.*` | yes | yes — `vX.Y.Z` + `latest` | `<STACK_PATH_PROD>` | GitHub Environment `production` (required reviewer) |

The `latest` floating tag is moved **only** by prod tag builds — staging
moves `dev-latest` instead, so a developer pull of `latest` always means
"the most recent reviewed release".

---

## 3. Required GitHub secrets and variables

Configured under *Settings → Secrets and variables → Actions* on the
`AkkodisAquaIA/AquaIA` repository:

### 3.1 Secrets (encrypted)

| Secret | Used by | Purpose |
|---|---|---|
| `DOCKERHUB_USER` | `cd-staging`, `cd-prod` | Docker Hub username with **write** access to `<DOCKERHUB_NAMESPACE>/aquaia`. |
| `DOCKERHUB_TOKEN` | `cd-staging`, `cd-prod` | Docker Hub PAT with **Read & Write** scope. Rotated periodically. |
| `VM_HOST` | `cd-staging`, `cd-prod` | Azure VM IP or DNS hostname. |
| `VM_SSH_PRIVATE_KEY` | `cd-staging`, `cd-prod` | OpenSSH private key for the `<DEPLOY_USER>` user on the VM. The matching public key sits in `~<DEPLOY_USER>/.ssh/authorized_keys`. |

### 3.2 Repository Variables (not encrypted, not credentials)

| Variable | Used by | Purpose |
|---|---|---|
| `DOCKERHUB_NAMESPACE` | `cd-staging`, `cd-prod` | Docker Hub org or user owning `<NAMESPACE>/aquaia`. Read-only to the public image consumer; not a security boundary, but kept out of the public Git history as a matter of hygiene. |
| `STACK_DIR_STAGING` | `cd-staging` | Absolute path to the staging stack directory on the VM. |
| `STACK_DIR_PROD` | `cd-prod` | Absolute path to the prod stack directory on the VM. |

These three values used to be hardcoded in the workflow `env:` block. They
were lifted out to keep infra-specific values outside the public repo.
Project-internal conventions that have no exploit value (Compose project
names like `aquaia-staging`, the image short name `aquaia`, the Linux user
`deploy` set up by the bootstrap script) **remain hardcoded** in the
workflows and the Compose file because they are part of the public contract
the repo describes.

`GITHUB_TOKEN` is provided automatically and is sufficient for
`contents: read` (the only permission these workflows request).

The VM additionally needs a Docker Hub login as the `<DEPLOY_USER>` user
with a **read-only** PAT (cf. `deploy/README.md` §1.4) — that credential
never leaves the VM and is not a GitHub secret.

---

## 4. Image tagging strategy

| Origin | SHA-pinned tag | Mobile tag | Lifetime |
|---|---|---|---|
| staging push | `dev-<7-char-sha>` | `dev-latest` | retained on Docker Hub indefinitely (used for rollback) |
| prod tag | `<tag-name>` (e.g. `v0.1.0`) | `latest` | retained on Docker Hub indefinitely |
| CI on PR | `aquaia:ci-<full-sha>` (local only) | — | discarded with the runner |

`cd-prod.yml` re-validates the tag at runtime against `^v[0-9]+\.[0-9]+\.[0-9]+$`
even though the trigger glob `v*.*.*` already filters — a pre-release like
`v1.0.0-rc1` would slip past the glob, and we never want such an image
moved to `latest`.

---

## 5. Runbooks

The detailed step-by-step procedures live in
[`deploy/README.md`](../deploy/README.md). This section gives a one-look
cross-reference:

| Scenario | Where it lives |
|---|---|
| Bootstrap a fresh VM | `deploy/README.md` §1 |
| Configure Docker Hub login on VM (read-only PAT) | `deploy/README.md` §1.4 |
| Configure GitHub secrets and the `production` environment | `deploy/README.md` §2 |
| Trigger an automated deploy (push to `development` / tag) | `deploy/README.md` §3.1 |
| Run a one-shot `train` or `infer` job ad hoc | `deploy/README.md` §3.2 |
| **Rollback** to a previous `IMAGE_TAG` | `deploy/README.md` §4.1 |
| **Promotion** staging → prod (tag-driven) | `deploy/README.md` §4.2 |
| **Rotate** the CI/CD SSH key | `deploy/README.md` §4.3 |
| **Rotate** the Docker Hub PAT (CI write + VM read-only) | `deploy/README.md` §4.4 |
| Re-target `development` → `main` | `deploy/README.md` §4.5 + §6 below |

---

## 6. État transitoire `main` / `development`

### Current state (April 2026)

- `development` is the GitHub default branch and is where the up-to-date
  code lives (210 commits, multiple active contributors).
- `main` is obsolete — empty or far behind.
- 0 git tags posted to date.

### Why the workflows still target `development`

The pipeline was built against the actual state of the repository. Pointing
`pull_request.branches` and `push.branches` at `development` makes the CI
useful **today**, on the branch where work is actually merged.

### How to flip later

When `main` is merged up to date and made the default branch, exactly two
files change, with one-line edits:

```diff
# .github/workflows/ci.yml
   on:
     pull_request:
-      branches: [development]
+      branches: [main]

# .github/workflows/cd-staging.yml
   on:
     push:
-      branches: [development]
+      branches: [main]
```

`cd-prod.yml` does **not** change — its trigger is the `v*.*.*` tag
pattern, which is branch-independent.

The branch protection rule (cf. `deploy/README.md` §2.3) must be moved
from `development` to `main` at the same time so the required status
checks (`lint`, `test`, `build`) keep gating PRs.

There is no other coupling: no workflow embeds the branch name in any
output, image tag, deploy path, or compose project name. Nothing on the
VM, in Docker Hub, or in the secrets needs to change.

---

## 7. Future hardening

The pipeline is intentionally simple. Reasonable next moves, none required
for MVP:

- **`VM_SSH_KNOWN_HOSTS` secret** — pin the VM host key instead of TOFU
  via `ssh-keyscan` on every run. One-time provisioning step.
- **Auto-rollback on failed smoke** — wrap the prod deploy step in a
  recovery branch that rewrites `IMAGE_TAG` to the previous value if the
  smoke test fails. Today this is operator-driven.
- **SBOM / vulnerability scan** — add a `trivy` or `grype` step to the
  `build-push` jobs and fail on critical CVEs.
- **Image digest pinning** — have the deploy job consume the
  `${IMAGE_REF}@sha256:<digest>` output of `docker/build-push-action`
  rather than the human-readable tag. Removes any window where Docker
  Hub could serve a different image for the same tag.
- **API service** — flip the `aquaia-api` profile from `__disabled__` to
  `api` once the FastAPI entry point lands (Tâche 2.4 of the AQUA-IA
  scientific dossier). The Compose skeleton is already in place.
