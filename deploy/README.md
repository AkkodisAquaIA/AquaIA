# `deploy/` — VM provisioning & operations runbook

> **Placeholders.** This document uses `<VM_HOST>`, `<ADMIN_USER>`, `<ADMIN_SSH_KEY>`, `<DEPLOY_USER>`, `<DOCKERHUB_NAMESPACE>`, `<STACK_PATH_STAGING>` for values that must not appear in a public repository. Actual values live in the GitHub Secrets / Variables and in the private Akkodis ops wiki.

This directory holds everything the Azure VM needs to run AquaIA:

| File | Purpose |
|---|---|
| [`docker-compose.yml`](./docker-compose.yml) | Compose stack. Copied verbatim by the deploy job on every push to `development`. |
| [`.env.example`](./.env.example) | Template for the per-stack `.env` (`IMAGE_TAG`, `STACK_NAME`, `DOCKERHUB_NAMESPACE`). |
| [`bootstrap-vm.sh`](./bootstrap-vm.sh) | Idempotent provisioning script. Run once on a fresh VM. |

The image is published to Docker Hub as `<DOCKERHUB_NAMESPACE>/aquaia:<tag>`. The VM only **pulls**; CI never sends data beyond the compose file and a rendered `.env`.

---

## 1. Bootstrap a fresh VM (run once)

Target: Ubuntu 24.04 LTS, NVIDIA driver already installed, Docker Engine + `docker-compose-plugin` installed.

### 1.1 Generate the CI/CD SSH key (on your laptop)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/aquaia_ci -C "aquaia-ci@github" -N ""
```

- `~/.ssh/aquaia_ci`     — private key → upload as GitHub secret `VM_SSH_PRIVATE_KEY`
- `~/.ssh/aquaia_ci.pub` — public key → installed on the VM by the bootstrap script

### 1.2 Copy the bootstrap script and public key onto the VM

```bash
scp -i <ADMIN_SSH_KEY> deploy/bootstrap-vm.sh <ADMIN_USER>@<VM_HOST>:/tmp/
scp -i <ADMIN_SSH_KEY> ~/.ssh/aquaia_ci.pub   <ADMIN_USER>@<VM_HOST>:/tmp/
```

### 1.3 Run the bootstrap script as root

```bash
ssh -i <ADMIN_SSH_KEY> <ADMIN_USER>@<VM_HOST>
sudo bash /tmp/bootstrap-vm.sh --ssh-pubkey /tmp/aquaia_ci.pub
# Override stack root if needed (default: /srv/aquaia):
# sudo bash /tmp/bootstrap-vm.sh --ssh-pubkey /tmp/aquaia_ci.pub --stack-root /custom/path
```

The script is idempotent. It will:
1. Create the `deploy` user (UID 1001) and add it to the `docker` group.
2. Create `<STACK_ROOT>/staging/{datasets,models,results,cache}`, owned by `deploy`.
3. Verify Docker Engine and Compose v2 are present.
4. Verify `nvidia-smi` works.
5. Verify the NVIDIA Container Toolkit is registered with Docker; install it if missing.
6. Install the SSH public key into `~deploy/.ssh/authorized_keys`.
7. Print a final summary.

Clean up after: `rm /tmp/bootstrap-vm.sh /tmp/aquaia_ci.pub`.

The `<STACK_ROOT>` must match the GitHub Repository Variable `STACK_DIR_STAGING` (§2.2).

### 1.4 Docker Hub login as the deploy user

The VM only ever **pulls** images. Generate a Docker Hub PAT with **Read-only** scope, then:

```bash
sudo -iu deploy docker login docker.io
# Username: <docker hub user>
# Password: <paste read-only PAT>
```

Credentials are stored in `~deploy/.docker/config.json` (mode `600`).

---

## 2. GitHub setup (one-time)

### 2.1 Repository secrets

*Settings → Secrets and variables → Actions → Secrets*:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username with **write** access to `<DOCKERHUB_NAMESPACE>/aquaia` |
| `DOCKERHUB_TOKEN` | Docker Hub PAT with **Read & Write** scope |
| `VM_HOST` | VM IP or DNS hostname |
| `VM_SSH_PRIVATE_KEY` | Contents of `~/.ssh/aquaia_ci` (private key from §1.1) |

### 2.2 Repository variables

*Settings → Secrets and variables → Actions → Variables*:

| Variable | Value |
|---|---|
| `DOCKERHUB_NAMESPACE` | Docker Hub org or username owning `<NAMESPACE>/aquaia` |
| `STACK_DIR_STAGING` | Absolute path to the staging stack on the VM |

### 2.3 Branch protection on `development`

*Settings → Branches → Add branch ruleset* targeting `development`:
- Require a pull request before merging
- Require status checks to pass: `lint`, `test`, `build & smoke` (from `ci.yml`)
- Require branches to be up to date before merging

---

## 3. Day-to-day operations

### 3.1 Trigger a deploy

Every push (or merged PR) to `development` triggers `cd.yml` automatically:
- Builds and pushes `<NAMESPACE>/aquaia:dev-<sha>` + `dev-latest` to Docker Hub
- SSHes into the VM, writes `.env`, runs `docker compose pull && up -d`

No manual action needed.

### 3.2 Run a one-shot training or inference job

```bash
ssh <DEPLOY_USER>@<VM_HOST>
cd <STACK_PATH_STAGING>
docker compose -p aquaia-staging --profile train up
# or
docker compose -p aquaia-staging --profile infer up
```

Both services have `restart: "no"` and exit on completion.

### 3.3 Exec into the idle container

```bash
docker compose -p aquaia-staging exec aquaia bash
```

### 3.4 Tail logs

```bash
docker compose -p aquaia-staging logs -f --tail 200 aquaia
```

### 3.5 Container runtime model (Compose profiles)

| Profile | Service | Command | Behaviour |
|---|---|---|---|
| `idle` | `aquaia` | `sleep infinity` | Default, restart `unless-stopped`, GPU healthcheck |
| `train` | `aquaia-train` | `python main.py train` | One-shot, `restart: "no"` |
| `infer` | `aquaia-infer` | `python main.py infer` | One-shot, `restart: "no"` |
| `api` | `aquaia-api` | (placeholder) | Disabled — reserved for Task 2.4 |

### 3.6 Dependency layout

| File | When to use |
|---|---|
| `requirements.txt` | Local development |
| `requirements-vm.txt` | VM Docker image (torch comes from the base image, not listed here) |
| `requirements-gpu.txt` | Heavy GPU dev workloads |
| `requirements-dev.txt` | Lint, tests, pre-commit |

---

## 4. Runbooks

### 4.1 Rollback

Every previously published image tag stays on Docker Hub. To roll back:

```bash
ssh <DEPLOY_USER>@<VM_HOST>
cd <STACK_PATH_STAGING>
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=dev-<previous-sha>/' .env
docker compose -p aquaia-staging pull
docker compose -p aquaia-staging --profile idle up -d
```

### 4.2 Pin a stable version on the VM

```bash
ssh <DEPLOY_USER>@<VM_HOST>
cd <STACK_PATH_STAGING>
nano .env   # set IMAGE_TAG=dev-<sha-of-stable-commit>
docker compose -p aquaia-staging pull
docker compose -p aquaia-staging --profile idle up -d
```

Git tags (`v*.*.*`) can be used as scientific reporting milestones; they do not trigger automated deployments. The `dev-<sha>` tags on Docker Hub provide full traceability per commit.

### 4.3 Rotate the CI/CD SSH key

```bash
# On laptop — generate new key
ssh-keygen -t ed25519 -f ~/.ssh/aquaia_ci_new -C "aquaia-ci@github" -N ""

# On the VM — swap authorized key
ssh <DEPLOY_USER>@<VM_HOST>
nano ~/.ssh/authorized_keys   # remove old line, add new one

# On GitHub — update secret
# Settings → Secrets → VM_SSH_PRIVATE_KEY → paste contents of aquaia_ci_new
```

Trigger a no-op deploy to confirm the new key works before deleting the local backup.

### 4.4 Rotate the Docker Hub PAT

1. Generate a new PAT on Docker Hub → *Account Settings → Security → Personal access tokens*.
2. CI: update `DOCKERHUB_TOKEN` in *Settings → Secrets → Actions*. Re-run the latest workflow.
3. VM: `sudo -iu deploy docker login docker.io` and paste the new read-only PAT.
4. Revoke the old PATs on Docker Hub.

### 4.5 Update a Repository Variable

Variables are edited under *Settings → Secrets and variables → Actions → Variables*. Changes apply immediately to subsequent workflow runs.

If you change `STACK_DIR_STAGING`, move the existing stack directory on the VM before the next deploy, otherwise the deploy job will write to a non-existent path.

### 4.6 Re-target workflows from `development` to `main`

When `main` becomes the default branch, two one-line edits are needed:

```diff
# .github/workflows/ci.yml
-    branches: [development]
+    branches: [main]

# .github/workflows/cd.yml
-    branches: [development]
+    branches: [main]
```

Move the branch protection rule from `development` to `main` at the same time.
