# `deploy/` — VM provisioning & operations runbook

> **Note on placeholders.** This document uses placeholders like `<VM_HOST>`,
> `<DEPLOY_USER>`, `<DOCKERHUB_NAMESPACE>`, `<STACK_PATH_STAGING>`, and
> `<STACK_PATH_PROD>` for any value that should not appear in a public
> repository. The actual values are documented in the private Akkodis ops
> wiki and accessible to authorized consortium members only. Substitute the
> placeholders with the real values when running the commands locally on the
> VM.

This directory holds everything the Azure VM needs to run AquaIA:

| File | Purpose |
|---|---|
| [`docker-compose.yml`](./docker-compose.yml) | Compose stack used by both `staging` and `prod`. Copied verbatim by the deploy job. |
| [`.env.example`](./.env.example) | Template for the per-stack `.env` file (`IMAGE_TAG`, `STACK_NAME`, `DOCKERHUB_NAMESPACE`). |
| [`bootstrap-vm.sh`](./bootstrap-vm.sh) | Idempotent provisioning script. Run once on a fresh VM. |
| `README.md` | This file. |

The image is published to **Docker Hub** as `<DOCKERHUB_NAMESPACE>/aquaia:<tag>`.
The VM pulls; the CI never SSHes data into the VM beyond the compose file and `.env`.

---

## 1. Bootstrap a fresh VM (run once)

Target: Ubuntu 24.04 LTS, NVIDIA driver already installed, Docker Engine + `docker-compose-plugin` already installed.

### 1.1 Generate the CI/CD SSH key (on your laptop)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/aquaia_ci -C "aquaia-ci@github" -N ""
```

You now have:
- `~/.ssh/aquaia_ci`     — **private** key, will be uploaded as the GitHub secret `VM_SSH_PRIVATE_KEY`
- `~/.ssh/aquaia_ci.pub` — public key, copied onto the VM by the bootstrap script

### 1.2 Copy the bootstrap script onto the VM

```bash
scp -i <ADMIN_SSH_KEY> deploy/bootstrap-vm.sh <ADMIN_USER>@<VM_HOST>:/tmp/
scp -i <ADMIN_SSH_KEY> ~/.ssh/aquaia_ci.pub  <ADMIN_USER>@<VM_HOST>:/tmp/
```

### 1.3 Run the bootstrap script as root

```bash
ssh -i <ADMIN_SSH_KEY> <ADMIN_USER>@<VM_HOST>
sudo bash /tmp/bootstrap-vm.sh --ssh-pubkey /tmp/aquaia_ci.pub
# Override the stack root if you do NOT use the default /srv/aquaia:
# sudo bash /tmp/bootstrap-vm.sh --ssh-pubkey /tmp/aquaia_ci.pub --stack-root /custom/path
```

The script is idempotent: re-running it on an already-provisioned VM is safe and only fixes drift. It will:

1. Create the dedicated `<DEPLOY_USER>` user (UID 1000) and add it to the `docker` group.
2. Create `<STACK_ROOT>/{staging,prod}/{datasets,models,results,cache}`, owned by the deploy user.
3. Verify Docker Engine and Compose v2 are present (refuses to continue otherwise).
4. Verify `nvidia-smi` works (refuses to install NVIDIA driver — that is a manual step).
5. Verify the **NVIDIA Container Toolkit** is registered with Docker; install and configure it if missing (Ubuntu repository, GPG key, `nvidia-ctk runtime configure --runtime=docker`, `systemctl restart docker`).
6. Install the SSH public key into `~<DEPLOY_USER>/.ssh/authorized_keys`.
7. Print a final summary of the VM state.

Once done, delete the temporary copy: `rm /tmp/bootstrap-vm.sh /tmp/aquaia_ci.pub`.

The `<STACK_ROOT>` you pick at this step **must match** the GitHub Repository Variables `STACK_DIR_STAGING` / `STACK_DIR_PROD` configured in §2.2 — otherwise the deploy jobs will write to a path that does not exist on the VM.

### 1.4 Docker Hub login as the deploy user (read-only PAT)

The VM only ever **pulls** images. Generate a Docker Hub Personal Access Token with **Read-only** scope on Docker Hub → *Account Settings* → *Security* → *Personal access tokens*, then:

```bash
sudo -iu <DEPLOY_USER> docker login docker.io
# Username: <docker hub user>
# Password: <paste read-only PAT>
```

Credentials end up in `~<DEPLOY_USER>/.docker/config.json` (mode `600`). Subsequent `docker compose pull` runs by that user are authenticated without further interaction.

### 1.5 Drop a `.env` for each stack

```bash
sudo -iu <DEPLOY_USER> bash <<'EOF'
mkdir -p <STACK_PATH_STAGING> <STACK_PATH_PROD>
cp /tmp/.env.example <STACK_PATH_STAGING>/.env  # adjust IMAGE_TAG and STACK_NAME=staging
cp /tmp/.env.example <STACK_PATH_PROD>/.env     # adjust IMAGE_TAG and STACK_NAME=prod
EOF
```

The CI rewrites `IMAGE_TAG` on every deploy; `STACK_NAME` and `DOCKERHUB_NAMESPACE` are set once.

---

## 2. GitHub setup (one-time)

### 2.1 Required repository secrets

Go to *Settings* → *Secrets and variables* → *Actions* → *Secrets* → *New repository secret*:

| Secret | Value |
|---|---|
| `DOCKERHUB_USER` | Docker Hub user with **write** access to `<DOCKERHUB_NAMESPACE>/aquaia` |
| `DOCKERHUB_TOKEN` | Docker Hub PAT with **Read & Write** scope (CI publishes images) |
| `VM_HOST` | The VM's IP or DNS hostname |
| `VM_SSH_PRIVATE_KEY` | Contents of `~/.ssh/aquaia_ci` (the **private** key from step 1.1) |

`VM_USER` is hardcoded to `deploy` in the workflows; no secret needed.

### 2.2 Required repository variables

Go to *Settings* → *Secrets and variables* → *Actions* → *Variables* → *New repository variable*. **These must be created before the first CI/CD run, otherwise the workflows will fail.**

| Variable | Value |
|---|---|
| `DOCKERHUB_NAMESPACE` | The Docker Hub org or user name owning `<NAMESPACE>/aquaia` |
| `STACK_DIR_STAGING` | Absolute path to the staging stack on the VM (e.g. `<STACK_PATH_STAGING>`) |
| `STACK_DIR_PROD` | Absolute path to the prod stack on the VM (e.g. `<STACK_PATH_PROD>`) |

These are repository **Variables** rather than Secrets because they are not credentials and are referenced by name in the workflows. They live outside the public Git history while remaining auditable to repo admins.

### 2.3 GitHub Environment for production

Production deploys are gated on a manual approval. Set this up once:

1. *Settings* → *Environments* → *New environment* → name it **`production`**.
2. Enable *Required reviewers* and add at least one team member.
3. (Optional) Add a *Wait timer* (e.g. 5 minutes) for an extra audit window.
4. The `cd-prod.yml` workflow declares `environment: production`, so every prod deploy triggered by a `v*.*.*` tag will wait on a reviewer's "Approve and deploy" click before SSHing into the VM.

### 2.4 Branch protection on `development`

*Settings* → *Branches* → *Add branch ruleset* targeting `development`:
- *Require a pull request before merging*
- *Require status checks to pass*: select `lint`, `test`, `build` (the three jobs from `ci.yml`)
- *Require branches to be up to date before merging*
- *Do not allow bypassing the above settings*

This ensures no green-on-trust merges to the deployable branch.

---

## 3. Day-to-day operations

### 3.1 Trigger a deploy (automated)

- **Staging**: push (or merge a PR) to `development`. `cd-staging.yml` runs, builds, pushes `dev-<sha>` to Docker Hub, copies `docker-compose.yml`, rewrites `.env`, runs `docker compose pull && up -d --remove-orphans`.
- **Production**: from a clean `development`, tag and push:
  ```bash
  git checkout development && git pull
  git tag -a v0.1.0 -m "AquaIA 0.1.0 — Tâche 2.4 alpha"
  git push origin v0.1.0
  ```
  `cd-prod.yml` runs, waits for the `production` environment approval, then deploys to the prod stack directory.

### 3.2 Run a one-shot training or inference job

SSH in as the deploy user (or `sudo -iu <DEPLOY_USER>` from your usual login):

```bash
cd <STACK_PATH_STAGING>
docker compose -p aquaia-staging --profile train up
# or
docker compose -p aquaia-staging --profile infer up
```

Both services have `restart: "no"` and exit when the run finishes. Logs are visible live; rerun with `--detach` if you want to background.

### 3.3 Inspect / `exec` into the idle container

```bash
docker compose -p aquaia-staging exec aquaia bash
# or run an ad-hoc Python command:
docker compose -p aquaia-staging exec aquaia python main.py infer --config detection/infer_config.yaml
```

### 3.4 Tail logs

```bash
docker compose -p aquaia-staging logs -f --tail 200 aquaia
```

---

## 4. Runbooks

### 4.1 Rollback

The previous image tag is still in Docker Hub (deploy jobs prune local images, never remote). To roll back:

```bash
ssh <DEPLOY_USER>@<VM_HOST>
cd <STACK_PATH_STAGING>        # or <STACK_PATH_PROD>
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=dev-<previous-sha>/' .env
docker compose -p aquaia-staging pull
docker compose -p aquaia-staging --profile idle up -d
```

For prod, replace `dev-<previous-sha>` with the previous semver tag (`v0.0.9` etc.) and use `-p aquaia-prod`. Deploy jobs do **not** auto-rollback on failure (yet — could be added once smoke tests are richer); operator-driven rollback is the current model.

### 4.2 Promotion staging → prod

Promotion is intentionally **explicit**, never automatic. Process:

1. Confirm the staging deploy is healthy: `docker compose -p aquaia-staging ps` shows `(healthy)`, smoke commands work.
2. Locally, on a clean `development` branch:
   ```bash
   git checkout development
   git pull --ff-only
   git tag -a vMAJOR.MINOR.PATCH -m "<one-line release note>"
   git push origin vMAJOR.MINOR.PATCH
   ```
3. The `cd-prod.yml` workflow triggers on the tag push, builds the image with that exact tag, and waits at the `production` environment gate.
4. A reviewer approves in GitHub UI → CI deploys to the prod stack directory.
5. Verify on the VM: `docker compose -p aquaia-prod ps`, smoke command.

The semver tag becomes a stable reference for the scientific reporting (Tâche 2.4 deliverables).

### 4.3 Rotate the CI/CD SSH key

```bash
# On laptop
ssh-keygen -t ed25519 -f ~/.ssh/aquaia_ci_new -C "aquaia-ci@github" -N ""

# On the VM, append the new pubkey, remove the old one
ssh <DEPLOY_USER>@<VM_HOST>
nano ~/.ssh/authorized_keys     # delete the old line, add the new one

# In GitHub
# Settings → Secrets → update VM_SSH_PRIVATE_KEY with contents of ~/.ssh/aquaia_ci_new
```

Then trigger a no-op deploy (re-run a workflow) to confirm the new key works before deleting the local backup.

### 4.4 Rotate the Docker Hub PAT

1. Generate a new PAT in Docker Hub → *Account Settings* → *Security* → *Personal access tokens* (write scope for CI; read-only for VM).
2. CI: update `DOCKERHUB_TOKEN` in *Settings* → *Secrets* → *Actions*. Re-run the latest workflow to confirm it works.
3. VM: `sudo -iu <DEPLOY_USER> docker login docker.io` and paste the new read-only PAT.
4. Revoke the old PATs in Docker Hub.

### 4.5 Update a Repository Variable (e.g. moving the stack root)

Repository Variables are edited under *Settings* → *Secrets and variables* → *Actions* → *Variables*. Changes apply immediately to subsequent workflow runs — there is no propagation delay.

If you change `STACK_DIR_STAGING` or `STACK_DIR_PROD`, you must also move the existing stack directory on the VM (compose file, `.env`, volumes) before the next deploy, otherwise the deploy job will scp into a non-existent path and fail.

### 4.6 Future: re-targeting workflows from `development` to `main`

When `main` becomes the up-to-date branch, change two lines:

```diff
# .github/workflows/ci.yml
-    branches: [development]
+    branches: [main]

# .github/workflows/cd-staging.yml
-    branches: [development]
+    branches: [main]
```

`cd-prod.yml` does **not** change — its trigger is the `v*.*.*` tag pattern, which is branch-independent.

Re-target the branch protection rule from `development` to `main` at the same time.
