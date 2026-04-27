# syntax=docker/dockerfile:1.7
#
# AquaIA runtime image — multi-stage build.
#
# Stage 1 (builder): installs Python deps from requirements-vm.txt into a
#                    user-prefix tree (/install). Keeps build tooling out of
#                    the final image.
# Stage 2 (runtime): copies /install over the same CUDA-enabled PyTorch base,
#                    drops privileges, and exposes the AquaIA CLI as default
#                    command. Datasets, models, results and caches are NEVER
#                    baked in — they are mounted as volumes by docker compose.
#
# Build (locally):
#   docker build \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     -f docker/build/Dockerfile \
#     -t aquaia:dev .
#
# CMD is intentionally `python main.py --help` so that a bare `docker run`
# returns immediately with the CLI usage. Real workloads are launched by
# Compose profiles (idle / train / infer / api) — see deploy/docker-compose.yml.

ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUSERBASE=/install

# Build-time toolchain only. Some wheels (pycocotools on older Python tags,
# scipy under exotic platforms) may need gcc/g++ to compile; we keep them in
# the builder so they never reach the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Layer the requirements first so dependency installs are cached across
# code-only changes. Only requirements-vm.txt is needed at runtime.
COPY requirements-vm.txt ./requirements-vm.txt

# `--user` writes into PYTHONUSERBASE (/install). torch and torchvision come
# from the base image (conda) and are intentionally NOT in requirements-vm.txt
# to avoid duplicating GPU wheels. If a transitive pin tries to upgrade torch,
# fail fast — it would silently break CUDA at runtime.
RUN pip install --user --no-cache-dir -r requirements-vm.txt \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

# Build-time metadata, surfaced as labels and as env vars usable by the app.
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="aquaia" \
      org.opencontainers.image.description="AquaIA — computer vision pipeline for benthic macroinvertebrate identification (YOLO + DINOv3 + SAM)." \
      org.opencontainers.image.source="https://github.com/AkkodisAquaIA/AquaIA" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="Akkodis Research" \
      org.opencontainers.image.licenses="UNLICENSED"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUSERBASE=/install \
    PATH=/install/bin:${PATH} \
    GIT_SHA=${GIT_SHA}

# Minimal runtime libs required by OpenCV / Ultralytics / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID/GID 1000 aligns with the `deploy` user we create on the
# Azure VM, so bind-mounted volumes (datasets/, results/, cache/) are written
# back with the right ownership without chown gymnastics.
RUN groupadd --system --gid 1000 aquaia \
    && useradd  --system --uid 1000 --gid 1000 --create-home --shell /bin/bash aquaia

WORKDIR /app

# Pre-create runtime mountpoints with the right ownership so a fresh
# bind-mount inherits a writable directory.
RUN mkdir -p /app/datasets /app/models /app/results /app/cache \
    && chown -R aquaia:aquaia /app

# Copy the user-prefix install tree from the builder. Keeping ownership at
# 1000:1000 means the runtime user can drop site-packages updates if needed.
COPY --from=builder --chown=aquaia:aquaia /install /install

# Copy application code last to maximise layer cache reuse on dep-only edits.
# Datasets, models, results and caches are NOT copied — they are volumes.
COPY --chown=aquaia:aquaia main.py                ./main.py
COPY --chown=aquaia:aquaia detection              ./detection
COPY --chown=aquaia:aquaia classification         ./classification
COPY --chown=aquaia:aquaia dataloading            ./dataloading
COPY --chown=aquaia:aquaia data_processing        ./data_processing
COPY --chown=aquaia:aquaia data_cleaning          ./data_cleaning
COPY --chown=aquaia:aquaia dataset_utils          ./dataset_utils
COPY --chown=aquaia:aquaia sharepoint_dataloading ./sharepoint_dataloading
COPY --chown=aquaia:aquaia src                    ./src
COPY --chown=aquaia:aquaia IP                     ./IP

USER aquaia

# Healthcheck: GPU must be reachable from inside the container. Only meaningful
# at runtime on the VM (with --gpus all / nvidia runtime). On a CI host without
# GPU this would mark the container unhealthy, which is expected — CI does not
# rely on healthchecks, only on smoke commands.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command surfaces the CLI. Compose profiles override this for
# train / infer / api workloads.
CMD ["python", "main.py", "--help"]
