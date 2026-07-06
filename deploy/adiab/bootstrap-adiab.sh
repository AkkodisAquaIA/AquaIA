#!/usr/bin/env bash
#
# deploy/adiab/bootstrap-adiab.sh
#
# One-shot, idempotent provisioning script for the ADIAB VM.
# Tested target: Ubuntu 24.04 LTS — 4 vCPU, 16 GB RAM, no GPU.
#
# What it does:
#   1. Refuses to run as a non-root user.
#   2. Installs Docker Engine + Docker Compose plugin (if missing).
#   3. Installs git (if missing).
#   4. Creates the stack directory (/opt/aquaia-dataset-builder) and storage/.
#   5. Adds ubuntu user to the docker group so it can run docker without sudo.
#   6. Prints a final summary with the next manual steps.
#
# Usage:
#   sudo bash bootstrap-adiab.sh
#
# Run time: ~2-3 min on first install (Docker packages), <5s on re-run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly STACK_DIR="/opt/aquaia-dataset-builder"
readonly APP_USER="ubuntu"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { printf '%s [%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" "$2"; }
info() { log "INFO " "$*"; }
warn() { log "WARN " "$*" >&2; }
die()  { log "ERROR" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
require_root() {
    [[ "${EUID}" -eq 0 ]] || die "This script must run as root. Try: sudo $0"
}

# ---------------------------------------------------------------------------
# Step 1 — Docker Engine
# ---------------------------------------------------------------------------
install_docker() {
    info "Checking Docker Engine"

    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        info "  Docker $(docker --version) already present, skipping install"
        return 0
    fi

    info "  Docker not found — installing via apt"
    export DEBIAN_FRONTEND=noninteractive

    apt-get update -yq
    apt-get install -yq ca-certificates curl gnupg lsb-release

    # Official Docker GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Docker apt repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update -yq
    apt-get install -yq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin

    systemctl enable --now docker
    info "  Docker installed: $(docker --version)"
    info "  Compose installed: $(docker compose version | head -1)"
}

# ---------------------------------------------------------------------------
# Step 2 — git
# ---------------------------------------------------------------------------
install_git() {
    info "Checking git"
    if command -v git >/dev/null 2>&1; then
        info "  git $(git --version) already present"
    else
        info "  git not found — installing"
        apt-get install -yq git
        info "  git installed: $(git --version)"
    fi
}

# ---------------------------------------------------------------------------
# Step 3 — Stack directory + storage
# ---------------------------------------------------------------------------
ensure_stack_dir() {
    info "Ensuring stack directory ${STACK_DIR}"
    mkdir -p "${STACK_DIR}/storage"
    chown -R "${APP_USER}:${APP_USER}" "${STACK_DIR}"
    info "  ${STACK_DIR}/storage/ created and owned by ${APP_USER}"
}

# ---------------------------------------------------------------------------
# Step 4 — Add ubuntu user to docker group
# ---------------------------------------------------------------------------
ensure_docker_group() {
    info "Ensuring ${APP_USER} is in the docker group"
    if id -nG "${APP_USER}" | tr ' ' '\n' | grep -qx docker; then
        info "  ${APP_USER} already in docker group"
    else
        usermod -aG docker "${APP_USER}"
        info "  ${APP_USER} added to docker group (re-login required to take effect)"
    fi
}

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print_summary() {
    cat <<-EOF

==========================================================================
  ADIAB VM bootstrap — DONE
==========================================================================
  Docker       : $(docker --version 2>/dev/null || echo 'NOT FOUND')
  Compose      : $(docker compose version 2>/dev/null | head -1 || echo 'NOT FOUND')
  git          : $(git --version 2>/dev/null || echo 'NOT FOUND')
  stack dir    : ${STACK_DIR}/
  storage dir  : ${STACK_DIR}/storage/
  app user     : ${APP_USER} (in docker group: $(id -nG "${APP_USER}" | tr ' ' '\n' | grep -c docker | grep -q 1 && echo YES || echo 'NO — re-login needed'))

  ⚠  If ${APP_USER} was just added to the docker group, you MUST log out and back in
     (or run: newgrp docker) before running docker commands without sudo.

  Next steps — run as ubuntu user:
    1. Clone the repository:
         git clone <REPO_URL> /opt/aquaia-dataset-builder
         cd /opt/aquaia-dataset-builder/aquaia-dataset-builder

    2. Copy the existing database (if migrating from another server):
         scp -i key-adiab.pem /path/to/adiab.db ubuntu@172.30.24.4:/opt/aquaia-dataset-builder/storage/

    3. Build and start the stack:
         cd /opt/aquaia-dataset-builder/aquaia-dataset-builder
         docker compose -f docker-compose.prod.yml up -d --build

    4. Verify:
         docker compose -f docker-compose.prod.yml ps
         curl http://localhost/health

    App is accessible at: http://172.30.24.4
==========================================================================
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    require_root
    info "Starting ADIAB VM bootstrap (Ubuntu 24.04, no GPU)"
    install_docker
    install_git
    ensure_stack_dir
    ensure_docker_group
    print_summary
}

main "$@"
