#!/usr/bin/env bash
#
# deploy/bootstrap-vm.sh
#
# One-shot, idempotent provisioning script for the AquaIA Azure VM.
# Re-running it on an already-provisioned VM is safe: every action checks the
# current state before mutating anything.
#
# Tested target: Ubuntu 24.04 LTS, NVIDIA driver already installed.
#
# What it does:
#   1. Refuses to run as a non-root user.
#   2. Creates the dedicated `deploy` user (UID 1000) and adds it to `docker`.
#   3. Creates <stack-root>/{staging,prod}/{datasets,models,results,cache}
#      owned by deploy:deploy. Default <stack-root> is /srv/aquaia and can be
#      overridden via --stack-root.
#   4. Verifies that Docker and Docker Compose v2 are installed.
#   5. Verifies that the NVIDIA driver is reachable (`nvidia-smi`).
#   6. Verifies that the NVIDIA Container Toolkit is configured for Docker;
#      installs and configures it if missing.
#   7. Installs the provided SSH public key into ~deploy/.ssh/authorized_keys.
#   8. Prints a clear final summary.
#
# Usage:
#   sudo ./bootstrap-vm.sh --ssh-pubkey /path/to/aquaia_ci.pub [--stack-root /custom/path]
#
# Whatever path you pick must match the GitHub Repository Variables
# STACK_DIR_STAGING / STACK_DIR_PROD that the workflows use to reach the
# stacks. The default (/srv/aquaia) is the convention used by the AquaIA
# Akkodis ops setup; pick a different path only if you mirror the deployment
# elsewhere.
#
# All steps are logged with a timestamp. Any failure aborts immediately
# (set -euo pipefail).

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly DEPLOY_USER="deploy"
readonly DEPLOY_UID="1000"
readonly DEPLOY_GID="1000"
readonly DEFAULT_STACK_ROOT="/srv/aquaia"
readonly STACKS=("staging" "prod")
readonly STACK_SUBDIRS=("datasets" "models" "results" "cache")

# Mutable: overridable by --stack-root.
STACK_ROOT="${DEFAULT_STACK_ROOT}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()   { printf '%s [%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" "$2"; }
info()  { log "INFO " "$*"; }
warn()  { log "WARN " "$*" >&2; }
err()   { log "ERROR" "$*" >&2; }
die()   { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight: must be root, must have an SSH pubkey arg
# ---------------------------------------------------------------------------
require_root() {
	if [[ "${EUID}" -ne 0 ]]; then
		die "This script must run as root. Try: sudo $0 $*"
	fi
}

usage() {
	cat <<-EOF >&2
		Usage: sudo $0 --ssh-pubkey <path-to-public-key> [--stack-root <path>]

		Options:
		  --ssh-pubkey <file>   Path to the SSH public key the CI/CD pipeline will use
		                        to log in as 'deploy'. Required.
		  --stack-root <path>   Root directory under which staging/ and prod/ stack
		                        trees are created. Default: ${DEFAULT_STACK_ROOT}.
		                        MUST match the GitHub Repository Variables
		                        STACK_DIR_STAGING / STACK_DIR_PROD used by the
		                        workflows.
		  -h, --help            Show this help and exit.
	EOF
	exit 2
}

SSH_PUBKEY_PATH=""

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--ssh-pubkey)
				[[ $# -ge 2 ]] || die "--ssh-pubkey requires an argument"
				SSH_PUBKEY_PATH="$2"
				shift 2
				;;
			--stack-root)
				[[ $# -ge 2 ]] || die "--stack-root requires an argument"
				# Strip a trailing slash for consistency with the rest of the
				# script (we always join with /).
				STACK_ROOT="${2%/}"
				[[ "${STACK_ROOT}" = /* ]] || die "--stack-root must be an absolute path, got: $2"
				shift 2
				;;
			-h|--help)
				usage
				;;
			*)
				err "Unknown argument: $1"
				usage
				;;
		esac
	done

	[[ -n "${SSH_PUBKEY_PATH}" ]] || { err "--ssh-pubkey is required"; usage; }
	[[ -f "${SSH_PUBKEY_PATH}" ]] || die "SSH pubkey file not found: ${SSH_PUBKEY_PATH}"
}

# ---------------------------------------------------------------------------
# Step 3 — deploy user
# ---------------------------------------------------------------------------
ensure_deploy_user() {
	info "Ensuring user '${DEPLOY_USER}' (uid=${DEPLOY_UID}) exists"
	if id -u "${DEPLOY_USER}" >/dev/null 2>&1; then
		info "  user '${DEPLOY_USER}' already exists, skipping creation"
	else
		# Create matching group first so UID/GID stay aligned with the
		# 'aquaia' user inside the container (UID 1000) — keeps bind-mount
		# permissions sane.
		if ! getent group "${DEPLOY_USER}" >/dev/null 2>&1; then
			groupadd --gid "${DEPLOY_GID}" "${DEPLOY_USER}"
			info "  created group ${DEPLOY_USER} (gid=${DEPLOY_GID})"
		fi
		useradd \
			--create-home \
			--shell /bin/bash \
			--uid "${DEPLOY_UID}" \
			--gid "${DEPLOY_GID}" \
			"${DEPLOY_USER}"
		info "  created user ${DEPLOY_USER} (uid=${DEPLOY_UID})"
	fi

	if ! getent group docker >/dev/null 2>&1; then
		warn "  'docker' group does not exist yet. Docker may not be installed; will check next."
	elif id -nG "${DEPLOY_USER}" | tr ' ' '\n' | grep -qx docker; then
		info "  ${DEPLOY_USER} already in 'docker' group"
	else
		usermod -aG docker "${DEPLOY_USER}"
		info "  added ${DEPLOY_USER} to 'docker' group"
	fi
}

# ---------------------------------------------------------------------------
# Step 4 — stack directory tree
# ---------------------------------------------------------------------------
ensure_stack_tree() {
	info "Ensuring stack directories under ${STACK_ROOT}"
	mkdir -p "${STACK_ROOT}"
	for stack in "${STACKS[@]}"; do
		for sub in "${STACK_SUBDIRS[@]}"; do
			local d="${STACK_ROOT}/${stack}/${sub}"
			mkdir -p "${d}"
			info "  ensured ${d}"
		done
	done
	chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${STACK_ROOT}"
	chmod 750 "${STACK_ROOT}"
	info "  ownership set to ${DEPLOY_USER}:${DEPLOY_USER}"
}

# ---------------------------------------------------------------------------
# Step 5 — Docker & Compose v2 presence
# ---------------------------------------------------------------------------
verify_docker() {
	info "Verifying Docker and Compose v2"
	command -v docker >/dev/null 2>&1 || die "docker is not installed. Install Docker Engine first (https://docs.docker.com/engine/install/ubuntu/)."
	# Compose v2 is a CLI plugin. v1 (docker-compose) is unsupported here.
	if ! docker compose version >/dev/null 2>&1; then
		die "docker compose v2 is not available. Install the docker-compose-plugin package."
	fi
	info "  docker:   $(docker --version)"
	info "  compose:  $(docker compose version | head -1)"
}

# ---------------------------------------------------------------------------
# Step 6 — NVIDIA driver presence
# ---------------------------------------------------------------------------
verify_nvidia_driver() {
	info "Verifying NVIDIA driver"
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		die "nvidia-smi not found. Install the NVIDIA driver manually before re-running this script."
	fi
	if ! nvidia-smi >/dev/null 2>&1; then
		die "nvidia-smi present but failing. The driver/runtime is broken; investigate before re-running."
	fi
	info "  nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}

# ---------------------------------------------------------------------------
# Step 7 — NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
toolkit_already_configured() {
	# Two cheap signals: the binary is on PATH AND the docker daemon knows about
	# the nvidia runtime. Both must be true; either alone is insufficient.
	command -v nvidia-ctk >/dev/null 2>&1 || return 1
	docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

install_nvidia_container_toolkit() {
	info "Installing NVIDIA Container Toolkit (Ubuntu)"
	# Official NVIDIA instructions for Ubuntu, kept self-contained so this
	# script does not depend on a separate doc.
	export DEBIAN_FRONTEND=noninteractive

	# 1. GPG key (idempotent — overwriting is fine).
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
		| gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

	# 2. Apt source.
	curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
		| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
		> /etc/apt/sources.list.d/nvidia-container-toolkit.list

	# 3. Install + configure Docker runtime.
	apt-get update -yq
	apt-get install -yq nvidia-container-toolkit
	nvidia-ctk runtime configure --runtime=docker
	systemctl restart docker
	info "  toolkit installed and Docker daemon restarted"
}

ensure_nvidia_toolkit() {
	info "Verifying NVIDIA Container Toolkit"
	if toolkit_already_configured; then
		info "  toolkit already present and registered with Docker, skipping install"
		return 0
	fi
	warn "  toolkit missing or not registered — installing"
	install_nvidia_container_toolkit
	if ! toolkit_already_configured; then
		die "Toolkit install completed but Docker still does not advertise the nvidia runtime. Check 'docker info' and the journal."
	fi
	info "  toolkit verified after install"
}

# ---------------------------------------------------------------------------
# Step 8 — SSH pubkey
# ---------------------------------------------------------------------------
install_ssh_pubkey() {
	info "Installing CI/CD SSH public key for ${DEPLOY_USER}"
	local pubkey_content
	pubkey_content="$(cat "${SSH_PUBKEY_PATH}")"

	# Sanity check: must look like an OpenSSH public key.
	if ! printf '%s' "${pubkey_content}" | head -1 \
		| grep -qE '^(ssh-rsa|ssh-ed25519|ecdsa-sha2-nistp[0-9]+|sk-ssh-ed25519@openssh.com|sk-ecdsa-sha2-nistp[0-9]+@openssh.com) '; then
		die "File ${SSH_PUBKEY_PATH} does not look like a valid SSH public key (first line must start with ssh-rsa / ssh-ed25519 / ecdsa-sha2-...)."
	fi

	local home_dir ssh_dir auth_keys
	home_dir="$(getent passwd "${DEPLOY_USER}" | cut -d: -f6)"
	ssh_dir="${home_dir}/.ssh"
	auth_keys="${ssh_dir}/authorized_keys"

	install -d -m 700 -o "${DEPLOY_USER}" -g "${DEPLOY_USER}" "${ssh_dir}"
	touch "${auth_keys}"
	chown "${DEPLOY_USER}:${DEPLOY_USER}" "${auth_keys}"
	chmod 600 "${auth_keys}"

	if grep -qxF "${pubkey_content}" "${auth_keys}"; then
		info "  pubkey already present in ${auth_keys}, skipping"
	else
		printf '%s\n' "${pubkey_content}" >> "${auth_keys}"
		info "  pubkey appended to ${auth_keys}"
	fi
}

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print_summary() {
	cat <<-EOF

	==========================================================================
	  AquaIA VM bootstrap — DONE
	==========================================================================
	  user           : ${DEPLOY_USER} (uid=${DEPLOY_UID}), member of: $(id -nG "${DEPLOY_USER}" | tr ' ' ',')
	  stack root     : ${STACK_ROOT}
	  staging tree   : ${STACK_ROOT}/staging/{datasets,models,results,cache}
	  prod tree      : ${STACK_ROOT}/prod/{datasets,models,results,cache}
	  docker         : $(docker --version 2>/dev/null || echo 'NOT FOUND')
	  compose        : $(docker compose version 2>/dev/null | head -1 || echo 'NOT FOUND')
	  nvidia driver  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NOT FOUND')
	  nvidia runtime : $(docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -o '"nvidia"' | head -1 || echo 'NOT REGISTERED')
	  ssh authorized : ${SSH_PUBKEY_PATH##*/} -> /home/${DEPLOY_USER}/.ssh/authorized_keys

	  Next steps:
	    1. As ${DEPLOY_USER}, login to Docker Hub with a READ-ONLY PAT:
	         sudo -iu ${DEPLOY_USER} docker login docker.io
	    2. Drop deploy/.env files in ${STACK_ROOT}/{staging,prod}/ (cf. deploy/.env.example).
	    3. Trigger a deploy from CI: push to development → staging stack auto-deploys.
	==========================================================================
	EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
	# parse_args first so `--help` works without sudo and does not surprise
	# operators with a misleading "must run as root" message.
	parse_args "$@"
	require_root "$@"

	info "Starting AquaIA VM bootstrap"
	ensure_deploy_user
	ensure_stack_tree
	verify_docker
	verify_nvidia_driver
	ensure_nvidia_toolkit
	install_ssh_pubkey
	# `usermod -aG docker` from ensure_deploy_user runs *before* docker is
	# verified. If the docker group did not exist then, retry once now that
	# verify_docker has confirmed docker is installed.
	if ! id -nG "${DEPLOY_USER}" | tr ' ' '\n' | grep -qx docker; then
		usermod -aG docker "${DEPLOY_USER}"
		info "Retro-added ${DEPLOY_USER} to docker group (docker came in mid-script)"
	fi

	print_summary
}

main "$@"
