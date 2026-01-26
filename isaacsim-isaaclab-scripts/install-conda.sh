#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
CONDA_DIR="/opt/conda"
INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
INSTALLER_NAME="Miniforge3-Linux-x86_64.sh"
# ------------------------

log() {
  echo "[conda-install] $1"
}

fail() {
  echo "[conda-install][ERROR] $1" >&2
  exit 1
}

# ---- idempotency check (Ansible-style) ----
if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
  log "Conda already installed at ${CONDA_DIR}, skipping."
  exit 0
fi

log "Conda not found, proceeding with installation."

# ---- download ----
log "Downloading Miniforge installer..."
wget -q -O "${INSTALLER_NAME}" "${INSTALLER_URL}" || fail "Download failed"

# ---- install ----
log "Running installer..."
bash "${INSTALLER_NAME}" -b -p "${CONDA_DIR}" || fail "Installer failed"

# ---- cleanup ----
log "Cleaning up installer..."
rm -f "${INSTALLER_NAME}"

# ---- verification ----
if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  fail "Conda binary not found after installation"
fi

log "Conda successfully installed."
"${CONDA_DIR}/bin/conda" --version