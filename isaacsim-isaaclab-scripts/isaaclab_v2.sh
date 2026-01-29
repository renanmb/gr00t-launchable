#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TARGET_USER="ubuntu"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
WORKDIR="$TARGET_HOME"
ISAACSIM_PATH="$TARGET_HOME/isaacsim"
ISAACLAB_DIR="$WORKDIR/IsaacLab"
CONDA_ENV_NAME="isaaclab"

# -----------------------------------------------------------------------------
# Root guard
# -----------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
  echo "❌ This script must be run as root (sudo)"
  exit 1
fi

if ! id "$TARGET_USER" &>/dev/null; then
  echo "❌ User '$TARGET_USER' does not exist"
  exit 1
fi

if [[ ! -d "$ISAACSIM_PATH" ]]; then
  echo "❌ Isaac Sim not found at $ISAACSIM_PATH"
  exit 1
fi

echo "▶ Installing Isaac Lab for user: $TARGET_USER"

# -----------------------------------------------------------------------------
# System dependencies (idempotent)
# -----------------------------------------------------------------------------
echo "▶ Installing system dependencies"
apt-get update -y
apt-get install -y --no-install-recommends \
  git \
  cmake \
  build-essential

# -----------------------------------------------------------------------------
# Clone Isaac Lab repo (idempotent)
# -----------------------------------------------------------------------------
if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
  echo "▶ Cloning IsaacLab repository"
  sudo -u "$TARGET_USER" git clone \
    https://github.com/isaac-sim/IsaacLab.git \
    "$ISAACLAB_DIR"
else
  echo "▶ IsaacLab repo already exists, skipping clone"
fi

# -----------------------------------------------------------------------------
# Create Isaac Sim symlink (idempotent)
# -----------------------------------------------------------------------------
SYMLINK_PATH="$ISAACLAB_DIR/_isaac_sim"

if [[ -L "$SYMLINK_PATH" ]]; then
  echo "▶ _isaac_sim symlink already exists"
else
  echo "▶ Creating _isaac_sim symlink"
  sudo -u "$TARGET_USER" ln -s "$ISAACSIM_PATH" "$SYMLINK_PATH"
fi

# -----------------------------------------------------------------------------
# Create conda environment (idempotent)
# -----------------------------------------------------------------------------
echo "▶ Ensuring conda environment '$CONDA_ENV_NAME' exists"

sudo -u "$TARGET_USER" bash <<EOF
set -euo pipefail

# Load conda WITHOUT sourcing .bashrc
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | awk '{print \$1}' | grep -qx "isaaclab"; then
  cd "$ISAACLAB_DIR"
  ./isaaclab.sh --conda isaaclab
else
  echo "Conda env 'isaaclab' already exists, skipping"
fi
EOF

# -----------------------------------------------------------------------------
# Install Isaac Lab extensions (editable pip)
# -----------------------------------------------------------------------------
echo "▶ Installing Isaac Lab Python extensions"

sudo -u "$TARGET_USER" bash <<EOF
set -euo pipefail

# Load conda correctly (DO NOT source .bashrc)
source /opt/conda/etc/profile.d/conda.sh

cd "$ISAACLAB_DIR"

# ---------------------------------------------------------------------------
# Ensure conda environment exists
# ---------------------------------------------------------------------------
if ! conda env list | awk '{print \$1}' | grep -qx "isaaclab"; then
  echo "▶ Creating conda environment 'isaaclab'"
  ./isaaclab.sh --conda isaaclab
else
  echo "▶ Conda environment 'isaaclab' already exists"
fi

# ---------------------------------------------------------------------------
# Activate env and install Isaac Lab extensions
# ---------------------------------------------------------------------------
echo "▶ Activating conda environment"
conda activate isaaclab

echo "▶ Installing Isaac Lab dependencies into conda env"
isaaclab -i
EOF

# -----------------------------------------------------------------------------
echo "✅ Isaac Lab installation complete"
echo
echo "👉 To use Isaac Lab:"
echo "   sudo -u ubuntu bash"
echo "   conda activate $CONDA_ENV_NAME"
echo "   cd ~/IsaacLab"
