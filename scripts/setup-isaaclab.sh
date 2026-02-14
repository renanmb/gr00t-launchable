#!/usr/bin/env bash
set -euo pipefail

# isaaclab_v3.sh

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
# System dependencies
# -----------------------------------------------------------------------------
echo "▶ Installing system dependencies"
apt-get update -y
apt-get install -y --no-install-recommends git cmake build-essential

# -----------------------------------------------------------------------------
# Repo & Symlink setup
# -----------------------------------------------------------------------------
if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
  echo "▶ Cloning IsaacLab repository"
  sudo -u "$TARGET_USER" git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

SYMLINK_PATH="$ISAACLAB_DIR/_isaac_sim"
if [[ ! -L "$SYMLINK_PATH" ]]; then
  echo "▶ Creating _isaac_sim symlink"
  sudo -u "$TARGET_USER" ln -s "$ISAACSIM_PATH" "$SYMLINK_PATH"
fi

# -----------------------------------------------------------------------------
# Environment Setup & Extension Installation
# -----------------------------------------------------------------------------
echo "▶ Setting up Conda and installing Isaac Lab extensions"

sudo -u "$TARGET_USER" bash <<EOF
set -euo pipefail

# 1. Load Conda
source /opt/conda/etc/profile.d/conda.sh

# 2. Enter Repo Directory
cd "$ISAACLAB_DIR"

# 3. Create environment if it doesn't exist
if ! conda env list | awk '{print \$1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "▶ Creating conda environment: $CONDA_ENV_NAME"
  ./isaaclab.sh --conda "$CONDA_ENV_NAME"
else
  echo "▶ Conda environment '$CONDA_ENV_NAME' already exists"
fi

# 4. Activate and Install
echo "▶ Activating environment and running installation"
# Disable 'nounset' temporarily to prevent ZSH_VERSION errors
set +u
conda activate "$CONDA_ENV_NAME"
set -u

# This runs the 'isaaclab' executable found within the repo
./isaaclab.sh -i
EOF

# -----------------------------------------------------------------------------
echo "✅ Isaac Lab installation complete"
echo
echo "👉 To use Isaac Lab:"
echo "   sudo -u $TARGET_USER -i"
echo "   conda activate $CONDA_ENV_NAME"
echo "   cd ~/IsaacLab"