#!/usr/bin/env bash
set -euo pipefail

# setup-isaaclab_v6.sh (The Final Bulletproof Edition)

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
# Root guard & Pre-flight checks
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
# --- Escape the /root directory so cron doesn't trap Git ---
cd "$TARGET_HOME"

if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
  echo "▶ Cloning IsaacLab repository"
  sudo -H -u "$TARGET_USER" git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

SYMLINK_PATH="$ISAACLAB_DIR/_isaac_sim"
if [[ ! -L "$SYMLINK_PATH" ]]; then
  echo "▶ Creating _isaac_sim symlink"
  sudo -H -u "$TARGET_USER" ln -s "$ISAACSIM_PATH" "$SYMLINK_PATH"
fi

# -----------------------------------------------------------------------------
# Environment Setup & Extension Installation
# -----------------------------------------------------------------------------
echo "▶ Setting up Conda and installing Isaac Lab extensions"

sudo -H -u "$TARGET_USER" bash <<EOF
set -euo pipefail

# --- Escape the /root directory inside the subshell ---
cd "$TARGET_HOME" 

# 1. Load Conda
source /opt/conda/etc/profile.d/conda.sh

# 2. Enter Repo Directory
cd "$ISAACLAB_DIR"

# 3. Zombie Environment Guard
ENV_HEALTHY=false
if conda env list | awk '{print \$1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "▶ Conda environment '$CONDA_ENV_NAME' exists. Checking health..."
  
  if conda run -n "$CONDA_ENV_NAME" python --version &>/dev/null; then
    echo "▶ Environment is healthy."
    ENV_HEALTHY=true
  else
    echo "▶ Detected broken/incomplete environment! Removing for a clean slate..."
    conda env remove -n "$CONDA_ENV_NAME" -y
  fi
fi

# 4. Create environment if it doesn't exist (FIXED EXTRA ARGUMENT)
if [ "\$ENV_HEALTHY" = false ]; then
  echo "▶ Creating conda environment: $CONDA_ENV_NAME"
  ./isaaclab.sh --conda 
fi

# 5. Activate and Install
echo "▶ Activating environment and running installation"
# Disable 'nounset' temporarily to prevent Conda's internal ZSH_VERSION errors
set +u
conda activate "$CONDA_ENV_NAME"
set -u

# This runs the 'isaaclab' executable found within the repo
echo "▶ Building extensions (This may take 10+ minutes)..."
./isaaclab.sh -i

# 6. Post-Installation Health Check
echo "▶ Running Post-Install Verification..."
if python -c "import torch; print('PyTorch Version:', torch.__version__)" &>/dev/null; then
    echo "✅ Verification Passed: PyTorch is successfully installed."
else
    echo "❌ Verification Failed: Python environment seems broken after installation."
    exit 1
fi

# 7. Cleanup (Optional: Frees up gigabytes of cached tarballs)
echo "▶ Cleaning up Conda package cache to save disk space..."
conda clean --all -y > /dev/null

EOF

# -----------------------------------------------------------------------------
echo "✅ Isaac Lab installation complete"
echo
echo "👉 To use Isaac Lab:"
echo "   sudo -u $TARGET_USER -i"
echo "   conda activate $CONDA_ENV_NAME"
echo "   cd ~/IsaacLab"