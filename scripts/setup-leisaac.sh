#!/usr/bin/env bash
set -euo pipefail

############################
# CONFIG
############################
ANSIBLE_USER="ubuntu"
HOME_DIR="/home/${ANSIBLE_USER}"
CONDA_ENV_NAME="leisaac"
# LEISAAC_VERSION="v0.3.0"
LEISAAC_VERSION="dataset-handler"

log() { echo -e "\n>>> $*\n"; }

############################
# PRE-REQUISITES (APT)
############################
log "Installing system dependencies"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    curl \
    cmake \
    build-essential

############################
# ISAACLAB CONDA SETUP
############################
# This assumes IsaacLab is already cloned in the home directory
log "Initializing IsaacLab Conda environment: ${CONDA_ENV_NAME}"

if [ -d "$HOME_DIR/IsaacLab" ]; then
    # Run the IsaacLab conda setup script as the user
    sudo -u "$ANSIBLE_USER" -i bash -c "cd $HOME_DIR/IsaacLab && ./isaaclab.sh --conda $CONDA_ENV_NAME"
else
    log "ERROR: IsaacLab directory not found at $HOME_DIR/IsaacLab. Please clone it first."
    exit 1
fi

############################
# LEISAAC SETUP --- This uses the LightwheelAI repo
############################
# log "Cloning LeIsaac (${LEISAAC_VERSION})"
# if [ ! -d "$HOME_DIR/leisaac" ]; then
#     sudo -u "$ANSIBLE_USER" git clone --recursive https://github.com/LightwheelAI/leisaac.git "$HOME_DIR/leisaac"
#     cd "$HOME_DIR/leisaac"
#     sudo -u "$ANSIBLE_USER" git checkout "$LEISAAC_VERSION"
# fi

############################
# LEISAAC SETUP --- This uses custom repo
############################
log "Cloning LeIsaac (${LEISAAC_VERSION})"
if [ ! -d "$HOME_DIR/leisaac" ]; then
    sudo -u "$ANSIBLE_USER" git clone --recursive https://github.com/LightwheelAI/leisaac.git "$HOME_DIR/leisaac"
    cd "$HOME_DIR/leisaac"
    sudo -u "$ANSIBLE_USER" git checkout "$LEISAAC_VERSION"
fi

############################
# INSTALLATION
############################
log "Installing LeIsaac and GR00T dependencies"

# Using 'conda run' ensures we are in the correct env context without needing to 'source' conda.sh
sudo -u "$ANSIBLE_USER" -i conda run -n "$CONDA_ENV_NAME" --cwd "$HOME_DIR/leisaac" \
    pip install -e source/leisaac

log "Installing optional GR00T dependencies"
sudo -u "$ANSIBLE_USER" -i conda run -n "$CONDA_ENV_NAME" --cwd "$HOME_DIR/leisaac" \
    pip install -e "source/leisaac[gr00t]"

############################
# VERIFICATION
############################
log "Verifying installation"
sudo -u "$ANSIBLE_USER" -i conda run -n "$CONDA_ENV_NAME" isaaclab -i

log "LeIsaac setup completed successfully."