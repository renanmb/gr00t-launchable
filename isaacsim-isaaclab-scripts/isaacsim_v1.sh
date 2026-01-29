#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$HOME/isaacsim"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -f isaacsim_510.zip ]; then
  wget -O isaacsim_510.zip \
    'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip'
fi

unzip -o isaacsim_510.zip

echo "Running post-install..."
./post_install.sh

echo "Isaac Sim installation complete ✅"