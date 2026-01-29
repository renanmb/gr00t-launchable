# Installing IsaacSim and IsaacLab

```bash
exec bash -l
```

Overall steps to make the script to install IsaacSim and IsaacLab

```bash
scp install-conda.sh isaaclab_v1.sh isaacsim_v1.sh test-g6e-8xlarge-08106e:~
```

```bash
# chmod a+x script_name.sh # make all executable for all users
chmod +x script_name.sh
# or
chmod 755 script_name.sh
```

## Install Miniforge (conda) 

```bash
wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
rm Miniforge3-Linux-x86_64.sh
```

**Install Script for conda**

Error --- It required sudo to run. Means the permission was not set properly

```bash
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
```

**second attempt**, this is implementing root-only guard and other things that might be necessary further for making sure consistent ownership. Could be important for running with containers.

- Run as root-only guard
- Checksum validation of installer
- Retry logic (like retries / delay)
- Auto-add /opt/conda/bin to /etc/profile.d/
- Lockfile to prevent parallel runs


```bash
#!/usr/bin/env bash
set -euo pipefail

# ---------------- CONFIG ----------------
CONDA_DIR="/opt/conda"
PROFILE_D_FILE="/etc/profile.d/conda.sh"
INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
INSTALLER_NAME="Miniforge3-Linux-x86_64.sh"

# SHA256 checksum from official release (example — VERIFY before use)
INSTALLER_SHA256="b6f7d8c5f92a9c4b1d7f9b7f8e8e7f6c5b4a3d2c1e0f1234567890abcdef1234"

LOCKFILE="/var/lock/conda-install.lock"
RETRIES=5
DELAY=3
# ---------------------------------------

log() {
  echo "[conda-install] $1"
}

fail() {
  echo "[conda-install][ERROR] $1" >&2
  exit 1
}

# -------- root-only guard --------
if [[ "$(id -u)" -ne 0 ]]; then
  fail "This script must be run as root (needed for ${CONDA_DIR} and /etc/profile.d)"
fi

# -------- lockfile (prevent parallel runs) --------
exec 200>"${LOCKFILE}"
flock -n 200 || fail "Another conda installation is already running"

log "Acquired lock: ${LOCKFILE}"

# -------- idempotency check --------
if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
  log "Conda already installed at ${CONDA_DIR}, skipping installation."
else
  # -------- download with retry --------
  log "Downloading Miniforge installer..."
  for attempt in $(seq 1 "${RETRIES}"); do
    if wget -q -O "${INSTALLER_NAME}" "${INSTALLER_URL}"; then
      log "Download successful"
      break
    fi

    if [[ "${attempt}" -eq "${RETRIES}" ]]; then
      fail "Failed to download installer after ${RETRIES} attempts"
    fi

    log "Retrying download (${attempt}/${RETRIES})..."
    sleep "${DELAY}"
  done

  # -------- checksum validation --------
  log "Validating installer checksum..."
  echo "${INSTALLER_SHA256}  ${INSTALLER_NAME}" | sha256sum -c - \
    || fail "Checksum validation failed"

  # -------- install --------
  log "Running Miniforge installer..."
  bash "${INSTALLER_NAME}" -b -p "${CONDA_DIR}" || fail "Installer failed"

  # -------- cleanup --------
  rm -f "${INSTALLER_NAME}"

  # -------- verify --------
  [[ -x "${CONDA_DIR}/bin/conda" ]] || fail "Conda binary missing after install"
  log "Conda installed successfully"
fi

# -------- profile.d setup (idempotent) --------
if [[ ! -f "${PROFILE_D_FILE}" ]]; then
  log "Adding conda to system PATH via ${PROFILE_D_FILE}"
  cat <<EOF > "${PROFILE_D_FILE}"
# Conda system-wide initialization
export PATH=${CONDA_DIR}/bin:\$PATH
EOF
  chmod 0644 "${PROFILE_D_FILE}"
else
  log "Profile file already exists, skipping"
fi

# -------- final output --------
log "Installation complete"
"${CONDA_DIR}/bin/conda" --version

```

**Third attempt**

Conda is being installed at the wrong place and with the wrong user and credentials:

```bash
sudo /opt/conda/bin/conda --version
```


## Isaac Sim

The commands to install Isaac Sim

```bash
mkdir ~/isaacsim
cd ~/isaacsim
wget -O isaacsim_510.zip 'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip'
unzip isaacsim_510.zip
./post_install.sh
```

Example of script with task execution

- -e → exit immediately if any command fails
- -u → error on unset variables
- pipefail → fail if any command in a pipe fails


```bash
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
```

### Issues installing IsaacSim

**Solution 1** --- Best practice, Install and run Isaac Sim as a non-root user, e.g. ubuntu.

```bash
sudo mv /root/isaacsim /home/ubuntu/isaacsim
sudo chown -R ubuntu:ubuntu /home/ubuntu/isaacsim
```

Then log into noVNC as ubuntu and run:

```bash
cd ~/isaacsim
./isaac-sim.sh
```

This avoids GPU, X11, Wayland, and file permission issues long-term.

**Solution 2** --- Allow GUI user to access /root

```bash
sudo chmod o+rx /root
sudo chmod -R o+rx /root/isaacsim
```

This is bad but works

**Solution 3** --- Install it properly with the correct user

need to check which user and install it there

```bash
sudo -u ubuntu bash
cd ~
```

Implemented solution 3 in the isaacsim_v2.sh


## IsaacLab

The commands to install IsaacLab

### Verifying the Isaac Sim installation

To avoid the overhead of finding and locating the Isaac Sim installation directory every time, we recommend exporting the following environment variables to your terminal for the remaining of the installation instructions:

```bash
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/isaacsim"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

**Optional** - Check that the simulator runs as expected:

```bash
# note: you can pass the argument "--help" to see all arguments possible.
${ISAACSIM_PATH}/isaac-sim.sh
```

Check that the simulator runs from a standalone python script:

```bash
# checks that python path is set correctly
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
```

Running the Cubes standalone script is optional:

```bash
# checks that Isaac Sim can be launched from python
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py
```

Attention

If you have been using a previous version of Isaac Sim, you need to run the following command for the first time after installation to remove all the old user data and cached variables:

```bash
${ISAACSIM_PATH}/isaac-sim.sh --reset-user
```

This is very unlikely situation since its a fresh install.


### Installing Isaac Lab

Cloning Isaac Lab repo:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
```

Creating the Isaac Sim Symbolic Link

```bash
# enter the cloned repository
cd IsaacLab
# create a symbolic link
ln -s path_to_isaac_sim _isaac_sim
# For example: ln -s ${HOME}/isaacsim _isaac_sim
ln -s ${ISAACSIM_PATH} _isaac_sim
```

Setting up the conda environment

```bash
# Option 1: Default name for conda environment is 'env_isaaclab'
./isaaclab.sh --conda  # or "./isaaclab.sh -c"
```


```bash
# Option 2: Custom name for conda environment
./isaaclab.sh --conda isaaclab
```

Once created, be sure to activate the environment before proceeding!

```bash
conda activate isaaclab
```

Install dependencies using apt (on Linux only):

```bash
# these dependency are needed by robomimic which is not available on Windows
sudo apt install cmake build-essential
```

Run the install command that iterates over all the extensions in source directory and installs them using pip (with --editable flag):

```bash
./isaaclab.sh --install 
```

**Optional** - Verifying the Isaac Lab installation

```bash
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
# ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
isaaclab -p scripts/tutorials/00_sim/create_empty.py
```

```bash
# Option 2: Using python in your virtual environment
python scripts/tutorials/00_sim/create_empty.py
```

**Optional** - Train a robot!

Train Ant

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

Run inference:

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Ant-v0 --num_envs 32
```

Train Quadruped

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

Run inference:

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 32
```

### Isaaclab Fixing issues

Replacing, so it install the isaaclab within the conda env isaaclab. It was not running final step

```bash
# -----------------------------------------------------------------------------
# Install Isaac Lab extensions (editable pip)
# -----------------------------------------------------------------------------
echo "▶ Installing Isaac Lab Python extensions"

sudo -u "$TARGET_USER" bash <<EOF
set -euo pipefail

# Load conda correctly (DO NOT source .bashrc)
source /opt/conda/etc/profile.d/conda.sh

source "\$HOME/.bashrc"
conda activate "$CONDA_ENV_NAME"

cd "$ISAACLAB_DIR"
./isaaclab.sh --install
EOF
```

with

```bash
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
```

