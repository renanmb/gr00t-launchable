# making the Leatherback Installer


First Attempt

The script has an issue with the paths

```bash
>>> Starting Step 5: Leatherback Setup

❌ [ERROR] Leatherback directory not found at: /home/ubuntu/goat_racer_test/brev-launchable-scripts/leatherback
❌ Step 5 Failed. Halting.
```

```bash
#!/usr/bin/env bash
set -euo pipefail

# setup-leatherback.sh (version 01)
# Installs the Leatherback project into the 'isaaclab' conda environment

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_USER="ubuntu"
CONDA_ENV_NAME="isaaclab"
# Find the repository root (goat_racer_test) based on this script's location
REPO_ROOT="$(readlink -f "$(dirname "$0")")"
LEATHERBACK_DIR="$REPO_ROOT/leatherback"

log() { echo -e "\n>>> [Leatherback Setup] $*"; }
fail() { echo -e "\n❌ [ERROR] $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-checks
# -----------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
  fail "This script must be run as root (sudo)"
fi

if [[ ! -d "$LEATHERBACK_DIR" ]]; then
  fail "Leatherback directory not found at: $LEATHERBACK_DIR"
fi

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
log "Installing Leatherback into environment: $CONDA_ENV_NAME"

# We use -i (login shell) and -u (target user) to ensure PATHs are loaded correctly.
# 'conda run' executes the command within the specific environment context.
sudo -H -u "$TARGET_USER" -i \
    conda run -n "$CONDA_ENV_NAME" \
    --cwd "$LEATHERBACK_DIR" \
    python -m pip install -e source/leatherback

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
log "Verifying installation..."

if sudo -H -u "$TARGET_USER" -i conda run -n "$CONDA_ENV_NAME" python -c "import leatherback; print('Leatherback successfully imported!')" ; then
    log "✅ Leatherback setup completed successfully."
else
    fail "Verification failed. Leatherback module not found in '$CONDA_ENV_NAME'."
fi
```

Solution:

```bash
#!/usr/bin/env bash
set -euo pipefail

# setup-leatherback.sh (version 01)
# Installs the Leatherback project into the 'isaaclab' conda environment

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_USER="ubuntu"
CONDA_ENV_NAME="isaaclab"

# Go one directory up from the script's location to reach 'goat_racer_test'
REPO_ROOT="$(readlink -f "$(dirname "$0")/..")"
LEATHERBACK_DIR="$REPO_ROOT/leatherback"

log() { echo -e "\n>>> [Leatherback Setup] $*"; }
fail() { echo -e "\n❌ [ERROR] $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-checks
# -----------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
  fail "This script must be run as root (sudo)"
fi

if [[ ! -d "$LEATHERBACK_DIR" ]]; then
  fail "Leatherback directory not found at: $LEATHERBACK_DIR"
fi

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
log "Installing Leatherback into environment: $CONDA_ENV_NAME"

# We use -i (login shell) and -u (target user) to ensure PATHs are loaded correctly.
# 'conda run' executes the command within the specific environment context.
sudo -H -u "$TARGET_USER" -i \
    conda run -n "$CONDA_ENV_NAME" \
    --cwd "$LEATHERBACK_DIR" \
    python -m pip install -e source/leatherback

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
log "Verifying installation..."

if sudo -H -u "$TARGET_USER" -i conda run -n "$CONDA_ENV_NAME" python -c "import leatherback; print('Leatherback successfully imported!')" ; then
    log "✅ Leatherback setup completed successfully."
else
    fail "Verification failed. Leatherback module not found in '$CONDA_ENV_NAME'."
fi
```

## Verificaiton failure

```bash
>>> [Leatherback Setup] Verifying installation...
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/ubuntu/goat_racer_test/leatherback/source/leatherback/leatherback/__init__.py", line 11, in <module>
    from .tasks import *
  File "/home/ubuntu/goat_racer_test/leatherback/source/leatherback/leatherback/tasks/__init__.py", line 12, in <module>
    from isaaclab_tasks.utils import import_packages
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/__init__.py", line 33, in <module>
    from .utils import import_packages
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/__init__.py", line 9, in <module>
    from .parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", line 17, in <module>
    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/envs/__init__.py", line 45, in <module>
    from . import mdp, ui
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/envs/mdp/__init__.py", line 18, in <module>
    from .actions import *  # noqa: F401, F403
    ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/__init__.py", line 8, in <module>
    from .actions_cfg import *
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py", line 8, in <module>
    from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/controllers/__init__.py", line 14, in <module>
    from .differential_ik import DifferentialIKController
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik.py", line 12, in <module>
    from isaaclab.utils.math import apply_delta_pose, compute_pose_error
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/utils/__init__.py", line 14, in <module>
    from .mesh import *
  File "/home/ubuntu/IsaacLab/source/isaaclab/isaaclab/utils/mesh.py", line 14, in <module>
    from pxr import Usd, UsdGeom
ModuleNotFoundError: No module named 'pxr'
ERROR conda.cli.main_run:execute(142): `conda run python -c import leatherback; print('Leatherback successfully imported!')` failed. (See above for error)

❌ [ERROR] Verification failed. Leatherback module not found in 'isaaclab'.
❌ Step 5 Failed. Halting.
```

## Must initialize git lfs

Linux

For Debian/Ubuntu-based systems:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
```

Initialize Git LFS

```bash
git lfs install
```

```bash
#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_USER="ubuntu"

log() { echo -e "\n>>> [Git LFS Setup] $*"; }
fail() { echo -e "\n❌ [ERROR] $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-checks
# -----------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
  fail "This script must be run as root (sudo)"
fi

# -----------------------------------------------------------------------------
# Install Git LFS
# -----------------------------------------------------------------------------
log "Checking for Git LFS..."

if ! command -v git-lfs &> /dev/null; then
    log "Git LFS not found. Installing via PackageCloud..."
    
    # We don't need 'sudo bash' here because the script is already running as root
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get update
    apt-get install -y git-lfs
    
    log "✅ Git LFS installed successfully."
else
    log "✅ Git LFS is already installed."
fi

# -----------------------------------------------------------------------------
# Initialize Git LFS
# -----------------------------------------------------------------------------
log "Initializing Git LFS for user: $TARGET_USER"

# Run the initialization as the target user so their ~/.gitconfig is updated
sudo -H -u "$TARGET_USER" git lfs install

log "✅ Git LFS setup complete!"
```