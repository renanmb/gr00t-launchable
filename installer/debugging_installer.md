# Note son debugging the installer

Here there are a couple notes.

Using the Logs to find issues:

```bash
cat /var/log/install_output.log
```



## Permissions issues when running

Look at these two specific lines from your output:

1. **From Conda:** PermissionError: [Errno 13] Permission denied: '/root'

2. **From Isaac Sim:** /home/ubuntu/isaacsim/post_install.sh: line 30: popd: /root: Permission denied

**The Diagnosis:**

- The ```$HOME``` Variable Trap

**Why is the ```ubuntu``` user trying to access the restricted ```/root``` folder?**

Because your main script is launched by ```@reboot``` in the crontab, it runs as the ultimate ```root``` user. This means the system's ```$HOME``` variable is set to ```/root```, and the current working directory is ```/root```.

When you use the command ```sudo -u ubuntu```, it successfully changes the user ID to ubuntu, but it does not change the ```$HOME``` variable or the current directory! 

* Conda tried to read its config file at ```~/.condarc```, which expanded to ```/root/.condarc```, and crashed.

- Python tried to spawn a background process in its current directory (```/root```), and crashed.

- Isaac Sim's ```post_install.sh``` tried to return to its original directory (```/root```) and crashed.

**The Fix:**

- The ```-H``` Flag and ```cd```

To fix this permanently across all your scripts, we must add the ```-H``` flag (```sudo -H -u ubuntu```), which stands for "Home". This forces ```sudo``` to rewrite the ```$HOME``` variable to the target user's actual home directory (```/home/ubuntu```). We also need to add a quick ```cd "$HOME"``` so the scripts physically move out of the ```/root``` folder before executing anything.


1. Update setup-conda.sh

Find the two places near the bottom where you use sudo -u and add -H and cd:

```bash
# -----------------------------------------------------------------------------
# Initialize conda for target user
# -----------------------------------------------------------------------------
log "Initializing conda for user '${TARGET_USER}'"

sudo -H -u "${TARGET_USER}" bash <<'EOF'
set -e
cd "$HOME" # <--- NEW: Get out of the /root directory

# Ensure profile.d is loaded
source /etc/profile || true

# Initialize conda (safe to re-run)
if ! grep -q "conda initialize" "$HOME/.bashrc"; then
  /opt/conda/bin/conda config --set always_yes true
  /opt/conda/bin/conda init bash
fi
EOF

# -----------------------------------------------------------------------------
# Final verification (non-root)
# -----------------------------------------------------------------------------
log "Final verification as ${TARGET_USER}"

sudo -H -u "${TARGET_USER}" bash <<'EOF'
set -e
cd "$HOME" # <--- NEW
source /etc/profile.d/conda.sh
conda --version
EOF
```

2. Update setup-isaacsim.sh

Add a cd command to leave /root, and add the -H flag to the download, extraction, and post-install steps:

```bash
# ---- create install dir with correct ownership -----------------------------
mkdir -p "$INSTALL_DIR"
chown -R "$TARGET_USER:$TARGET_USER" "$INSTALL_DIR"

cd "$TARGET_HOME" # <--- NEW: Leave the /root directory

# ---- download as ubuntu -----------------------------------------------------
if [[ ! -f "$INSTALL_DIR/$ZIP_NAME" ]]; then
  echo "▶ Downloading Isaac Sim…"
  sudo -H -u "$TARGET_USER" wget -q -O "$INSTALL_DIR/$ZIP_NAME" "$ZIP_URL"
else
  echo "▶ Zip already exists, skipping download"
fi

# ---- unzip as ubuntu --------------------------------------------------------
echo "▶ Extracting…"
sudo -H -u "$TARGET_USER" unzip -oq "$INSTALL_DIR/$ZIP_NAME" -d "$INSTALL_DIR"

# ---- run post_install as ubuntu --------------------------------------------
if [[ -x "$INSTALL_DIR/post_install.sh" ]]; then
  echo "▶ Running post_install.sh as $TARGET_USER"
  sudo -H -u "$TARGET_USER" bash "$INSTALL_DIR/post_install.sh"
```

3. Update setup-isaaclab.sh

Add the -H flag to the git clone and symlink steps, and add cd to the bash block:

```bash
# -----------------------------------------------------------------------------
# Repo & Symlink setup
# -----------------------------------------------------------------------------
cd "$TARGET_HOME" # <--- NEW: Leave the /root directory

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
cd "$TARGET_HOME" # <--- NEW

# 1. Load Conda
source /opt/conda/etc/profile.d/conda.sh
# ... (The rest of the script remains exactly the same)
```