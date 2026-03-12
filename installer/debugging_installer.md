# Note son debugging the installer

Here there are a couple notes.

Using the Logs to find issues:

```bash
cat /var/log/install_output.log

# Check the Stage
cat /var/log/install_progress.log

# Rewind the progress bookmark back to Conda
echo "stage_conda" | sudo tee /var/log/install_progress.log

# check if stuck
ps aux | grep installer.sh
```

## Brev Launch Script

**Method 1: The HTTPS Token Method**

You can use a GitHub Personal Access Token (PAT). 

1. Get a Token: Go to GitHub -> Settings -> Developer Settings -> Personal Access Tokens (Tokens (classic)). Generate a new token and give it ONLY the repo scope.
2. Paste this into Brev:

```bash
#!/bin/bash
set -e

# 1. Clone the private repo using the token embedded in the HTTPS URL
# Replace <YOUR_TOKEN> with your actual GitHub PAT
# Replace <GITHUB_USER> with your username or org
git clone https://<YOUR_TOKEN>@github.com/renanmb/goat_racer_collab.git /home/ubuntu/goat_racer_collab

# 2. Fix ownership since the startup script runs as root
chown -R ubuntu:ubuntu /home/ubuntu/goat_racer_collab

# 3. Start your main installer script in the background!
# (Adjust the path if your installer is inside a subfolder like /scripts)
sudo bash /home/ubuntu/goat_racer_collab/brev-launchable-scripts/installer.sh >> /var/log/install_output.log 2>&1 &
```

Second attempt at method 1

```bash
#!/bin/bash
set -e

echo "Cloning repository..."

git clone https://USERNAME:TOKEN@github.com/ORG/goat_racer_collab.git /home/ubuntu/goat_racer_collab

echo "Starting installer..."

nohup bash /home/ubuntu/goat_racer_collab/brev-launchable-scripts/installer.sh \
  >> /home/ubuntu/install_output.log 2>&1 &

echo "Setup launched successfully."
```

**Method 2: Using the Deploy Key** (This dont work at all no matter what)

If you already generated an SSH Deploy key and prefer to use that, you can use a bash trick called a "Here-Doc" (cat << 'EOF'). This allows your script to "write" the key file to the new machine dynamically before trying to clone.

```bash
#!/bin/bash
set -e

KEY_FILE="/root/.ssh/github_deploy_key"
mkdir -p /root/.ssh

# 1. Write the private key directly to the file system
# Paste your ENTIRE private key between the EOF tags
cat << 'EOF' > "$KEY_FILE"
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACBAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
... (paste the rest of your private key here) ...
-----END OPENSSH PRIVATE KEY-----
EOF

# 2. Secure the key file (SSH requires this)
chmod 600 "$KEY_FILE"

# 3. Clone the private repo using the key (Updated with Reality-Web-Services)
GIT_SSH_COMMAND="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
git clone git@github.com:Reality-Web-Services/goat_racer_collab.git /home/ubuntu/goat_racer_collab

# 4. Clean up the key file immediately for security
rm -f "$KEY_FILE"

# 5. Fix ownership so the ubuntu user can access the files
chown -R ubuntu:ubuntu /home/ubuntu/goat_racer_collab

# 6. Trigger your main installer in the background
sudo bash /home/ubuntu/goat_racer_collab/brev-launchable-scripts/installer.sh >> /var/log/install_output.log 2>&1 &
```

Seconda attempt using the logger to debug

```bash
#!/bin/bash
# Enable debug logging, do NOT exit on error
set -x

echo "--- STARTING BREV SETUP SCRIPT ---"

KEY_FILE="/root/.ssh/github_deploy_key"
mkdir -p /root/.ssh

echo "Writing SSH key..."
cat << 'EOF' > "$KEY_FILE"
-----BEGIN OPENSSH PRIVATE KEY-----
(PASTE YOUR PRIVATE KEY HERE)
-----END OPENSSH PRIVATE KEY-----
EOF

echo "Setting permissions..."
chmod 600 "$KEY_FILE"

echo "Attempting to clone repository..."
GIT_SSH_COMMAND="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
git clone git@github.com:Reality-Web-Services/goat_racer_collab.git /home/ubuntu/goat_racer_collab

echo "Cleaning up key..."
rm -f "$KEY_FILE"

echo "Fixing ownership..."
chown -R ubuntu:ubuntu /home/ubuntu/goat_racer_collab

echo "Triggering installer..."
sudo bash /home/ubuntu/goat_racer_collab/brev-launchable-scripts/installer.sh >> /var/log/install_output.log 2>&1 &

echo "--- SCRIPT FINISHED ---"
```

Third Attempt 

Using base64

```bash
#!/bin/bash
set -e
set -x

# 1. We tell the script where to create the key file INSIDE the Brev machine
KEY_FILE="/root/.ssh/github_deploy_key"
mkdir -p /root/.ssh

# 2. The script decodes your single-line string back into a real SSH file!
# ---> REPLACE THE TEXT BELOW WITH YOUR BASE64 STRING <---
echo "PASTE_YOUR_BASE64_STRING_HERE" | base64 --decode > "$KEY_FILE"

# 3. Secure the newly created key file (SSH demands this)
chmod 600 "$KEY_FILE"

# 4. Use the key to clone your private repo
GIT_SSH_COMMAND="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
git clone git@github.com:Reality-Web-Services/goat_racer_collab.git /home/ubuntu/goat_racer_collab

# 5. Delete the key file from the Brev machine for security
rm -f "$KEY_FILE"

# 6. Fix ownership so the ubuntu user can access the repo
chown -R ubuntu:ubuntu /home/ubuntu/goat_racer_collab

# 7. Start your master installer script in the background!
sudo bash /home/ubuntu/goat_racer_collab/brev-launchable-scripts/installer.sh >> /var/log/install_output.log 2>&1 &
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

Adter the changes are implemented run the following commands:

```bash
# 1. Stop the looping script
sudo pkill -f installer.sh

# 2. Delete the corrupted Conda folder to start totally fresh
sudo rm -rf /opt/conda

# 3. Delete the corrupted Isaac Sim folder (since post_install crashed)
sudo rm -rf /home/ubuntu/isaacsim

# 4. Rewind the progress bookmark back to Conda
echo "stage_conda" | sudo tee /var/log/install_progress.log

# 5. Kick off the fixed installation!
sudo bash /home/ubuntu/gr00t-launchable/scripts/installer.sh &
```


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


## Brev Launchable Startup Script Issues

Ubuntu cloud-initialization system (cloud-init) tried to execute your script, but it crashed instantly with a fatal error.

```txt
cc_scripts_per_boot.py[WARNING]: Failed to run module scripts_per_boot...
cc_scripts_per_instance.py[WARNING]: Failed to run module scripts_per_instance...
```

We can use ```set -x``` turns on "Debug Mode". It tells Bash to print every single command to the log file right before it executes it, along with the exact error message

```bash
# Run this command to see the exact reason it failed:
sudo grep -i "scripts_per_instance" /var/log/cloud-init.log -A 10

# check all logs
sudo cat /var/log/cloud-init-output.log
```


**Hypothesis: An API Payload Bug** (Most likely the issue)

Looking at the user-data.txt you printed earlier, we saw the exact MIME multipart boundaries for always.sh, instance.sh, and once.sh. This proves Brev's infrastructure did try to use standard cloud-init to pass scripts to the VM.

However, because the spaces between the boundaries were completely empty, the hypothesis here is that there is a bug in Brev's deployment API. When you clicked "Deploy," their backend generated the cloud-init "envelope" but failed to Base64-encode and inject your actual script text into the payload before sending the request to the underlying cloud provider (AWS/EC2).

- Where it is: Nowhere on the VM. It exists only in Brev's web database, and the text was dropped during the handoff to the server.

**Hypothesis: Container Entrypoint Injection** (There is no Container)

Brev relies heavily on containerized environments (especially for their "Launchables"). Your instance might actually be running a Docker container on top of the Ubuntu host you SSH'd into. Brev might be passing your startup script as a docker run --entrypoint command or mounting it as a volume directly into a container, bypassing the host's cloud-init entirely.

- Where to look: Check if your environment is containerized.

- Try running: docker ps to see if there is a primary workspace container running, and docker inspect <container_id> to see if your script was passed in as an environment variable or entrypoint argument.


The Cloud-Init is hanging up:

- Failed to run module scripts_per_boot
- Failed to run module scripts_per_instance

```bash
Cloud-init v. 25.3-0ubuntu1~22.04.1 running 'modules:config' at Wed, 11 Mar 2026 09:13:42 +0000. Up 23.07 seconds. Cloud-init v. 25.3-0ubuntu1~22.04.1 running 'modules:final' at Wed, 11 Mar 2026 09:13:54 +0000. Up 35.58 seconds. 2026-03-11 09:13:55,019 - cc_scripts_per_boot.py[WARNING]: Failed to run module scripts_per_boot (per-boot in /var/lib/cloud/scripts/per-boot) 2026-03-11 09:13:55,019 - log_util.py[WARNING]: Running module scripts_per_boot (<module 'cloudinit.config.cc_scripts_per_boot' from '/usr/lib/python3/dist-packages/cloudinit/config/cc_scripts_per_boot.py'>) failed 2026-03-11 09:13:55,028 - cc_scripts_per_instance.py[WARNING]: Failed to run module scripts_per_instance (per-instance in /var/lib/cloud/scripts/per-instance) 2026-03-11 09:13:55,028 - log_util.py[WARNING]: Running module scripts_per_instance (<module 'cloudinit.config.cc_scripts_per_instance' from '/usr/lib/python3/dist-packages/cloudinit/config/cc_scripts_per_instance.py'>) failed Cloud-init v. 25.3-0ubuntu1~22.04.1 finished at Wed, 11 Mar 2026 09:13:55 +0000. Datasource DataSourceEc2Local.
```



## Error with the Installer

```bash
✅ Isaac Sim installed successfully for user 'ubuntu'
Transitioning to stage_isaaclab at Wed Mar 11 17:25:05 UTC 2026
Transitioning to stage_isaaclab at Wed Mar 11 17:25:05 UTC 2026
>>> Starting Step 4: Isaac Lab Installation
▶ Installing Isaac Lab for user: ubuntu
▶ Installing system dependencies
Hit:1 http://us-east-1.ec2.archive.ubuntu.com/ubuntu jammy InRelease
Hit:2 http://us-east-1.ec2.archive.ubuntu.com/ubuntu jammy-updates InRelease
Hit:3 http://us-east-1.ec2.archive.ubuntu.com/ubuntu jammy-backports InRelease
Hit:4 https://download.docker.com/linux/ubuntu jammy InRelease
Hit:5 https://apt.grafana.com stable InRelease
Hit:6 https://apt.corretto.aws stable InRelease
Hit:7 https://nvidia.github.io/libnvidia-container/stable/deb/amd64  InRelease
Hit:8 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Hit:9 https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu jammy InRelease
Get:10 https://repos.influxdata.com/debian stable InRelease [6922 B]
Hit:11 http://security.ubuntu.com/ubuntu jammy-security InRelease
Fetched 6922 B in 1s (6105 B/s)
Reading package lists...
W: Target Packages (Packages) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list:1 and /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list:1
W: Target Translations (en) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list:1 and /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list:1
W: Target Packages (Packages) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list:1 and /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list:1
W: Target Translations (en) is configured multiple times in /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list:1 and /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list:1
Reading package lists...
Building dependency tree...
Reading state information...
build-essential is already the newest version (12.9ubuntu3).
cmake is already the newest version (3.22.1-1ubuntu1.22.04.2).
git is already the newest version (1:2.34.1-1ubuntu1.17).
0 upgraded, 0 newly installed, 0 to remove and 9 not upgraded.
▶ Cloning IsaacLab repository
Cloning into '/home/ubuntu/IsaacLab'...
▶ Creating _isaac_sim symlink
▶ Setting up Conda and installing Isaac Lab extensions
▶ Creating conda environment: isaaclab
'unknown': I need something more specific.
❌ setup-isaaclab.sh encountered a fatal error!
Halting master installer.
```

The Error is happening at:

```bash
# 4. Create environment if it doesn't exist (or was just wiped)
if [ "\$ENV_HEALTHY" = false ]; then
  echo "▶ Creating conda environment: $CONDA_ENV_NAME"
  ./isaaclab.sh --conda "$CONDA_ENV_NAME"
fi
```

**My reflection of what might be happening**

The ```setup-isaaclab.sh``` script executes a subshell as the ```ubuntu``` user. Even though it sources ```/opt/conda/etc/profile.d/conda.sh```, that file only sets the ```PATH```; it does not necessarily set the ```SHELL``` variable that the ```isaaclab.sh``` wrapper script expects for its internal logic.

Since the script requires an initialized shell to run ```./isaaclab.sh --conda```, and that initialization **only becomes "active" after a shell restart**, the script fails on the first pass.

So a solution might be to perform a second pass, another shell restart or a complete reboot of the machine.

**Option 1: Process Restart**

Using ```exec``` to replace the current process with a brand-new bash instance.

```bash
run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    # Run the setup script
    if sudo SHELL=/bin/bash bash "$BASE_DIR/setup-isaaclab.sh"; then
        # ... (Verification logic) ...
        update_status "stage_gr00t"
    else
        echo "⚠️ Environment initialization failed. Relaunching installer to refresh shell..."
        sleep 2
        # RESTART: This relaunch picks up from 'stage_isaaclab' due to the log status
        exec /bin/bash "$SCRIPT_PATH"
    fi
}
```

**Option 2: System Reboot**

```bash
run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    # Check if a 'reboot flag' file exists to prevent infinite reboot loops
    if [ ! -f "/tmp/isaaclab_rebooted" ]; then
        echo "▶ Performing pre-install system reboot to ensure Conda/X11 readiness..."
        touch "/tmp/isaaclab_rebooted"
        sudo reboot
        exit 0
    fi

    # After reboot, the script resumes here
    if sudo SHELL=/bin/bash bash "$BASE_DIR/setup-isaaclab.sh"; then
        rm -f "/tmp/isaaclab_rebooted"
        update_status "stage_gr00t"
    else
        echo "❌ Step 4 failed even after reboot."
        exit 1
    fi
}
```

**Option 3: Reboot after conda**

This might work

```bash
run_conda() {
    echo ">>> Starting Step 2: Conda Installation"
    if sudo bash "$BASE_DIR/setup-conda.sh"; then
        # Update status so it resumes at the NEXT step after rebooting
        update_status "stage_isaacsim"
        
        echo "✅ Conda installed successfully. Rebooting to refresh shell environment..."
        sleep 2
        sudo reboot
        exit 0 # Exit the current script instance
    else
        echo "❌ Step 2 Failed. Halting."
        exit 1
    fi
}
```

Previously there is no reboot but it seems to fail to properly intialize because shell script is not interactive.

```bash
run_conda() {
    echo ">>> Starting Step 2: Conda Installation"
    if sudo bash "$BASE_DIR/setup-conda.sh"; then
        update_status "stage_isaacsim"
    else
        echo "❌ Step 2 Failed. Halting."
        exit 1
    fi
}
```

```bash
# 3. Zombie Environment Guard
ENV_HEALTHY=false
if conda env list | awk '{print \$1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "▶ Conda environment '$CONDA_ENV_NAME' exists. Checking health..."
  
  # Test if the environment is actually functional by querying python
  if conda run -n "$CONDA_ENV_NAME" python --version &>/dev/null; then
    echo "▶ Environment is healthy."
    ENV_HEALTHY=true
  else
    echo "▶ Detected broken/incomplete environment! Removing for a clean slate..."
    conda env remove -n "$CONDA_ENV_NAME" -y
  fi
fi
```


```bash
awk: cmd. line:1: {print \$1}
awk: cmd. line:1:        ^ backslash not last character on line
awk: cmd. line:1: {print \$1}
awk: cmd. line:1:        ^ syntax error
Exception ignored on flushing sys.stdout:
```