# The Setup IsaacLab Improvements

test

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Ant-v0 --num_envs 32
```

## experiment with a Launchable script

The script runs, still same issue with the Stage IsaacLab stuck.

```bash
#!/bin/bash
set -e

# 1. Define where the repo will live
WORKSPACE_DIR="/home/ubuntu" 
REPO_NAME="gr00t-launchable"
REPO_PATH="$WORKSPACE_DIR/$REPO_NAME"

# 2. Navigate directly into the scripts directory
echo "Navigating to scripts directory..."
cd "$REPO_PATH/scripts"

# 2. Make the installer executable
chmod +x installer.sh setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh setup-gr00t.sh setup-leisaac.sh

# 3. Run the installer
echo "Executing installer.sh..."

# Running it with nohup so it doesn't block the rest of the Brev setup process, 
# and routing all output to a log file for easy debugging.
nohup ./installer.sh > /home/ubuntu/installer_output.log 2>&1 &

echo "=== Setup Launched Successfully ==="
```


## Modified run_isaaclab with Retry Logic

In the installer script add a retry logic, seems that running twice has fixed the issue.

Update the ```run_isaaclab``` function in your ```installer.sh``` to attempt the installation, source the environment, and try one more time if it fails.

```bash
run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    local ATTEMPT=1
    local MAX_ATTEMPTS=2

    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        echo ">>> Installation Attempt $ATTEMPT of $MAX_ATTEMPTS..."
        
        # 1. Run the script
        if sudo bash "$BASE_DIR/setup-isaaclab.sh"; then
            echo ">>> Performing final verification of Isaac Lab Conda Environment..."
            
            # 2. Source conda and verify
            if sudo -H -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda run -n isaaclab python -c 'import torch'" >/dev/null 2>&1; then
                echo "✅ Isaac Lab successfully verified!"
                if [ "$INSTALL_OPTIONAL" = true ]; then
                    update_status "stage_gr00t"
                else
                    update_status "completed"
                fi
                return 0
            fi
        fi

        # If we reached here, the first attempt failed or verification failed
        echo "⚠️ Attempt $ATTEMPT failed."
        if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
            echo ">>> Sourcing Conda and retrying..."
            # Force a refresh of the conda profile for the ubuntu user
            sudo -H -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda init bash" > /dev/null 2>&1
            sleep 5
        fi
        ATTEMPT=$((ATTEMPT+1))
    done

    echo "❌ Step 4 Failed after $MAX_ATTEMPTS attempts. Halting."
    exit 1
}
```

The issues might be solved by:

- Doing retries within the script: avoid exiting the master script immediately upon the first failure of the sub-script.
- Manual Sourcing: The command source ```/opt/conda/etc/profile.d/conda.sh``` ensures that even if the current shell doesn't recognize the conda command yet, the subshell used for verification will.
- Environment Initialization: Running ```conda init bash``` between attempts ensures the .bashrc is correctly updated for the ubuntu user before the second try.



## Structural Improvement for ```setup-isaaclab.sh```

This is questionable changes, but it is an interesting thought to keep in mind.

To make the retry logic above even more effective, ensure your ```setup-isaaclab.sh``` is truly idempotent by adding this cleanup at the very beginning of the environment setup block:

```bash
# Inside setup-isaaclab.sh
# 3. Zombie Environment Guard
echo "▶ Ensuring clean state for environment: $CONDA_ENV_NAME"
# Remove by name AND physical directory to fix the 'unknown' error
sudo -H -u "$TARGET_USER" /opt/conda/bin/conda env remove -n "$CONDA_ENV_NAME" -y || true
rm -rf "/opt/conda/envs/$CONDA_ENV_NAME"
```

Trying the exec command

```bash
run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    local ATTEMPT=1
    local MAX_ATTEMPTS=2

    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        echo ">>> Installation Attempt $ATTEMPT of $MAX_ATTEMPTS..."
        
        # 1. Run the script
        if sudo bash "$BASE_DIR/setup-isaaclab.sh"; then
            echo ">>> Performing final verification of Isaac Lab Conda Environment..."
            
            # 2. Source conda and verify
            if sudo -H -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda run -n isaaclab python -c 'import torch'" >/dev/null 2>&1; then
                echo "✅ Isaac Lab successfully verified!"
                if [ "$INSTALL_OPTIONAL" = true ]; then
                    update_status "stage_gr00t"
                else
                    update_status "completed"
                fi
                return 0
            fi
        fi

        # If we reached here, the first attempt failed or verification failed
        echo "⚠️ Attempt $ATTEMPT failed."
        if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
            echo ">>> Refreshing disk-level initialization..."
            sudo -H -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda init bash" > /dev/null 2>&1
            sleep 2
        else
            # --- THE EXEC SOLUTION ---
            echo "❌ Max attempts reached. Relaunching script to force environment refresh..."
            sleep 2
            # This replaces the current stale script with a fresh, Conda-aware one
            exec /bin/bash "$SCRIPT_PATH"
        fi
        ATTEMPT=$((ATTEMPT+1))
    done

    echo "❌ Step 4 Failed after $MAX_ATTEMPTS attempts. Halting."
    exit 1
}
```


SImpler version 10: it might need to cd into the repository in order to test it is installed

```bash
run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    local ATTEMPT=1
    local MAX_ATTEMPTS=2

    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        echo ">>> Installation Attempt $ATTEMPT of $MAX_ATTEMPTS..."
        
        # Explicitly pass SHELL to bypass the 'unknown' error
        if sudo SHELL=/bin/bash bash "$BASE_DIR/setup-isaaclab.sh"; then
            echo ">>> Performing final verification..."
            
            # --- Fix: Use a login shell and set the working directory ---
            if sudo -i -u ubuntu bash -c "cd ~/IsaacLab && /opt/conda/bin/conda run -n isaaclab python -c 'import torch'"; then
                echo "✅ Isaac Lab successfully verified!"
                update_status "completed"
                return 0
            fi
        fi

        echo "⚠️ Attempt $ATTEMPT failed."
        if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
            echo ">>> Refreshing shell initialization..."
            sudo -H -u ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh && conda init bash" > /dev/null 2>&1
            sleep 5
        else
            echo "❌ Max attempts reached. Relaunching installer to force fresh pass..."
            sleep 2
            # exec replaces the current process with a fresh shell instance
            exec /bin/bash "$SCRIPT_PATH"
        fi
        ATTEMPT=$((ATTEMPT+1))
    done
}
```


## Error everything installs and passes but fails

```bash
(base) ubuntu@brev-dpo7w313b:~$ if sudo -i -u ubuntu bash -c "cd ~/IsaacLab && /opt/conda/bin/conda run -n isaaclab python -c 'import torch'"; then
                echo "✅ Isaac Lab successfully verified!"
            fi                           
✅ Isaac Lab successfully verified!
(base) ubuntu@brev-dpo7w313b:~$ conda activate isaaclab
(isaaclab) ubuntu@brev-dpo7w313b:~$ cd IsaacLab/
(isaaclab) ubuntu@brev-dpo7w313b:~/IsaacLab$ isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
[INFO] Using python from: /opt/conda/envs/isaaclab/bin/python                                                                                  
Traceback (most recent call last):
  File "/home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py", line 13, in <module>
    from isaaclab.app import AppLauncher
ModuleNotFoundError: No module named 'isaaclab'
```

This was solved by running the proper environment on conda with conda run

## Installer V5 still has the Optional Logic

It also has no reboot after installing conda 

The run_isaaclab with optional:

```bash
run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    # 1. Run the script and check its exit code
    if sudo bash "$BASE_DIR/setup-isaaclab.sh"; then
        echo ">>> Performing final verification of Isaac Lab Conda Environment..."
        
        # 2. Hard Verification: Ask the environment to import torch
        if sudo -H -u ubuntu /opt/conda/bin/conda run -n isaaclab python -c "import torch" >/dev/null 2>&1; then
            echo "✅ Isaac Lab successfully verified!"
            
            # Only transition if verification passes
            if [ "$INSTALL_OPTIONAL" = true ]; then
                update_status "stage_gr00t"
            else
                update_status "completed"
            fi
        else
            echo "❌ Verification Failed: Conda environment 'isaaclab' is missing or broken!"
            echo "Halting master installer."
            exit 1
        fi
    else
        echo "❌ setup-isaaclab.sh encountered a fatal error!"
        echo "Halting master installer."
        exit 1
    fi
}
```


The full script for the Version 05, has issues but it has the optional logic and simples functions rellying on the subscripts to verify installation: ```installer_v5.sh```

```bash
#!/bin/bash

# installer Version: 05 (Verification-Aware Edition)

# --- Configuration ---
# Dynamically find the repository directory, no matter where it was cloned
BASE_DIR="$(dirname "$(readlink -f "$0")")"
TARGET_HOME="/home/ubuntu"
STATUS_LOG="/var/log/install_progress.log"
OUTPUT_LOG="/var/log/install_output.log"
SCRIPT_PATH="$BASE_DIR/installer.sh"
INSTALL_OPTIONAL=false 

# Helper to log and update status
update_status() {
    local next_stage=$1
    echo "Transitioning to $next_stage at $(date)" | sudo tee -a "$OUTPUT_LOG"
    echo "$next_stage" | sudo tee "$STATUS_LOG" > /dev/null
}

# --- Crontab Management ---
add_to_cron() {
    sudo crontab -l 2>/dev/null | grep -q "$SCRIPT_PATH"
    if [ $? -ne 0 ]; then
        (sudo crontab -l 2>/dev/null; echo "@reboot /bin/bash $SCRIPT_PATH >> $OUTPUT_LOG 2>&1") | sudo crontab -
    fi
}

remove_from_cron() {
    sudo crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | sudo crontab -
}

# --- Action Functions (Using Absolute Repo Paths) ---

run_novnc() {
    echo ">>> Starting Step 1: noVNC and Desktop Setup"
    if sudo bash "$BASE_DIR/setup-novnc.sh"; then
        update_status "stage_conda"
        exit 0 
    else
        echo "❌ Step 1 Failed. Halting."
        exit 1
    fi
}

run_conda() {
    echo ">>> Starting Step 2: Conda Installation"
    if sudo bash "$BASE_DIR/setup-conda.sh"; then
        update_status "stage_isaacsim"
    else
        echo "❌ Step 2 Failed. Halting."
        exit 1
    fi
}

run_isaacsim() {
    echo ">>> Starting Step 3: Isaac Sim Installation"
    if sudo bash "$BASE_DIR/setup-isaacsim.sh"; then
        update_status "stage_isaaclab"
    else
        echo "❌ Step 3 Failed. Halting."
        exit 1
    fi
}

run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    
    # 1. Run the script and check its exit code
    if sudo bash "$BASE_DIR/setup-isaaclab.sh"; then
        echo ">>> Performing final verification of Isaac Lab Conda Environment..."
        
        # 2. Hard Verification: Ask the environment to import torch
        if sudo -H -u ubuntu /opt/conda/bin/conda run -n isaaclab python -c "import torch" >/dev/null 2>&1; then
            echo "✅ Isaac Lab successfully verified!"
            
            # Only transition if verification passes
            if [ "$INSTALL_OPTIONAL" = true ]; then
                update_status "stage_gr00t"
            else
                update_status "completed"
            fi
        else
            echo "❌ Verification Failed: Conda environment 'isaaclab' is missing or broken!"
            echo "Halting master installer."
            exit 1
        fi
    else
        echo "❌ setup-isaaclab.sh encountered a fatal error!"
        echo "Halting master installer."
        exit 1
    fi
}

run_gr00t() {
    echo ">>> Starting Step 5: Isaac-GR00T Setup (Optional)"
    if sudo bash "$BASE_DIR/setup-gr00t.sh"; then
        update_status "stage_leisaac"
    else
        echo "❌ Step 5 Failed. Halting."
        exit 1
    fi
}

run_leisaac() {
    echo ">>> Starting Step 6: LeIsaac Setup (Optional)"
    if sudo bash "$BASE_DIR/setup-leisaac.sh"; then
        update_status "completed"
    else
        echo "❌ Step 6 Failed. Halting."
        exit 1
    fi
}

# --- Main Logic ---

# Ensure logs exist
sudo touch "$OUTPUT_LOG" "$STATUS_LOG"

# --- System Readiness Guard ---
echo "Checking system readiness at $(date)..." >> "$OUTPUT_LOG"

MAX_RETRIES=10
RETRY_COUNT=0

# Check if the target user environment exists before proceeding
while [ ! -d "$TARGET_HOME" ] || [ $RETRY_COUNT -lt 1 ]; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "System not ready after $MAX_RETRIES attempts. Exiting." >> "$OUTPUT_LOG"
        exit 1
    fi
    echo "Waiting for environment... (Attempt $((RETRY_COUNT+1)))" >> "$OUTPUT_LOG"
    sleep 20
    RETRY_COUNT=$((RETRY_COUNT+1))
    
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        echo "Network and Disk are ready!" >> "$OUTPUT_LOG"
        break
    fi
done

# Initialize Cron
add_to_cron

# Initialize status if empty
if [ ! -s "$STATUS_LOG" ]; then
    echo "stage_novnc" | sudo tee "$STATUS_LOG" > /dev/null
fi

# --- Execution Loop ---
while true; do
    CURRENT_STATUS="$(cat "$STATUS_LOG")"

    case "$CURRENT_STATUS" in
        stage_novnc)    run_novnc ;;
        stage_conda)    run_conda ;;
        stage_isaacsim) run_isaacsim ;;
        stage_isaaclab) run_isaaclab ;;
        stage_gr00t)    run_gr00t ;;
        stage_leisaac)  run_leisaac ;;
        completed)
            echo "✅ Installation Complete!"
            remove_from_cron
            exit 0
            ;;
        *)
            echo "Unknown status: $CURRENT_STATUS"
            exit 1
            ;;
    esac
    sleep 2 
done
```