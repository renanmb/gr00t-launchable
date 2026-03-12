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