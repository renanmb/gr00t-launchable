# The Setup IsaacLab Improvements


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