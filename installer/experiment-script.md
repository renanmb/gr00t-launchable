# Experimenting with Script installer with staging


First Script

This script seems to have a flawed logic, it is using stagging but the logic for the optional variable is flawed and it does not hold idenpotency and state.

This is a simple start point

Runs in the order:

1. setup-novnc.sh 
2. setup-conda.sh 
3. setup-isaacsim.sh 
4. setup-isaaclab.sh 
5. setup-gr00t.sh (optional)
6. setup-leisaac.sh (optional)

This script is a good starting point but there are several mistakes, there is no way for it to check if the status has run and jumping steps that were completed.

After reboot the script will not run since it has not been setup on cron tab 

The ```INSTALL_OPTIONAL=true ``` will determine if Isaac-Gr00T and LeIsaac is installed.

```bash
#!/bin/bash

# Configuration
STATUS_LOG="$HOME/install_progress.log"
# Set this to "true" if you want to run the optional GR00T and LeIsaac steps
INSTALL_OPTIONAL=true 

# Helper to log and update status
update_status() {
    local next_stage=$1
    echo "Transitioning to $next_stage at $(date)"
    echo "$next_stage" > "$STATUS_LOG"
}

# Check current status
if [[ -f $STATUS_LOG ]]; then
    CURRENT_STATUS="$(cat "$STATUS_LOG")"
else
    CURRENT_STATUS="stage_novnc"
    echo "$CURRENT_STATUS" > "$STATUS_LOG"
fi

# --- Action Functions ---

run_novnc() {
    echo ">>> Starting Step 1: noVNC and Desktop Setup"
    # setup-novnc.sh handles its own reboot at the end
    sudo bash ./setup-novnc.sh
    update_status "stage_conda"
    # The novnc script usually ends with a reboot; 
    # if it doesn't, we exit to allow manual control or system update.
    exit 0
}

run_conda() {
    echo ">>> Starting Step 2: Conda Installation"
    sudo bash ./setup-conda.sh
    update_status "stage_isaacsim"
}

run_isaacsim() {
    echo ">>> Starting Step 3: Isaac Sim Installation"
    sudo bash ./setup-isaacsim.sh
    update_status "stage_isaaclab"
}

run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    sudo bash ./setup-isaaclab.sh
    if [ "$INSTALL_OPTIONAL" = true ]; then
        update_status "stage_gr00t"
    else
        update_status "completed"
    fi
}

run_gr00t() {
    echo ">>> Starting Step 5: Isaac-GR00T Setup (Optional)"
    sudo bash ./setup-gr00t.sh
    update_status "stage_leisaac"
}

run_leisaac() {
    echo ">>> Starting Step 6: LeIsaac Setup (Optional)"
    sudo bash ./setup-leisaac.sh
    update_status "completed"
}

# --- Execution Logic ---

case "$CURRENT_STATUS" in
    stage_novnc)
        run_novnc
        ;;
    stage_conda)
        run_conda
        # Fallthrough or re-run script
        $0 
        ;;
    stage_isaacsim)
        run_isaacsim
        $0
        ;;
    stage_isaaclab)
        run_isaaclab
        $0
        ;;
    stage_gr00t)
        run_gr00t
        $0
        ;;
    stage_leisaac)
        run_leisaac
        $0
        ;;
    completed)
        echo "==============================================="
        echo "✅ All software components have been installed!"
        echo "Current Status: $CURRENT_STATUS"
        echo "==============================================="
        ;;
    *)
        echo "Unknown status: $CURRENT_STATUS"
        exit 1
        ;;
esac
```

In other script, had $0 (which means "run myself again") at the end of each function.

This was dumb, Trying to use a While loop as it updates the status

```bash
# --- Improved Execution Logic ---
while true; do
    # Refresh the status from the log file
    CURRENT_STATUS="$(cat "$STATUS_LOG")"

    case "$CURRENT_STATUS" in
        stage_novnc)
            run_novnc # This function contains an 'exit' because of the reboot
            ;;
        stage_conda)
            run_conda
            ;;
        stage_isaacsim)
            run_isaacsim
            ;;
        stage_isaaclab)
            run_isaaclab
            ;;
        stage_gr00t)
            run_gr00t
            ;;
        stage_leisaac)
            run_leisaac
            ;;
        completed)
            echo "✅ All components installed!"
            # Optional: Remove the cron job here so it doesn't run on next boot
            exit 0
            ;;
        *)
            echo "Unknown status: $CURRENT_STATUS"
            exit 1
            ;;
    esac
done
```

Need to Add logic to add the cron job and cleaning logic at the end to remove from the cron jobs as well clean the messy files

## Dealing with reboot with crontab

There is an alternative which is using the systemd instead of crontab

Add crontab job

```bash
crontab -e
```

Add this line at the bottom (replace /path/to/your/script.sh with the actual path):

```bash
@reboot /bin/bash /path/to/your/script.sh >> ~/install_output.log 2>&1
```

Once the installation is completed, remember to remove the @reboot line from your crontab. 

Otherwise, your server will try to "re-install" (or at least check the status) every single time you restart it for the rest of eternity.

This needs to review

```bash
(sudo crontab -l 2>/dev/null; echo "@reboot /bin/bash $(realpath your_script.sh) >> /root/install_output.log 2>&1") | sudo crontab -
```

## Installer fails to get LeIsaac


Running the script twice there is no safeguard or logic to fix the install process.

```bash
/tmp/tmpfcuw05kb: line 18: isaaclab: command not found
ERROR conda.cli.main_run:execute(142): `conda run isaaclab -i` failed. (See above for error)
```

The Scripts is not installing LeIsaac Properly, I t creates the Conda Virtual Environment but it failed to properly install the module IsaacLab and it could not therefore run LeIsaac.

Manual installation works, so the issue is the script being inconsistent.


These are the steps causing inonsistencies:

```bash
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
# LEISAAC SETUP --- This uses custom repo
############################
log "Cloning LeIsaac (${LEISAAC_VERSION})"
if [ ! -d "$HOME_DIR/leisaac" ]; then
    sudo -u "$ANSIBLE_USER" git clone --recursive https://github.com/renanmb/leisaac.git "$HOME_DIR/leisaac"
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
```


## Debugging tricks

Find the Log File globally:

```bash
sudo find / -name "install_progress.log" 2>/dev/null
```

Check if the script is actually running right now

```bash
ps aux | grep -E "setup-|.sh"
```

To see the current stage:

```bash
sudo cat /root/install_progress.log
```