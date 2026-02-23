#!/bin/bash

# Configuration - Using absolute path to avoid Root vs User confusion
STATUS_LOG="/var/log/install_progress.log"
SCRIPT_PATH="$(readlink -f "$0")"
INSTALL_OPTIONAL=true 

# --- Crontab Management Functions ---

add_to_cron() {
    # Check if entry already exists to avoid duplicates
    sudo crontab -l 2>/dev/null | grep -q "$SCRIPT_PATH"
    if [ $? -ne 0 ]; then
        echo "Adding script to crontab for reboot persistence..."
        (sudo crontab -l 2>/dev/null; echo "@reboot /bin/bash $SCRIPT_PATH >> /var/log/install_output.log 2>&1") | sudo crontab -
    fi
}

remove_from_cron() {
    echo "Cleaning up: Removing script from crontab..."
    sudo crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | sudo crontab -
}

# --- Helper to log and update status ---
update_status() {
    local next_stage=$1
    echo "Transitioning to $next_stage at $(date)" | sudo tee -a /var/log/install_output.log
    echo "$next_stage" | sudo tee "$STATUS_LOG" > /dev/null
}

# Ensure the log exists and we have permissions
if [[ ! -f $STATUS_LOG ]]; then
    echo "stage_novnc" | sudo tee "$STATUS_LOG" > /dev/null
fi

# Initialize Cron on first run
add_to_cron

# --- Action Functions ---

run_novnc() {
    echo ">>> Starting Step 1: noVNC and Desktop Setup"
    sudo bash ./setup-novnc.sh
    update_status "stage_conda"
    # If setup-novnc.sh triggers a reboot, the script exits here.
    # If it DOESN'T reboot, we exit anyway to let the system settle.
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

while true; do
    CURRENT_STATUS="$(cat "$STATUS_LOG")"

    case "$CURRENT_STATUS" in
        stage_novnc)
            run_novnc
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
            echo "==============================================="
            echo "✅ All software components have been installed!"
            echo "==============================================="
            remove_from_cron
            exit 0
            ;;
        *)
            echo "Unknown status: $CURRENT_STATUS"
            exit 1
            ;;
    esac
done