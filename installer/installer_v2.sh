#!/bin/bash

# --- Configuration ---
# We use /home/ubuntu because that's where your files are
BASE_DIR="/home/ubuntu"
STATUS_LOG="/var/log/install_progress.log"
OUTPUT_LOG="/var/log/install_output.log"
SCRIPT_PATH="$BASE_DIR/installer_v2.sh"
INSTALL_OPTIONAL=true 

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

# --- Action Functions (Using Absolute Paths) ---

run_novnc() {
    echo ">>> Starting Step 1: noVNC and Desktop Setup"
    sudo bash "$BASE_DIR/setup-novnc.sh"
    update_status "stage_conda"
    exit 0 # Exit to allow the reboot to take over
}

run_conda() {
    echo ">>> Starting Step 2: Conda Installation"
    sudo bash "$BASE_DIR/setup-conda.sh"
    update_status "stage_isaacsim"
}

run_isaacsim() {
    echo ">>> Starting Step 3: Isaac Sim Installation"
    sudo bash "$BASE_DIR/setup-isaacsim.sh"
    update_status "stage_isaaclab"
}

run_isaaclab() {
    echo ">>> Starting Step 4: Isaac Lab Installation"
    sudo bash "$BASE_DIR/setup-isaaclab.sh"
    if [ "$INSTALL_OPTIONAL" = true ]; then
        update_status "stage_gr00t"
    else
        update_status "completed"
    fi
}

run_gr00t() {
    echo ">>> Starting Step 5: Isaac-GR00T Setup (Optional)"
    sudo bash "$BASE_DIR/setup-gr00t.sh"
    update_status "stage_leisaac"
}

run_leisaac() {
    echo ">>> Starting Step 6: LeIsaac Setup (Optional)"
    sudo bash "$BASE_DIR/setup-leisaac.sh"
    update_status "completed"
}

# --- Main Logic ---

# Ensure logs exist
sudo touch "$OUTPUT_LOG" "$STATUS_LOG"
add_to_cron

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
    sleep 2 # Prevent CPU spiking if a file is missing
done