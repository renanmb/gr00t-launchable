#!/bin/bash

# installer Version: 04 (Repo-Aware)

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
    sudo bash "$BASE_DIR/setup-novnc.sh"
    update_status "stage_conda"
    exit 0 
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