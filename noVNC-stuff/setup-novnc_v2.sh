#!/usr/bin/env bash
set -euo pipefail

############################
# CONFIG
############################
ANSIBLE_USER="ubuntu"
HOME_DIR="/home/${ANSIBLE_USER}"
NEEDS_REBOOT=0

log() { echo -e "\n>>> $*\n"; }
need_reboot() { NEEDS_REBOOT=1; }
ensure_line_in_file() {
  local file="$1"
  local line="$2"
  local after_regex="${3:-}"
  touch "$file"
  if ! grep -Fxq "$line" "$file"; then
    if [[ -n "$after_regex" ]] && grep -Eq "$after_regex" "$file"; then
      sed -i "/$after_regex/a $line" "$file"
    else
      echo "$line" >> "$file"
    fi
  fi
}

############################
# APT + BASE PACKAGES
############################
log "Installing base packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ubuntu-desktop \
  python3-pip \
  net-tools \
  eog \
  expect \
  x11vnc \
  snapd

############################
# GDM / AUTOLOGIN / FORCE XORG
############################
log "Configuring GDM auto-login and forcing Xorg"
GDM_CONF="/etc/gdm3/custom.conf"
ensure_line_in_file "$GDM_CONF" "AutomaticLoginEnable=true" "\[daemon\]"
ensure_line_in_file "$GDM_CONF" "AutomaticLogin=${ANSIBLE_USER}" "\[daemon\]"
ensure_line_in_file "$GDM_CONF" "WaylandEnable=false" "\[daemon\]"
need_reboot

############################
# DISABLE SLEEP / SCREEN LOCK
############################
log "Disabling suspend and screen lock"
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
need_reboot

sudo -u "$ANSIBLE_USER" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u $ANSIBLE_USER)/bus" \
  gsettings set org.gnome.desktop.session idle-delay 0 || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.screensaver lock-enabled false || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.lockdown disable-lock-screen true || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.screensaver idle-activation-enabled false || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0 || true

############################
# GNOME THEME / FONT / TERMINAL
############################
log "Applying GNOME tweaks"
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.interface text-scaling-factor 1.25
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.interface gtk-theme 'Yaru-dark'

PROFILE_ID=$(sudo -u "$ANSIBLE_USER" gsettings get org.gnome.Terminal.ProfilesList default | tr -d \')
BASE="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:$PROFILE_ID/"
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" font 'Monospace 12'
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" use-system-font false
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" background-transparency-percent 10
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" use-transparent-background true
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" background-color "rgb(23,20,33)"
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" foreground-color "rgb(208,207,204)"
sudo -u "$ANSIBLE_USER" gsettings set "$BASE" use-theme-colors false

############################
# DISABLE RELEASE PROMPT
############################
log "Disabling Ubuntu release upgrade prompt"
sed -i 's/^Prompt=.*/Prompt=never/' /etc/update-manager/release-upgrades
need_reboot

############################
# COPY VIRTUAL DISPLAY FILES
############################
log "Installing X11 virtual display files"
install -m 0644 vdisplay.edid /etc/X11/vdisplay.edid
install -m 0644 xorg.conf /etc/X11/xorg.conf
need_reboot

log "Ensuring .Xauthority exists"
install -o "$ANSIBLE_USER" -g "$ANSIBLE_USER" -m 0666 /dev/null "$HOME_DIR/.Xauthority"

############################
# BUS ID UPDATER
############################
log "Installing GPU BusID updater"
cat >/opt/update-busid <<'EOF'
#!/bin/bash
BUS_ID=$(nvidia-xconfig --query-gpu-info | grep 'PCI BusID' | head -n 1 | cut -c15-)
sed -i "s/BusID.*$/BusID \"$BUS_ID\"/" /etc/X11/xorg.conf
EOF
chmod 0755 /opt/update-busid
ensure_line_in_file /etc/gdm3/PreSession/Default "/opt/update-busid"

############################
# SNAP + VS CODE
############################
log "Installing VS Code"
snap install code --classic
sudo -u "$ANSIBLE_USER" code --install-extension ms-vscode-remote.vscode-remote-extensionpack || true

log "Creating VS Code desktop shortcut"
mkdir -p "$HOME_DIR/Desktop"
cat >"$HOME_DIR/Desktop/vscode.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Visual Studio Code
Exec=/snap/bin/code --no-sandbox %F
Icon=/snap/code/current/meta/gui/vscode.png
Terminal=false
Categories=Utility;TextEditor;Development;IDE;
EOF
chown "$ANSIBLE_USER:$ANSIBLE_USER" "$HOME_DIR/Desktop/vscode.desktop"
chmod 0755 "$HOME_DIR/Desktop/vscode.desktop"
sudo -u "$ANSIBLE_USER" gio set "$HOME_DIR/Desktop/vscode.desktop" metadata::trusted true || true

############################
# PYTHON PACKAGES
############################
log "Installing Python requirements"
pip3 install --upgrade pexpect

############################
# SYSTEMD SERVICES
############################
log "Installing x11vnc systemd service"
install -m 0444 x11vnc-ubuntu.service /etc/systemd/system/x11vnc-ubuntu.service

log "Installing noVNC systemd service"
install -m 0444 novnc.service /etc/systemd/system/novnc.service

log "Reloading systemd"
systemctl daemon-reload

log "Enabling x11vnc and noVNC"
systemctl enable --now x11vnc-ubuntu
systemctl enable --now novnc

############################
# FINAL REBOOT
############################
if [[ "$NEEDS_REBOOT" -eq 1 ]]; then
  log "Reboot required. Rebooting now..."
  reboot
else
  log "Setup completed without reboot."
fi
