# Isaac Automator Ansible

Notes:

This one works the other attemps fail and the cliboard is not working:

```bash
setup-novnc.sh
```

Some exploration witht he font sizes and background might be needed.


Installation process from Isaac Automator

It installs the rdesktop requirements and then install IsaacSim and Isaaclab using Docker.

The Docker is not ideal for development since it has extra complexity with the networking and setting up the volumes.

Copy local files to instance:

```bash
scp <local-file-path> <brev-instance-name>:<remote-file-path>
```

test-g6e-8xlarge-65964e

```bash
scp setup-novnc_v2.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-65964e:~

scp setup-novnc_v3.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-65964e:~

chmod +x setup-novnc_v3.sh
```

Important note: must disable wayland 

```bash
sudo vim /etc/gdm3/custom.conf
```

```txt
[daemon]
# Uncomment and force Xorg
WaylandEnable=false
AutomaticLoginEnable=true
AutomaticLogin=ubuntu
```

## Connect to instance

```bash
curl ifconfig.me
```
54.198.39.94

http://<instance-public-ip>:6080/vnc.html

http://44.204.96.88:6080/vnc.html

## The entrypoint for rdesktop

The Prerequisites is checking if ubuntu-desktop and python3-pip is installed and it has a conditional case if it is Ubuntu 20.04 which is not necessary

To check if the ubuntu-desktop package is installed:

```bash
dpkg -l | grep ubuntu-desktop
```

The main.yml 

```yml
---
# check if we need to skip stuff
- name: Check installed services
  service_facts:

- name: Prerequisites (1)
  apt: |
    name="{{ item }}"
    state=latest
    update_cache=yes
    install_recommends=no
  with_items:
    - ubuntu-desktop
    - python3-pip

# install only if ubuntu 20
- name: Prerequisites (2)
  apt: name=yaru-theme-gtk state=latest
  when: ansible_distribution_release == "focal"

- name: Configure desktop environment
  import_tasks: desktop.yml

- name: Virtual display
  import_tasks: virtual-display.yml

# updates bus id of the gpu in the xorg.conf file
# needed for starting from the image without ansible
- name: Bus ID updater
  import_tasks: busid.yml

# install misc utils
- name: Misc utils
  import_tasks: utils.yml

# install visual studio code
- name: Visual Studio Code
  import_tasks: vscode.yml
  tags:
    - __vscode

# VNC
- name: VNC server
  import_tasks: vnc.yml

# NoMachine

- name: NoMachine server
  import_tasks: nomachine.yml
  when: "'nxserver.service' not in ansible_facts.services"
  tags:
    - skip_in_ovami

# NoVNC
- name: NoVNC server
  import_tasks: novnc.yml

# do reboots if needed
- name: Reboot if needed
  meta: flush_handlers

```

## Configure desktop environment

The variables and what they are assigned:

**ansible_user** --- ubuntu



The desktop.yml


```yml
---
#
# Good desktop experience
#

- name: Configure auto login [1]
  lineinfile:
    path: /etc/gdm3/custom.conf
    state: present
    line: "AutomaticLoginEnable=true"
    insertafter: "\\[daemon\\]"
  notify: reboot

- name: Configure auto login [2]
  lineinfile:
    path: /etc/gdm3/custom.conf
    state: present
    line: "AutomaticLogin=ubuntu"
    insertafter: "\\[daemon\\]"
  notify: reboot

# disable blank screen
- name: Mask sleep targets
  shell: systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
  notify: reboot

# disable screen lock
- name: Disable screen lock
  shell: "{{ item }}"
  with_items:
    - gsettings set org.gnome.desktop.session idle-delay 0
    - gsettings set org.gnome.desktop.screensaver lock-enabled 'false'
    - gsettings set org.gnome.desktop.lockdown disable-lock-screen 'true'
    - gsettings set org.gnome.desktop.screensaver idle-activation-enabled 'false'
    - gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0
  become_user: "{{ ansible_user }}"
  notify: reboot

# increase font size
- name: Set font size to 125%
  shell: gsettings set org.gnome.desktop.interface text-scaling-factor 1.25
  become_user: "{{ ansible_user }}"

# enable dark theme
- name: Make it dark
  shell: gsettings set org.gnome.desktop.interface gtk-theme 'Yaru-dark'
  become_user: "{{ ansible_user }}"

- name: Tweak terminal settings
  shell: gsettings set org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:$(gsettings get org.gnome.Terminal.ProfilesList default|tr -d \')/ {{ item.setting }} {{ item.value }}
  become_user: "{{ ansible_user }}"
  with_items:
    - { setting: "font", value: "'Monospace 12'" }
    - { setting: "use-system-font", value: "false" }
    - { setting: "background-transparency-percent", value: "10" }
    - { setting: "use-transparent-background", value: "true" }
    - { setting: "background-color", value: '"rgb(23,20,33)"' }
    - { setting: "foreground-color", value: '"rgb(208,207,204)"' }
    - { setting: "use-theme-colors", value: "false" }
  tags: _u22

# disable new ubuntu version prompt

- name: Disable new ubuntu version prompt
  lineinfile:
    path: /etc/update-manager/release-upgrades
    regexp: "Prompt=.*"
    line: "Prompt=never"
  notify: reboot
```

## Virtual display

virtual-display.yml

```yml
---
- name: Copy EDID file
  template: src=vdisplay.edid
    dest=/etc/X11/vdisplay.edid
    mode=644
  notify: reboot

- name: Get PCI Bus ID of the first GPU
  shell: nvidia-xconfig --query-gpu-info | grep 'PCI BusID' | head -n 1 | cut -c15-
  register: GPU0_PCI_BUS_ID

- name: Write X11 config
  template: src=xorg.conf
    dest=/etc/X11/xorg.conf
    mode=644
  notify: reboot

- name: Create Xauthority file
  file:
    path: /home/{{ ansible_user }}/.Xauthority
    state: touch
    owner: "{{ ansible_user }}"
    group: "{{ ansible_user }}"
    mode: 0666
```

## Bus ID updater

Updating the BusID in xorg.conf before GDM starts is necessary to explicitly tell the Xorg server which GPU to use as the primary display adapter, preventing startup failures or incorrect graphics rendering

busid.yml

```yml

# update BusID in xorg.conf before GDM start
# executed from /etc/gdm3/PreSession/Default

- name: Create BusID updater
  copy:
    content: |
      #!/bin/bash
      BUS_ID=$(nvidia-xconfig --query-gpu-info | grep 'PCI BusID' | head -n 1 | cut -c15-)
      sed -i "s/BusID.*$/BusID \"$BUS_ID\"/" /etc/X11/xorg.conf
    dest: /opt/update-busid
    mode: 0755

# add /opt/update-busid to /etc/gdm3/PreSession/Default
- name: Add BusID updater to /etc/gdm3/PreSession/Default
  lineinfile:
    path: /etc/gdm3/PreSession/Default
    line: /opt/update-busid
    insertafter: EOF
    state: present
```

## install misc utils

This is installing EOG image viewer

utils.yml

```yml
# install extra packages
- name: Install extra packages
  apt: name={{ item }}
    state=latest
    update_cache=yes
    install_recommends=no
  with_items:
    - eog # EOG image viewer (https://help.gnome.org/users/eog/stable/)
```

## install visual studio code

Must replace the ansible_user

vscode.yml

```yml
- name: Prerequisites
  apt:
    name: snapd
    state: latest

# install visual studio code
- name: Install Visual Studio Code
  snap:
    name: code
    state: present
    classic: yes

# install remote development extension pack
- name: Install Remote Development extension pack
  shell: code --install-extension ms-vscode-remote.vscode-remote-extensionpack
  become_user: "{{ ansible_user }}"

# make sure desktop directory exists
- name: Make sure desktop directory exists
  file:
    path: /home/{{ ansible_user }}/Desktop
    state: directory
    mode: 0755
    owner: "{{ ansible_user }}"
    group: "{{ ansible_user }}"

# create a desktop shortcut for visual studio code
- name: Create desktop shortcut for Visual Studio Code
  copy:
    dest: /home/{{ ansible_user }}/Desktop/vscode.desktop
    mode: 0755
    owner: "{{ ansible_user }}"
    group: "{{ ansible_user }}"
    content: |
      [Desktop Entry]
      Version=1.0
      Type=Application
      Name=Visual Studio Code
      GenericName=Text Editor
      Comment=Edit text files
      Exec=/snap/bin/code --no-sandbox --unity-launch %F
      Icon=/snap/code/current/meta/gui/vscode.png
      StartupWMClass=Code
      StartupNotify=true
      Terminal=false
      Categories=Utility;TextEditor;Development;IDE;
      MimeType=text/plain;inode/directory;
      Actions=new-empty-window;
      Keywords=vscode;
  become_user: "{{ ansible_user }}"

- name: Allow execution of desktop icon for Visual Studio Code
  shell: gio set "/home/{{ ansible_user }}/Desktop/{{ item }}" metadata::trusted true
  become_user: "{{ ansible_user }}"
  with_items:
    - vscode.desktop
```

## VNC

vnc.yml

```yml
---
- name: Prerequisites (PIP packages)
  pip:
    name: "{{ item }}"
    state: latest
  with_items:
    - pexpect

- name: Install x11vnc and helper util
  apt: name={{ item }}
    update_cache=yes
    state=latest
  with_items:
    - x11vnc
    - expect

- name: Add x11vnc systemd config
  template: src=x11vnc-ubuntu.service
    dest=/etc/systemd/system
    mode=0444
    owner=root
    group=root

- name: Start x11vnc
  systemd: name=x11vnc-ubuntu
    daemon_reload=yes
    enabled=yes
    state=restarted

- name: Clear VNC password
  file:
    path: /home/ubuntu/.vnc/passwd
    state: absent

- name: Set VNC password
  expect:
    command: /usr/bin/x11vnc -storepasswd
    responses:
      (?i).*password:.*: "{{ vnc_password }}\r"
      (?i)write.*: "y\r"
    creates: /home/ubuntu/.vnc/passwd
  become_user: ubuntu
  tags:
    - skip_in_image

- name: Cleanup VNC password
  file:
    path: /home/ubuntu/.vnc/passwd
    state: absent
  tags:
    - never
    - cleanup
```

## Install noVNC



novnc.yml

```yml
# Install noVNC

- name: Prerequisites
  apt:
    name: snapd
    state: latest

# Install noVNC via snap package
- name: Install noVNC
  snap:
    name: novnc
    state: present

- name: Add noVNC systemd config
  template: src=novnc.service
    dest=/etc/systemd/system
    mode=0444
    owner=root
    group=root

- name: Start noVNC
  systemd: name=novnc
    daemon_reload=yes
    enabled=yes
    state=restarted
```


## Putting it all together

Generate a bash script that works like Ansible to install all the requirements as stated below. The goal is to run and configure the machine to run noVNC.

```yml
# check if we need to skip stuff
- name: Check installed services
  service_facts:

- name: Prerequisites (1)
  apt: |
    name="{{ item }}"
    state=latest
    update_cache=yes
    install_recommends=no
  with_items:
    - ubuntu-desktop
    - python3-pip

# - name: Configure desktop environment
#   import_tasks: desktop.yml

- name: Configure auto login [1]
  lineinfile:
    path: /etc/gdm3/custom.conf
    state: present
    line: "AutomaticLoginEnable=true"
    insertafter: "\\[daemon\\]"
  notify: reboot

- name: Configure auto login [2]
  lineinfile:
    path: /etc/gdm3/custom.conf
    state: present
    line: "AutomaticLogin=ubuntu"
    insertafter: "\\[daemon\\]"
  notify: reboot

# disable blank screen
- name: Mask sleep targets
  shell: systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
  notify: reboot

# disable screen lock
- name: Disable screen lock
  shell: "{{ item }}"
  with_items:
    - gsettings set org.gnome.desktop.session idle-delay 0
    - gsettings set org.gnome.desktop.screensaver lock-enabled 'false'
    - gsettings set org.gnome.desktop.lockdown disable-lock-screen 'true'
    - gsettings set org.gnome.desktop.screensaver idle-activation-enabled 'false'
    - gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0
  become_user: "{{ ansible_user }}"
  notify: reboot

# increase font size
- name: Set font size to 125%
  shell: gsettings set org.gnome.desktop.interface text-scaling-factor 1.25
  become_user: "{{ ansible_user }}"

# enable dark theme
- name: Make it dark
  shell: gsettings set org.gnome.desktop.interface gtk-theme 'Yaru-dark'
  become_user: "{{ ansible_user }}"

- name: Tweak terminal settings
  shell: gsettings set org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:$(gsettings get org.gnome.Terminal.ProfilesList default|tr -d \')/ {{ item.setting }} {{ item.value }}
  become_user: "{{ ansible_user }}"
  with_items:
    - { setting: "font", value: "'Monospace 12'" }
    - { setting: "use-system-font", value: "false" }
    - { setting: "background-transparency-percent", value: "10" }
    - { setting: "use-transparent-background", value: "true" }
    - { setting: "background-color", value: '"rgb(23,20,33)"' }
    - { setting: "foreground-color", value: '"rgb(208,207,204)"' }
    - { setting: "use-theme-colors", value: "false" }
  tags: _u22

# disable new ubuntu version prompt

- name: Disable new ubuntu version prompt
  lineinfile:
    path: /etc/update-manager/release-upgrades
    regexp: "Prompt=.*"
    line: "Prompt=never"
  notify: reboot

# - name: Virtual display
#   import_tasks: virtual-display.yml

- name: Copy EDID file
  template: src=vdisplay.edid
    dest=/etc/X11/vdisplay.edid
    mode=644
  notify: reboot

- name: Get PCI Bus ID of the first GPU
  shell: nvidia-xconfig --query-gpu-info | grep 'PCI BusID' | head -n 1 | cut -c15-
  register: GPU0_PCI_BUS_ID

- name: Write X11 config
  template: src=xorg.conf
    dest=/etc/X11/xorg.conf
    mode=644
  notify: reboot

- name: Create Xauthority file
  file:
    path: /home/{{ ansible_user }}/.Xauthority
    state: touch
    owner: "{{ ansible_user }}"
    group: "{{ ansible_user }}"
    mode: 0666

# updates bus id of the gpu in the xorg.conf file
# needed for starting from the image without ansible
# - name: Bus ID updater
#   import_tasks: busid.yml

# update BusID in xorg.conf before GDM start
# executed from /etc/gdm3/PreSession/Default

- name: Create BusID updater
  copy:
    content: |
      #!/bin/bash
      BUS_ID=$(nvidia-xconfig --query-gpu-info | grep 'PCI BusID' | head -n 1 | cut -c15-)
      sed -i "s/BusID.*$/BusID \"$BUS_ID\"/" /etc/X11/xorg.conf
    dest: /opt/update-busid
    mode: 0755

# add /opt/update-busid to /etc/gdm3/PreSession/Default
- name: Add BusID updater to /etc/gdm3/PreSession/Default
  lineinfile:
    path: /etc/gdm3/PreSession/Default
    line: /opt/update-busid
    insertafter: EOF
    state: present

# install misc utils
# - name: Misc utils
#   import_tasks: utils.yml

# install extra packages
- name: Install extra packages
  apt: name={{ item }}
    state=latest
    update_cache=yes
    install_recommends=no
  with_items:
    - eog # EOG image viewer (https://help.gnome.org/users/eog/stable/)

# install visual studio code
# - name: Visual Studio Code
#   import_tasks: vscode.yml
#   tags:
#     - __vscode

- name: Prerequisites
  apt:
    name: snapd
    state: latest

# install visual studio code
- name: Install Visual Studio Code
  snap:
    name: code
    state: present
    classic: yes

# install remote development extension pack
- name: Install Remote Development extension pack
  shell: code --install-extension ms-vscode-remote.vscode-remote-extensionpack
  become_user: "{{ ansible_user }}"

# make sure desktop directory exists
- name: Make sure desktop directory exists
  file:
    path: /home/{{ ansible_user }}/Desktop
    state: directory
    mode: 0755
    owner: "{{ ansible_user }}"
    group: "{{ ansible_user }}"

# create a desktop shortcut for visual studio code
- name: Create desktop shortcut for Visual Studio Code
  copy:
    dest: /home/{{ ansible_user }}/Desktop/vscode.desktop
    mode: 0755
    owner: "{{ ansible_user }}"
    group: "{{ ansible_user }}"
    content: |
      [Desktop Entry]
      Version=1.0
      Type=Application
      Name=Visual Studio Code
      GenericName=Text Editor
      Comment=Edit text files
      Exec=/snap/bin/code --no-sandbox --unity-launch %F
      Icon=/snap/code/current/meta/gui/vscode.png
      StartupWMClass=Code
      StartupNotify=true
      Terminal=false
      Categories=Utility;TextEditor;Development;IDE;
      MimeType=text/plain;inode/directory;
      Actions=new-empty-window;
      Keywords=vscode;
  become_user: "{{ ansible_user }}"

- name: Allow execution of desktop icon for Visual Studio Code
  shell: gio set "/home/{{ ansible_user }}/Desktop/{{ item }}" metadata::trusted true
  become_user: "{{ ansible_user }}"
  with_items:
    - vscode.desktop

# - name: VNC server
#   import_tasks: vnc.yml

- name: Prerequisites (PIP packages)
  pip:
    name: "{{ item }}"
    state: latest
  with_items:
    - pexpect

- name: Install x11vnc and helper util
  apt: name={{ item }}
    update_cache=yes
    state=latest
  with_items:
    - x11vnc
    - expect

- name: Add x11vnc systemd config
  template: src=x11vnc-ubuntu.service
    dest=/etc/systemd/system
    mode=0444
    owner=root
    group=root

- name: Start x11vnc
  systemd: name=x11vnc-ubuntu
    daemon_reload=yes
    enabled=yes
    state=restarted

- name: Clear VNC password
  file:
    path: /home/ubuntu/.vnc/passwd
    state: absent

- name: Set VNC password
  expect:
    command: /usr/bin/x11vnc -storepasswd
    responses:
      (?i).*password:.*: "{{ vnc_password }}\r"
      (?i)write.*: "y\r"
    creates: /home/ubuntu/.vnc/passwd
  become_user: ubuntu
  tags:
    - skip_in_image

- name: Cleanup VNC password
  file:
    path: /home/ubuntu/.vnc/passwd
    state: absent
  tags:
    - never
    - cleanup

# NoVNC
# - name: NoVNC server
#   import_tasks: novnc.yml

# Install noVNC

- name: Prerequisites
  apt:
    name: snapd
    state: latest

# Install noVNC via snap package
- name: Install noVNC
  snap:
    name: novnc
    state: present

- name: Add noVNC systemd config
  template: src=novnc.service
    dest=/etc/systemd/system
    mode=0444
    owner=root
    group=root

- name: Start noVNC
  systemd: name=novnc
    daemon_reload=yes
    enabled=yes
    state=restarted

# do reboots if needed
- name: Reboot if needed
  meta: flush_handlers
```

# Install Script

```bash
#!/usr/bin/env bash
set -euo pipefail

############################
# CONFIG
############################
ANSIBLE_USER="ubuntu"
HOME_DIR="/home/${ANSIBLE_USER}"
NEEDS_REBOOT=0

############################
# HELPERS
############################
log() {
  echo -e "\n>>> $*\n"
}

need_reboot() {
  NEEDS_REBOOT=1
}

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
# APT PREREQUISITES
############################
log "Installing desktop + base packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ubuntu-desktop \
  python3-pip

############################
# AUTO LOGIN CONFIG
############################
log "Configuring GDM auto-login"

GDM_CONF="/etc/gdm3/custom.conf"

ensure_line_in_file "$GDM_CONF" "AutomaticLoginEnable=true" "\[daemon\]"
ensure_line_in_file "$GDM_CONF" "AutomaticLogin=${ANSIBLE_USER}" "\[daemon\]"
need_reboot

############################
# DISABLE SUSPEND / LOCK
############################
log "Disabling sleep targets"
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
need_reboot

log "Disabling GNOME lock & power settings"
sudo -u "$ANSIBLE_USER" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u $ANSIBLE_USER)/bus" \
  gsettings set org.gnome.desktop.session idle-delay 0 || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.screensaver lock-enabled false || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.lockdown disable-lock-screen true || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.screensaver idle-activation-enabled false || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0 || true
need_reboot

############################
# GNOME TWEAKS
############################
log "Applying GNOME appearance tweaks"

sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.interface text-scaling-factor 1.25 || true
sudo -u "$ANSIBLE_USER" gsettings set org.gnome.desktop.interface gtk-theme 'Yaru-dark' || true

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
# VIRTUAL DISPLAY / X11
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
# EXTRA PACKAGES
############################
log "Installing extra utilities"
apt-get install -y --no-install-recommends eog expect x11vnc

############################
# SNAP + VSCODE
############################
log "Installing snapd + VS Code"
apt-get install -y snapd
snap install code --classic

log "Installing VS Code Remote extensions"
sudo -u "$ANSIBLE_USER" code --install-extension ms-vscode-remote.vscode-remote-extensionpack || true

############################
# DESKTOP SHORTCUT
############################
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
# X11VNC SERVICE
############################
log "Installing x11vnc systemd service"
install -m 0444 x11vnc-ubuntu.service /etc/systemd/system/x11vnc-ubuntu.service

systemctl daemon-reload
systemctl enable --now x11vnc-ubuntu

############################
# NOVNC
############################
log "Installing noVNC"
snap install novnc

install -m 0444 novnc.service /etc/systemd/system/novnc.service

systemctl daemon-reload
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

```

## Issues


✅ Xorg is running

✅ Display :0 exists

✅ GDM started

❌ x11vnc is still failing → this narrows it to authentication

```bash
ubuntu@brev-9wo4s0s8b:~$ ls -l /tmp/.X11-unix/ 
total 0 
srwxrwxr-x 1 ubuntu ubuntu 0 Jan 26 02:34 X0 
srwxrwxr-x 1 ubuntu ubuntu 0 Jan 26 02:34 X1
```

Your service is using:

```bash
-auth /home/ubuntu/.Xauthority
```

But in GDM-based sessions:

.Xauthority is often empty, regenerated, or ignored

The real authority file is managed by GDM

Hard-coding .Xauthority breaks x11vnc on reboot

That’s why -auth guess exists — and why almost every working x11vnc service uses it.


Inside

```bash
sudo nano /etc/systemd/system/x11vnc-ubuntu.service
```

Change this:

```txt
ExecStart=/usr/bin/x11vnc -auth /home/ubuntu/.Xauthority -display :0 -rfbport 5900 -shared -usepw
```

with

```txt
ExecStart=/usr/bin/x11vnc -auth guess -display :0 -rfbport 5900 -shared -usepw
```

New issues

🔴 Root cause (almost certainly this)

Your service still has:

```txt
-usepw
```

Since noVNC usually sits behind:

SSH tunnel

security group

auth proxy

You typically do not want VNC-level auth.

change the /etc/systemd/system/x11vnc-ubuntu.service

```txt
ExecStart=/usr/bin/x11vnc -auth guess -display :0 -rfbport 5900 -shared -nopw
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart x11vnc-ubuntu
```

Another issue

1️⃣ -auth guess failed

Wayland running not X11

```bash
loginctl show-session $(loginctl | grep ubuntu | awk '{print $1}') -p Type
Type=wayland

Type=tty
```

```bash
sudo vim /etc/gdm3/custom.conf
```

```txt
[daemon]
# Uncomment and force Xorg
WaylandEnable=false
AutomaticLoginEnable=true
AutomaticLogin=ubuntu
```


2️⃣ netstat: not found (minor, but annoying)

x11vnc sometimes tries to call netstat.
Install it to remove noise:

```bash
sudo apt update
sudo apt install -y net-tools
```

3️⃣ Once Xorg is running, test manually

```bash
sudo -u ubuntu DISPLAY=:0 x11vnc -auth guess -nopw
```

```bash
journalctl -u x11vnc-ubuntu -b --no-pager | tail -20
```

```bash
systemctl status x11vnc-ubuntu --no-pager
```


## Connect to instance

```bash
curl ifconfig.me
```
54.198.39.94

http://<instance-public-ip>:6080/vnc.html

http://44.204.96.88:6080/vnc.html

## Add Clipboard Support

None of these worked

```bash
xclip -o
sudo apt install xclip
```

```bash
sudo apt install xfce4 xfce4-goodies
```

1️⃣ Clipboard support in x11vnc

x11vnc must be started with clipboard sharing enabled. There are two main options:

```bash
-xfixes          # needed for reliable clipboard updates
-clipboard       # enables clipboard sharing
```

Inside systemd service for x11vnc should be updated like this:

```bash
sudo vim /etc/systemd/system/x11vnc-ubuntu.service
```


```ini
[Unit]
Description=x11vnc Service
After=display-manager.service
Requires=display-manager.service

[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -auth guess -display :0 -rfbport 5900 -shared -nopw -xfixes -clipboard
Restart=always
RestartSec=5

[Install]
WantedBy=graphical.target
```

Then restart the service

```bash
sudo systemctl daemon-reload
sudo systemctl restart x11vnc-ubuntu
```


2️⃣ noVNC clipboard

noVNC has clipboard integration by default, but it only works if:

- The browser supports it (all modern browsers do)
- x11vnc has -clipboard enabled
- You use the correct noVNC page: /vnc.html

Test it:

1. Connect to noVNC (http://localhost:6080/vnc.html)

2. Copy text on your local machine

3. Paste in the remote desktop (e.g., gedit or terminal)

4. Copy text from remote → paste locally

Tip: Browser may ask for clipboard access permission — allow it.


3️⃣ Optional: Clipboard options

x11vnc supports extra options to fine-tune clipboard:

-noxrecord → avoids freezing on heavy clipboard use

-forever → keeps server alive if client disconnects

-acceptclipboard → allow clipboard from client to server only

-sendclipboard → allow clipboard from server to client only