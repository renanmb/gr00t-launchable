# Install Remote Desktop 


## noVNC

[noVNC: HTML VNC client library and application](https://github.com/novnc/noVNC)

Prerequisites (PIP packages)

```bash
pip3 install pexpect
```

Install x11vnc and helper util

```bash
sudo apt install -y xfce4 xfce4-goodies dbus-x11 x11vnc expect
```

Add x11vnc systemd config

- template: src=x11vnc-ubuntu.service
- dest=/etc/systemd/system
- mode=0444
- owner=root
- group=root

```bash
#!/usr/bin/env bash
set -e

SRC="x11vnc-ubuntu.service"
DEST="/etc/systemd/system/x11vnc-ubuntu.service"

# Copy the service file
install -o root -g root -m 0444 "$SRC" "$DEST"

# Reload systemd so it sees the new unit
systemctl daemon-reload
```



## Tiger VNC


```bash
sudo apt install -y xfce4 xfce4-goodies
```

```bash
sudo apt install -y tigervnc-standalone-server tigervnc-common
```


```bash
mkdir -p ~/.vnc
```

```bash
vim ~/.vnc/xstartup
```

```txt
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4 &
```

```bash
chmod +x ~/.vnc/xstartup
```

```bash
vncserver -localhost no
```

Create a Tunnel

```bash
ssh -i key.pem -N -L 5901:localhost:5901 ubuntu@EC2_PUBLIC_IP
```

localhost:5901


```bash
vncserver -list
vncserver -kill :1
vncserver :1
```


