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


## EDID (Extended Display Identification Data)

Why are EDID necessary in VMs. Without a physical monitor, graphics cards may disable output or default to a very low resolution. EDID tricks the system into thinking a display is connected.

Article as example:

[Why EDID Is Important for KVM Video Transmission？](https://www.kinankvm.com/blog/why-edid-is-important-for-kvm-video-transmission%EF%BC%9F/)

```txt
00 ff ff ff ff ff ff 00 10 ac 7e 40 4c 54 41 41
15 17 01 03 80 3c 22 78 ea 4b b5 a7 56 4b a3 25
0a 50 54 a5 4b 00 81 00 b3 00 d1 00 71 4f a9 40
81 80 d1 c0 01 01 56 5e 00 a0 a0 a0 29 50 30 20
35 00 55 50 21 00 00 1a 00 00 00 ff 00 47 4b 30
4b 44 33 35 4f 41 41 54 4c 0a 00 00 00 fc 00 44
45 4c 4c 20 55 32 37 31 33 48 4d 0a 00 00 00 fd
00 31 56 1d 71 1c 00 0a 20 20 20 20 20 20 00 6f

```

00 ff ff ff ff ff ff 00: Header

10 ac: Vendor ID (Dell)

7e 40: Product ID

44 45 4c 4c 20 55 32 37 31 33 48 4d: "DELL U2713HM" (ASCII)

0a: Line Feed (end of string)

There is more to explore for sure

## Clipboard on noVNC and TIgerVNC

Desktop environment requirements. Clipboard will not work on:

- Minimal X11
- Headless Xorg without clipboard manager

```bash
sudo apt install xfce4 xfce4-goodies
```

or you can install a clipboard daemon:

```bash
sudo apt install xfce4-clipman
```

Note: Some browsers block clipboard access unless initiated by user action.

Check if the clipboard is wire on the host machine

```bash
xclip -o
```

noVNC server config

```bash
novnc_proxy --vnc localhost:5901 --listen 6080
```

Clipboard is enabled by default, but if you’re embedding noVNC or using custom flags, ensure:

clipboard: true

enableClipboard: true (older configs)

x11vnc does NOT sync clipboard by default.

```bash
x11vnc -display :0 -forever -shared -clipboard
```

if still doesnt work

```bash
x11vnc -display :0 -forever -shared -clipboard -noxfixes
```

TightVNC / TigerVNC

Clipboard sync depends on X11 selection support.

```bash
sudo apt install autocutsel xclip
```

Then start:

```bash
autocutsel -fork
```


