# Setting up brev instance with noVNC

Run the Following scripts and copy to the machine the files so it install noVNC and several dependencies.


## Step-1 --- configure the machine VNC + noVNC + base dependencies

setup-novnc.bash

Must add to the script, make sure the files are copied or exist on the target machine

Instance Name: test-g6e-8xlarge-f0631e

```bash
scp setup-novnc_v3.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-f0631e:~
```

Make sure it is executable

```bash
chmod +x setup-novnc_v3.sh
```

## Install Vim and chromium-browser

```bash
sudo apt update
sudo apt install vim
sudo apt install chromium-browser
```

## Connect to instance

```bash
curl ifconfig.me
```

3.80.167.103

http://<instance-public-ip>:6080/vnc.html

http://3.80.167.103:6080/vnc.html