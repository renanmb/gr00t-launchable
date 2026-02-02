# gr00t-launchable
 
Some important commands:

```bash
source ~/.bashrc
brev ls
```

 
## Connect to instance

```bash
curl ifconfig.me
```
34.235.131.168

http://<instance-public-ip>:6080/vnc.html

http://34.235.131.168:6080/vnc.html

## Step-1 --- configure the machine VNC + noVNC + base dependencies

setup-novnc.bash

Must add to the script, make sure the files are copied or exist on the target machine

```bash
scp setup-novnc.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-a5b412:~
```

Make sure it is executable

```bash
chmod +x setup-novnc.sh
```


## Step-2 --- Install Conda + IsaacSim + IsaacLab

install-conda_v2.sh

isaacsim_v2.sh

isaaclab_v2.sh

```bash
scp install-conda_v2.sh isaacsim_v2.sh isaaclab_v2.sh test-g6e-8xlarge-a5b412:~
```

Make sure it is executable

```bash
chmod +x install-conda_v2.sh isaacsim_v2.sh isaaclab_v2.sh
```

Note:

The script ```isaaclab_v2.sh``` needs change to run ```isaaclab -i``` so it finishes the installation

## Step-3 --- Install Lerobot + GR00T + Leisaac

