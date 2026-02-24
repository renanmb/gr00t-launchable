# Keep here the scripts

Start locking the scripts that work

Instance Name: 

test-g6e-8xlarge-51b10e

```bash
scp installer_v0.sh setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh setup-gr00t.sh setup-leisaac.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-51b10e:~
```

```bash
chmod +x installer_v0.sh setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh setup-gr00t.sh setup-leisaac.sh
```


> [!IMPORTANT]  
> Before Running ```setup-gr00t.sh``` make sure to have the new ```pyproject.toml``` at the home folder.

The [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) has some dependencies that need to pay attention so it works with LeRobot.

```bash
scp pyproject.toml test-g6e-8xlarge-51b10e:~
```

The script ```setup-gr00t.sh``` should ahndle it automatically but you can also manually delete and change the pyproject.toml

```bash
# Delete original pyproject.toml
sudo rm -f ~/Isaac-GR00T/pyproject.toml

# Change pyproject.toml
scp pyproject.toml test-g6e-8xlarge-2eb89f:~/Isaac-GR00T
```

Test the Isaac-GR00T Training

Copy the Modality.json into the folder: ```~/Isaac-GR00T/demo_data/leisaac-pick-orange/meta/```

```bash
# Get sample Dataset leisaac-pick-orange
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./demo_data/leisaac-pick-orange

# Copy the Modality.json to leisaac-pick-orange/meta
scp so100_dualcam__modality.json test-g6e-8xlarge-51b10e:~/Isaac-GR00T/demo_data/leisaac-pick-orange/meta/modality.json
```

Connect to instance using VNC must first get the IP:

```bash
curl ifconfig.me
```

test-g6e-8xlarge-51b10e

3.81.117.20

http://<instance-public-ip>:6080/vnc.html

http://3.81.117.20:6080/vnc.html


## Notes

The LeIsaac script needs attention to which repo using since it needs to decouple LEROBOT dependency

The GR00T repo needs to make sure the pyproject.toml is correct. Might evaluate using a custom repo.