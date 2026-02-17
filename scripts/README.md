# Keep here the scripts

Start locking the scripts that work

Instance Name: 

test-g6e-8xlarge-2eb89f

```bash
scp setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh setup-gr00t.sh setup-leisaac.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-2eb89f:~
```

```bash
chmod +x setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh setup-gr00t.sh setup-leisaac.sh
```

Delete and change the pyproject.toml

```bash
# Delete original pyproject.toml
sudo rm -f ~/Isaac-GR00T/pyproject.toml

# Change pyproject.toml
scp pyproject.toml test-g6e-8xlarge-2eb89f:~/Isaac-GR00T

scp pyproject.toml test-g6e-8xlarge-2eb89f:~
```

Copy the Modality.json

```bash
# Get sample Dataset leisaac-pick-orange
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./demo_data/leisaac-pick-orange

# Copy the Modality.json to leisaac-pick-orange/meta
scp so100_dualcam__modality.json test-g6e-8xlarge-2eb89f:~/Isaac-GR00T/demo_data/leisaac-pick-orange/meta/modality.json
```

54.175.214.162

http://<instance-public-ip>:6080/vnc.html

http://54.175.214.162:6080/vnc.html

