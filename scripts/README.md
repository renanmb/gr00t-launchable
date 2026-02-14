# Keep here the scripts

Start locking the scripts that work

Instance Name: 

```bash
scp setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-65964e:~
```

```bash
chmod +x setup-novnc.sh setup-conda.sh setup-isaacsim.sh setup-isaaclab.sh
```

Delete and change the pyproject.toml

```bash
# Delete original pyproject.toml
sudo rm -f ~/Isaac-GR00T/pyproject.toml

# Change pyproject.toml
scp pyproject.toml test-g6e-8xlarge-f0631e:~/Isaac-GR00T
```

Copy the Modality.json

```bash
# Get sample Dataset leisaac-pick-orange
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./demo_data/leisaac-pick-orange

# Copy the Modality.json to leisaac-pick-orange/meta
scp so100_dualcam__modality.json test-g6e-8xlarge-65964e:~/Isaac-GR00T/demo_data/leisaac-pick-orange/meta/modality.json
```

