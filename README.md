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

174.129.81.179

http://<instance-public-ip>:6080/vnc.html

http://174.129.81.179:6080/vnc.html

http://34.160.111.145:6080/vnc.html

## Step-1 --- configure the machine VNC + noVNC + base dependencies

Copy the files to the machine

```bash
scp setup-novnc_v3.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-65964e:~
```

Make sure it is executable

```bash
chmod +x setup-novnc_v3.sh
```

## Step-2 --- Install Conda + IsaacSim + IsaacLab

TODO: must add a checkout command to guarantee it will checkout the tag for the right version

Python = 3.10

IsaacSim = 5.1

isaaclab = 2.3.0

| Dependency | Isaac Sim 5.1 |
| :--- | :---: |
| **Python** | 3.11 |
| **Isaac Lab** | v2.3.0 |
| **CUDA** | 12.8 |
| **PyTorch** | 2.7.0 |


```bash
scp install-conda_v2.sh isaacsim_v2.sh isaaclab_v3.sh test-g6e-8xlarge-65964e:~
```

Make sure it is executable

```bash
chmod +x install-conda_v2.sh isaacsim_v2.sh isaaclab_v3.sh
```


## Step-3 --- Install GR00T

Current Issues:

- flash_attn == 2.7.4.post1 (gr00t) only installed on gr00t env
- transformer == 4.51.3 (gr00t) for some reason transformers == 4.57.3 (isaaclab/leisaac)
- lerobot == 0.4.2

```bash
# Make sure it is outisde any conda env
conda deactivate

# Install Huggingface CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Install LeRobot and Gr00t same environment
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout v0.4.3
conda create -y -n gr00t python=3.10
conda activate gr00t

# conda install ffmpeg --- Had isues with Torchcodec 
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev libavdevice-dev
conda install -y ffmpeg -c conda-forge

# Install lerobot in editable mode
pip install -e . 

# Clone and Install GR00T
cd ~
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# Change pyproject.toml 

# Install Isaac-GR00T source in editable mode
pip install -e . --no-build-isolation
```

Change the pyproject.toml inside Isaac-GR00T

```bash
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

## Step-4 --- Install LeIsaac

Install LeIsaac from source according to their docs.

```bash
git clone https://github.com/LightwheelAI/leisaac.git --recursive
cd leisaac
git checkout v0.3.0

# Create and activate environment
conda create -y -n leisaac python=3.11
conda activate leisaac

# Install cuda-toolkit
conda install -y -c "nvidia/label/cuda-12.8.1" cuda-toolkit

# Install PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install IsaacSim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Verify IsaacSim Installation
# note: you can pass the argument "--help" to see all arguments possible.
isaacsim

# Install IsaacLab
sudo apt install cmake build-essential

cd ~

cd leisaac/dependencies/IsaacLab
./isaaclab.sh --install

cd ../..
pip install -e source/leisaac
```



## Test Isaac-GR00T

Download the sample dataset LightwheelAI/leisaac-pick-orange.

```bash
# Get sample Dataset leisaac-pick-orange
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./demo_data/leisaac-pick-orange
```

Must add so100_dualcam__modality.json at the sample dataset LightwheelAI/leisaac-pick-orange

```bash
# Copy the Modality.json to leisaac-pick-orange/meta
scp so100_dualcam__modality.json test-g6e-8xlarge-f0631e:~/Isaac-GR00T/demo_data/leisaac-pick-orange/meta/modality.json
```

### Finetuning Example

Finetuning for dataset LightwheelAI/leisaac-pick-orange.

It will output at: ./tmp/so100_finetune_orange 

typically it would output like at this path: /tmp/so100_finetune/checkpoint-10000

```bash
python gr00t/experiment/launch_finetune.py --base_model_path nvidia/GR00T-N1.6-3B --dataset_path demo_data/leisaac-pick-orange/ --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir ./tmp/so100_finetune_orange --save_steps 1000 --save_total_limit 5 --max_steps 10000 --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --no-use_wandb --global_batch_size 2 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader_num_workers 4 --no-tune-diffusion-model
```

### Open-Loop Evaluation

```bash
python gr00t/eval/open_loop_eval.py --dataset-path demo_data/leisaac-pick-orange/ --embodiment-tag NEW_EMBODIMENT --model-path tmp/so100_finetune_orange/checkpoint-10000 --traj-ids 0 --action-horizon 16 --steps 400
```

It will output the results at: /tmp/open_loop_eval

```bash
cd /tmp/open_loop_eval
```

TODO: modify the output directory

### Isaac-GR00T Client server

Start Isaac-GR00T Client server


```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path ./tmp/so100_finetune_orange/checkpoint-10000 --device cuda:0 --host 127.0.0.0 --port 5555 --strict
```

Prefer --host 127.0.0.0 --port 5555 sometimes says 127.0.0.1 is busy

```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path ./tmp/so100_finetune_orange/checkpoint-10000 --device cuda:0 --host 127.0.0.1 --port 5555 --strict
```

## Testing leisaac

Asset Preparation

Lightwheel provides an example USD asset (a kitchen scene). Please download related scene [here](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0) and extract it into the `assets` directory. The directory structure should look like this:

```
<assets>
├── robots/
│   └── so101_follower.usd
└── scenes/
    └── kitchen_with_orange/
        ├── scene.usd
        ├── assets
        └── objects/
            ├── Orange001
            ├── Orange002
            ├── Orange003
            └── Plate
```

Links

- https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0
- https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip


Run Policy inference

```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.0 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```

Sometimes it tells that the --policy_host=127.0.0.1 --policy_port=5555 is busy

```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.1 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```



**Optional** - Train a robot!

Train Ant

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

Run inference:

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Ant-v0 --num_envs 32
```

Train Quadruped

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

Run inference:

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 32
```

## TODO

Add chromium, make sure vim is installed, install nvtop


## Ports required open

**SSH**

- Protocol = TCP
- port = 22

**nomachine**

- Protocol = TCP
- port = 4000

- Protocol = UDP
- port = 4000

**VNC server**

- Protocol = TCP
- port = 5900

**noVNC**

- Protocol = TCP
- port = 6080

**Outbound Traffic**

- Protocol = TCP
- port = 443

- Protocol = TCP
- port = 80

- Protocol = TCP
- port = 53

- Protocol = UDP
- port = 53

- Protocol = UDP
- port = 123