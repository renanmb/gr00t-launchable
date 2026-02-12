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

54.198.161.130

http://<instance-public-ip>:6080/vnc.html

http://54.198.161.130:6080/vnc.html

## Step-1 --- configure the machine VNC + noVNC + base dependencies

setup-novnc.bash

Must add to the script, make sure the files are copied or exist on the target machine

```bash
scp setup-novnc.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-65964e:~
```

Make sure it is executable

```bash
chmod +x setup-novnc.sh
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

install-conda_v2.sh

isaacsim_v2.sh

isaaclab_v2.sh

```bash
scp install-conda_v2.sh isaacsim_v2.sh isaaclab_v2.sh test-g6e-8xlarge-65964e:~
# Trying to run isaaclab -i
scp install-conda_v2.sh isaacsim_v2.sh isaaclab_v3.sh test-g6e-8xlarge-65964e:~
```

Make sure it is executable

```bash
chmod +x install-conda_v2.sh isaacsim_v2.sh isaaclab_v2.sh

chmod +x install-conda_v2.sh isaacsim_v2.sh isaaclab_v3.sh
```

Note:

The script ```isaaclab_v2.sh``` needs change to run ```isaaclab -i``` so it finishes the installation

## Step-3 --- Install Lerobot + GR00T + Leisaac

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
# git checkout v0.4.2
git checkout v0.4.3

# conda create -y -n lerobot python=3.10
conda create -y -n gr00t python=3.10
# conda create -y -n lerobot python=3.11 # trying 3.11

# conda activate lerobot
conda activate gr00t

# conda install ffmpeg
conda install -y ffmpeg -c conda-forge # on lerobot docs

pip install -e . # docs recommend this
# pip install --no-binary= av -e .

# This will break evertyhing
# sudo apt install -y cmake build-essential pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config


# Clone and Install GR00T
cd ~
# conda deactivate
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# Change pyproject.toml

pip install -e . --no-build-isolation

# conda create -y -n gr00t python=3.10
# conda activate gr00t
# pip install --upgrade setuptools # lerobot requires setuptools-80.10.2
# pip install --upgrade setuptools==80.10.2

# pip install psutil # check with version I require 7.2.2 seems fine

# Install dependencies transformers==4.51.3 -- this step breaks stuff
# pip install albumentations av diffusers dm-tree lmdb msgpack msgpack-numpy pandas peft termcolor torch==2.7.0 torchvision transformers==4.51.3 tyro click datasets einops gymnasium matplotlib numpy omegaconf scipy torchcodec wandb pyzmq deepspeed

# Need review -- flash_attn giving issues
# It takes too long to compile flash attention
# Install Flash Attention limited parallel jobs
# MAX_JOBS=4 pip install flash_attn --no-build-isolation

# Install Flash Attention 
# pip install flash_attn --no-build-isolation

# Run this if necessary
pip install -e . --no-build-isolation

# INSTALL LEISAAC

# Make sure to deactivate
conda deactivate
# Make sure to be on home directory
cd ~

cd IsaacLab
./isaaclab.sh --conda leisaac
conda activate leisaac
./isaaclab.sh -i


cd ~
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac

pip install -e source/leisaac
pip install pynput pyserial deepdiff feetech-servo-sdk

# Install lerobot inside the leisaac environment
cd lerobot
git checkout v0.4.3
conda install -y ffmpeg -c conda-forge # on lerobot docs
pip install -e .

```

Copy the Modality.json

```bash
# Get sample Dataset leisaac-pick-orange
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./demo_data/leisaac-pick-orange
# Copy the Modality.json to leisaac-pick-orange/meta
scp so100_dualcam__modality.json test-g6e-8xlarge-65964e:~/Isaac-GR00T/demo_data/leisaac-pick-orange/meta/modality.json

scp environment.yml test-g6e-8xlarge-679b5d:~
conda env update --file environment.yml --prune

conda env create -f environment.yml
conda activate isaaclab
```

Old Need to review but doesnt work

```bash
# This is wrong need to create the env with Isaalab
# Install leisaac -- This process is wrong
# Create and activate environment
conda create -n leisaac python=3.10
conda activate leisaac
# Install cuda-toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

## TEST

```bash
python gr00t/experiment/launch_finetune.py --base_model_path nvidia/GR00T-N1.6-3B --dataset_path demo_data/leisaac-pick-orange/ --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir ./tmp/so100_finetune_orange --save_steps 1000 --save_total_limit 5 --max_steps 10000 --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --no-use_wandb --global_batch_size 2 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader_num_workers 4 --no-tune-diffusion-model
```

Start server

```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path /tmp/so100_finetune/checkpoint-10000 --device cuda:0 --host 127.0.0.1 --port 5555 --strict
```

```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path /tmp/so100_finetune/checkpoint-10000 --device cuda:0 --host 127.0.0.0 --port 5555 --strict

python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path ./tmp/so100_finetune_orange/checkpoint-10000 --device cuda:0 --host 127.0.0.0 --port 5555 --strict
```

Start leisaac

```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.0 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```

```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.1 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```

Check the trained model:

```bash
cd /tmp/open_loop_eval
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