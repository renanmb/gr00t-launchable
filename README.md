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

http://54.196.132.167:6080/vnc.html

## Step-1 --- configure the machine VNC + noVNC + base dependencies

setup-novnc.bash

Must add to the script, make sure the files are copied or exist on the target machine

```bash
scp setup-novnc.sh xorg.conf vdisplay.edid x11vnc-ubuntu.service novnc.service test-g6e-8xlarge-3dfe06:~
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
scp install-conda_v2.sh isaacsim_v2.sh isaaclab_v2.sh test-g6e-8xlarge-3dfe06:~
```

Make sure it is executable

```bash
chmod +x install-conda_v2.sh isaacsim_v2.sh isaaclab_v2.sh
```

Note:

The script ```isaaclab_v2.sh``` needs change to run ```isaaclab -i``` so it finishes the installation

## Step-3 --- Install Lerobot + GR00T + Leisaac

Current Issues:

-- flash_attn

```bash
# Make sure it is outisde any conda env
conda deactivate
# Install Huggingface CLI
curl -LsSf https://hf.co/cli/install.sh | bash
# Install LeRobot
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot

conda create -y -n lerobot python=3.10
conda activate lerobot

conda install ffmpeg

pip install --no-binary= av -e .

sudo apt install -y cmake build-essential pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config

# Install GR00T
cd ~
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

conda deactivate
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools

pip install psutil
# Install dependencies transformers==4.51.3
pip install albumentations av diffusers dm-tree lmdb msgpack msgpack-numpy pandas peft termcolor torch torchvision transformers==4.51.3 tyro click datasets einops gymnasium matplotlib numpy omegaconf scipy torchcodec wandb pyzmq deepspeed

# Need review -- flash_attn giving issues
# It takes too long to compile flash attention
# Install Flash Attention limited parallel jobs
MAX_JOBS=4 pip install flash_attn --no-build-isolation

# Install Flash Attention 
pip install flash_attn --no-build-isolation

# Run this if necessary
pip install -e . --no-build-isolation
# Make sure to deactivate
conda deactivate
# Make sure to be on home directory
cd ~

# Install leisaac -- This process is wrong
# Create and activate environment
conda create -n leisaac python=3.10
conda activate leisaac
# Install cuda-toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

cd ~
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac

pip install -e source/leisaac
pip install pynput pyserial deepdiff feetech-servo-sdk
```

## TEST

Start server

```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path /tmp/so100_finetune/checkpoint-10000 --device cuda:0 --host 127.0.0.1 --port 5555 --strict
```

Start leisaac

```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.0 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```