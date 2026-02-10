# Putting Together Script to Install 

Must install lerobot + isaac-gr00t + leisaac:

## CLI command steps to install

Steps to install Huggingface CLI + lerobot + GR00T + leisaac, altogether just need to build a script that retains state and verify each step.

```bash
# Entrypoint
# Make sure it is outisde any conda env and at home
cd ~
conda deactivate

# Install Huggingface CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Install LeRobot
cd ~ # unnecessary but make sure it is at home
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

# Install Flash Attention --- runs in 55 minutes can cause OOM in some machines
pip install flash_attn --no-build-isolation

# Run this if necessary
pip install -e . --no-build-isolation

# Install leisaac
# Make sure to deactivate
conda deactivate
# Make sure to be on home directory
cd ~

# LeIsaac needs to be installed using isaaclab.sh
cd IsaacLab
./isaaclab.sh --conda leisaac
conda activate leisaac
# Install all the libraries in IsaacLab
isaaclab -i

# Clone and Install leisaac
cd ~
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac

pip install -e source/leisaac
pip install pynput pyserial deepdiff feetech-servo-sdk
```

The website recommends using the pip install of IsaacLab

```bash
# Need review -- flash_attn giving issues
# It takes too long to compile flash attention

# Install Flash Attention limited parallel jobs
MAX_JOBS=4 pip install flash_attn --no-build-isolation

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


## Dependencies


"torch==2.7.0" changed to "torch==2.7.1",

"torchvision==0.22.0" changed to "torchvision==0.22.1",

