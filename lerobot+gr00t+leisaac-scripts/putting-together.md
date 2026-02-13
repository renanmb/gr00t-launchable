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

## Old stuff

```bash
scp environment.yml test-g6e-8xlarge-679b5d:~
conda env update --file environment.yml --prune

conda env create -f environment.yml
conda activate isaaclab
```



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