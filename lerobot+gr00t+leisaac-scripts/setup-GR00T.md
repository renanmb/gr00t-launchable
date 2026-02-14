# How to install GR00T

Putting together the process to install Isaac-GR00T. It requires lerobot and several dependencies to properly build flash_attn.

Dependencies:

- python == 3.10
- flash_attn == 2.7.4.post1
- transformer == 4.51.3
- lerobot == 0.4.3
- datasets>=4.0.0,<4.2.0
- wandb>=0.24.0,<0.25.0

Instance Name: test-g6e-8xlarge-f0631e

```bash
# Make sure it is outisde any conda env and at Home
cd ~
conda deactivate

# Install Huggingface CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Install LeRobot and Gr00t at the same environment conda env
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout v0.4.3
conda create -y -n gr00t python=3.10
conda activate gr00t

# conda install ffmpeg
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev libavdevice-dev
conda install -y ffmpeg -c conda-forge

# Install lerobot in editable mode
pip install -e . 

# Clone and Install Isaac-GR00T
cd ~
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# Make sure the pyproject.toml has been changed before running pip install -e
pip install -e . --no-build-isolation
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

Prefer --host 127.0.0.0 --port 5555 sometimes says 127.0.0.1 is busy

```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path ./tmp/so100_finetune_orange/checkpoint-10000 --device cuda:0 --host 127.0.0.1 --port 5555 --strict
```

```bash
python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path ./tmp/so100_finetune_orange/checkpoint-10000 --device cuda:0 --host 127.0.0.0 --port 5555 --strict
```

## Issues with Torchcodec

Check Torchdec version:

```bash
python -c "import torchcodec; print('torchcodec version:', torchcodec.__version__)"
```

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev libavdevice-dev
conda install -c conda-forge ffmpeg
```

## MISC

```bash
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
git clone https://github.com/LightwheelAI/leisaac.git --recursive
cd leisaac
git checkout v0.3.0

pip install -e source/leisaac
pip install pynput pyserial deepdiff feetech-servo-sdk

# Install lerobot inside the leisaac environment
cd lerobot
git checkout v0.4.3
conda install -y ffmpeg -c conda-forge # on lerobot docs
pip install -e .
```

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

# Install IsaacLab
sudo apt install cmake build-essential

cd ~

cd leisaac/dependencies/IsaacLab
./isaaclab.sh --install

cd ../..
pip install -e source/leisaac
```


## Changes to Isaac-GR00T pyproject.toml

- datasets>=4.0.0,<4.2.0
- wandb>=0.24.0,<0.25.0

Isaac-GR00T pyproject.toml

```bash
scp pyproject.toml test-g6e-8xlarge-f0631e:~/Isaac-GR00T
```

```toml
#This is the single project pyproject.toml

[build-system]
requires = ["setuptools>=67", "wheel", "pip"]
build-backend = "setuptools.build_meta"

[project]
name = "gr00t"
version = "0.1.0"
requires-python = "==3.10.*"
# Mirror the main repo's baseline dependencies so install behaves the same.
dependencies = [
    "albumentations==1.4.18",
    "av==15.0.0",
    "diffusers==0.35.1",
    "dm-tree==0.1.8",
    "lmdb==1.7.5",
    "msgpack==1.1.0",
    "msgpack-numpy==0.4.8",
    "pandas==2.2.3",
    "peft==0.17.1",
    "termcolor==3.2.0",
    "torch==2.7.1",
    "torchvision==0.22.1",
    "transformers==4.51.3",
    "tyro==0.9.17",
    "flash-attn==2.7.4.post1",
    "click==8.1.8",
    "datasets>=4.0.0,<4.2.0",
    "einops==0.8.1",
    "gymnasium==1.2.2",
    "matplotlib==3.10.1",
    "numpy>=2",
    "omegaconf==2.3.0",
    "scipy==1.15.3",
    "torchcodec==0.4.0",
    "wandb>=0.24.0,<0.25.0",
    "pyzmq==27.0.1",
    "deepspeed==0.17.6",
]

[project.optional-dependencies]
dev = [
    "ruff",
    "ipython",
]
tensorrt = [
    "onnx>=1.20.0",
    "tensorrt>=10.14.1.48.post1",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["gr00t*"]

[tool.uv.extra-build-dependencies]
flash-attn = ["torch==2.7.0", "numpy==1.26.4"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["gr00t"]
exclude = [
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
    ".venv",
    "dist",
    "logs",
    "*.ipynb",
    "gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2",
    "external_dependencies",
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]

[tool.ruff.lint.isort]
case-sensitive = false
combine-as-imports = true
force-sort-within-sections = true
force-wrap-aliases = false
split-on-trailing-comma = false
lines-after-imports = 2
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
```


## Error tracking for Script

Running the script

setup-gr00t.sh

```bash
>>> Installing LeRobot in editable mode                                                           
                                                                                                  
                                                                                                  
CondaError: Run 'conda init' before 'conda activate' 
```

