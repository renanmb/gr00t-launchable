# Install LeIsaac+Lerobot+gr00t

Installing LeIsaac and Lerobot and gr00t have some conflicting dependencies.

remember to connect to the VM GUI

```bash
curl ifconfig.me
```

Example

54.198.207.114

http://<instance-public-ip>:6080/vnc.html

http://54.198.207.114:6080/vnc.html


## Install Hugginface CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Review: make sure the installed CLI is acessible by all users.


## Install lerobot+gr00t

**Step 1**: Install the dependencies lerobot

The gr00t project depends on the lerobot project

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```


```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

This step will ask for additional install of dependencies

```bash
conda install ffmpeg
```

```bash
pip install --no-binary=av -e .
```

This has an error because missing several packages

```bash
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [29 lines of output]
      Package libavformat was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libavformat.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libavformat' found
      Package libavcodec was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libavcodec.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libavcodec' found
      Package libavdevice was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libavdevice.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libavdevice' found
      Package libavutil was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libavutil.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libavutil' found
      Package libavfilter was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libavfilter.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libavfilter' found
      Package libswscale was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libswscale.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libswscale' found
      Package libswresample was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libswresample.pc'
      to the PKG_CONFIG_PATH environment variable
      No package 'libswresample' found
      pkg-config could not find libraries ['avformat', 'avcodec', 'avdevice', 'avutil', 'avfilter', 'swscale', 'swresample']
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'av' when getting requirements to build wheel
```

NOTE: If you encounter build errors, you may need to install additional dependencies (cmake, build-essential, and ffmpeg libs). On Linux, run:

```bash
sudo apt install cmake build-essential python-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config
```

Error Messages

Error: Package 'python-dev' has no installation candidate

```bash
sudo apt install -y cmake build-essential pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config
```

**Step 2**: install gr00t

```bash
cd ~
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

**TODO** need to perform changes

make sure to be outside the other conda environment

```bash
conda deactivate
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

need to check the issue with flash attention and build isolation

Also [base] seems not sufficient

```bash
pip install --no-build-isolation flash-attn==2.7.1.post4
pip install -e . --no-build-isolation
```

```bash
pip install psutil
pip install flash_attn --no-build-isolation
```

Maybe need to install these first:

flash-attn = ["torch==2.7.0", "numpy==1.26.4"]

## This works to install gr00t

```bash
pip install albumentations==1.4.18 av==15.0.0 diffusers==0.35.1 dm-tree==0.1.8 lmdb==1.7.5 msgpack==1.1.0 msgpack-numpy==0.4.8 pandas==2.2.3 peft==0.17.1 termcolor==3.2.0 torch==2.7.0 torchvision==0.22.0 transformers==4.51.3 tyro==0.9.17 click==8.1.8 datasets==3.6.0 einops==0.8.1 gymnasium==1.2.2 matplotlib==3.10.1 numpy==1.26.4 omegaconf==2.3.0 scipy==1.15.3 torchcodec==0.4.0 wandb==0.23.0 pyzmq==27.0.1 deepspeed==0.17.6
```

The gr00t 0.1.0 requires tranformers == 4.53.0

what if install the latest of everything ???

```bash
pip install albumentations av diffusers dm-tree lmdb msgpack msgpack-numpy pandas peft termcolor torch torchvision transformers==4.51.3 tyro click datasets einops gymnasium matplotlib numpy omegaconf scipy torchcodec wandb pyzmq deepspeed
```

**VERY IMPORTANT** --- transformers version

transformers has to be version 4.51.3


**Attention**: This command takes too long and causes CPU to run at 100%

```bash
pip install flash-attn==2.7.1.post4 --no-build-isolation 
```

After running that single pip install line, this seems to work best to install flash-attn:

```bash
pip install -e . --no-build-isolation
```

Debugging the transformers

```bash
python - << 'EOF'
import transformers
print(transformers.__version__)
print(transformers.__file__)
EOF
```


**Install Errors** --- Issue with flash attention

```bash
  error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [17 lines of output]
      /home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/wheel/bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of this command.
        warn(
      Traceback (most recent call last):
        File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
        File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 175, in prepare_metadata_for_build_wheel
          return hook(metadata_directory, config_settings)
        File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/setuptools/build_meta.py", line 378, in prepare_metadata_for_build_wheel
          self.run_setup()
        File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/setuptools/build_meta.py", line 518, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 22, in <module>
      ModuleNotFoundError: No module named 'torch'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> flash-attn

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```



Important Configuration for Single GPU

If you're using a single GPU for fine-tuning, you need to modify /Isaac-GR00T/gr00t/data/dataset/factory.py 

Change torch.distributed.barrier() to:

```python
import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
dist.barrier()
```

### gr00t Python dependencies

All the dependecies gr00t pyproject.toml


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
    "torch==2.7.0",
    "torchvision==0.22.0",
    "transformers==4.51.3",
    "tyro==0.9.17",
    "flash-attn==2.7.4.post1",
    "click==8.1.8",
    "datasets==3.6.0",
    "einops==0.8.1",
    "gymnasium==1.2.2",
    "matplotlib==3.10.1",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "scipy==1.15.3",
    "torchcodec==0.4.0",
    "wandb==0.23.0",
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

All dependencies I wanna

```txt
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
    "torch==2.7.0",
    "torchvision==0.22.0",
    "transformers==4.51.3",
    "tyro==0.9.17",
    "flash-attn==2.7.4.post1",
    "click==8.1.8",
    "datasets==3.6.0",
    "einops==0.8.1",
    "gymnasium==1.2.2",
    "matplotlib==3.10.1",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "scipy==1.15.3",
    "torchcodec==0.4.0",
    "wandb==0.23.0",
    "pyzmq==27.0.1",
    "deepspeed==0.17.6",
    "flash-attn==2.7.1.post4",
]
flash-attn = ["torch==2.7.0", "numpy==1.26.4"]
```


## Install LeIsaac

Source: [LeIsaac](https://github.com/LightwheelAI/leisaac)

Step 1: Install the dependencies leisaac

```bash
# Create and activate environment
conda create -n leisaac python=3.10
conda activate leisaac
```

```bash
# Install cuda-toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

```bash
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

**Important:** Isaaclab must be installed

Install LeIsaac into the same environment that contains lerobot, isaac-gr00t, isaaclab

```bash
cd ~
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac
```

```bash
pip install -e source/leisaac
pip install pynput pyserial deepdiff feetech-servo-sdk
```

# LeIsaac

It has implementation to collect the dataset in te HDF5 format.

## Collect Data

Using the LeIsaac, the idea is to collect the dataset using teleoperation using the leader real robot with the simulation.

Connect the SO-ARM101 leader to an Ubuntu computer via USB cable, then use commands to grant serial port permissions.

**TODO** need review

```bash
ls /dev/ttyACM*
sudo chmod 666 /dev/ttyACM0
```

Run teleoperation tasks with the following script to collect dataset:

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
--task=LeIsaac-SO101-PickOrange-v0 \
--teleop_device=so101leader \
--port=/dev/ttyACM0 \
--num_envs=1 \
--device=cpu \
--enable_cameras \
--record \
--dataset_file=./datasets/dataset.hdf5
```

After entering the IsaacLab window, press the b key on your keyboard to start teleoperation.

You can then use the specified teleop_device to control the robot in the simulation.

If you need to reset the environment after completing your operation, simply press the **r** or **n** key. **r** means resetting the environment and marking the task as failed, while **n** means resetting the environment and marking the task as successful.

## Dataset Replay

After teleoperation, you can replay the collected dataset in the simulation environment using the following script:

**TODO** need review

```bash
python scripts/environments/teleoperation/replay.py \
--task=LeIsaac-SO101-PickOrange-v0 \
--num_envs=1 \
--device=cpu \
--enable_cameras \
--dataset_file=./datasets/dataset.hdf5 \
--episode_index=0
```




# Lerobot + gr00t 

## Example Datasets

- [youliangtan/so100-table-cleanup](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fyouliangtan%2Fso100-table-cleanup%2Fepisode_0)
- [youliangtan/so100_strawberry_grape](https://huggingface.co/datasets/youliangtan/so100_strawberry_grape/tree/main)
- [izuluaga/finish_sandwich](https://huggingface.co/datasets/izuluaga/finish_sandwich)
- [LightwheelAI/leisaac-pick-orange-v0](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0/tree/main)


## Prepare Datasets

Download Datasets from Hugginfacpython scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich --root examples/SO100/finish_sandwich_lerobote

```bash
hf download --repo-type dataset youliangtan/so101-table-cleanup --local-dir ./demo_data/so101-table-cleanup
hf download --repo-type dataset izuluaga/finish_sandwich --local-dir ./demo_data/finish_sandwich
# first test with the Pick Orange
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./demo_data/leisaac-pick-orange
```

The modality.json needs some work:

```bash
# table cleanup
cp getting_started/examples/so100_dualcam__modality.json ./demo_data/so101-table-cleanup/meta/modality.json
# example
cp getting_started/examples/so100__modality.json ./demo_data/<DATASET_PATH>/meta/modality.json
# finish sandwich
cp examples/SO100/modality.json examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich/meta/modality.json
```

## LeRobot data conversion

Convert dataset from lerobot version3 to version 2

```bash
python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich --root examples/SO100/finish_sandwich_lerobot
```

## Fine Tunning model

```bash
export NUM_GPUS=1
python gr00t/experiment/launch_finetune.py --base_model_path nvidia/GR00T-N1.6-3B --dataset_path examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus $NUM_GPUS --output_dir /tmp/so100_finetune --save_steps 1000 --save_total_limit 5 --max_steps 10000 --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --no-use_wandb --global_batch_size 32 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader_num_workers 4 --no-tune-diffusion-model
```

```bash
python gr00t/experiment/launch_finetune.py --base_model_path nvidia/GR00T-N1.6-3B --dataset_path examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir /tmp/so100_finetune --save_steps 1000 --save_total_limit 5 --max_steps 10000 --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --no-use_wandb --global_batch_size 2 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader_num_workers 4 --no-tune-diffusion-model
```

```bash
python gr00t/experiment/launch_finetune.py --base_model_path nvidia/GR00T-N1.6-3B --dataset_path demo_data/leisaac-pick-orange --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir /tmp/so100_finetune --save_steps 1000 --save_total_limit 5 --max_steps 10000 --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --no-use_wandb --global_batch_size 2 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader_num_workers 4 --no-tune-diffusion-model
```

Tip: The default fine-tuning settings require ~25G of VRAM.

If you don't have that much VRAM, try adding the --no-tune_diffusion_model flag to the gr00t_finetune.py script.


Very Important

If you're using a single GPU for fine-tuning, you need to modify /Isaac-GR00T/gr00t/data/dataset/factory.py 

Change torch.distributed.barrier() to:

```python
import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
dist.barrier()
```

## Open-loop Evaluation

Once the training is complete and your fine-tuned policy is generated, you can visualize its performance in an open-loop setting by running the following command:

```bash
python gr00t/eval/open_loop_eval.py --dataset-path examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich --embodiment-tag NEW_EMBODIMENT --model-path /tmp/so100_finetune/checkpoint-10000 --traj-ids 0 --action-horizon 16 --steps 400
```

```bash
python gr00t/eval/open_loop_eval.py --dataset-path demo_data/leisaac-pick-orange --embodiment-tag NEW_EMBODIMENT --model-path /tmp/so100_finetune/checkpoint-10000 --traj-ids 0 --action-horizon 16 --steps 400
```

Congratulations! You have successfully finetuned GR00T-N1.6 on a new embodiment.

Plot and save trajectory results comparing ground truth and predicted actions.


Need to update the Arguments used to match the current script.

Args:

- **state_joints_across_time**: Array of state joints over time
- **gt_action_across_time**: Ground truth actions over time
- **pred_action_across_time**: Predicted actions over time
- **traj_id**: Trajectory ID
- **state_keys**: List of state modality keys
- **action_keys**: List of action modality keys
- **action_horizon**: Action horizon used for inference
- **save_plot_path**: Path to save the plot

## Deployment Option

After successfully fine-tuning and evaluating your policy, the final step is to deploy it onto your physical robot for real-world execution.

To connect your SO-101 robot and begin the evaluation, execute the following commands in your terminal:

1. First Run the Policy as a server:

```bash
python scripts/inference_service.py --server --model_path ./so101-checkpoints --embodiment-tag new_embodiment --data-config so100_dualcam --denoising-steps 4
```

2. On a separate terminal, run the eval script as client. Make sure to update the port and id for the robot, as well as the index and parameters for cameras, to match your configuration.

```bash
python getting_started/examples/eval_lerobot.py --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" --policy_host=10.112.209.136 --lang_instruction="Grab pens and place into pen holder."
```

# Putting it all together

Steps to install Huggingface CLI + lerobot + GR00T + leisaac, altogether just need to build a script that retains state and verify each step.

```bash
# Install Huggingface CLI
curl -LsSf https://hf.co/cli/install.sh | bash
# Install LeRobot
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot

conda create -y -n lerobot python=3.10
conda activate lerobot

conda install ffmpeg

pip install --no-binary=av -e .

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
# Install Flash Attention
pip install flash_attn --no-build-isolation
# Run this if necessary
pip install -e . --no-build-isolation
# Make sure to deactivate
conda deactivate
# Make sure to be on home directory
cd ~

# Install leisaac
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



# gr00t Issues and solutions

As mentined in the github issue:

- [maybe not bump transformers from 4.51.3 to 4.53.0 #513](https://github.com/NVIDIA/Isaac-GR00T/issues/513)

THere are some library issues and dependencies. 

I set up the environment for the model server but failed to make it run with following error:

```python
from transformers.image_utils import ImageInput, VideoInput, get_image_size, to_numpy_array ImportError: cannot import name 'VideoInput' from 'transformers.image_utils' (/home/hc81/Isaac-GR00T/.venv/lib/python3.10/site-packages/transformers/image_utils.py)
```

but after i reset it back to 4.51.3, everything just work out fine.

Below is a detailed list of all packages and Versions that work as of 01/31/2026

```txt
Python 3.10.19
accelerate==1.12.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
albucore==0.0.17
albumentations==1.4.18
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
async-timeout==5.0.1
attrs==25.4.0
av==15.0.0
certifi==2026.1.4
charset-normalizer==3.4.4
click==8.1.8
cloudpickle==3.1.2
contourpy==1.3.2
cycler==0.12.1
datasets==3.6.0
deepspeed==0.17.6
diffusers==0.35.1
dill==0.3.8
dm-tree==0.1.8
docstring_parser==0.17.0
einops==0.8.1
eval_type_backport==0.3.1
Farama-Notifications==0.0.4
filelock==3.20.3
flash_attn==2.7.4.post1
fonttools==4.61.1
frozenlist==1.8.0
fsspec==2026.1.0
gitdb==4.0.12
GitPython==3.1.46
-e git+https://github.com/NVIDIA/Isaac-GR00T@d331b68ee8603be3f07167826d944f458d15a691#egg=gr00t
gymnasium==1.2.2
hf-xet==1.2.0
hjson==3.1.0
huggingface-hub==0.36.0
idna==3.11
ImageIO==2.37.2
importlib_metadata==8.7.1
Jinja2==3.1.6
kiwisolver==1.4.9
lazy_loader==0.4
lmdb==1.7.5
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.1
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.1.0
msgpack-numpy==0.4.8
multidict==6.7.0
multiprocess==0.70.16
networkx==3.4.2
ninja==1.13.0
numpy==2.2.6
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
omegaconf==2.3.0
opencv-python-headless==4.11.0.86
packaging==26.0
pandas==2.2.3
peft==0.17.1
pillow==12.1.0
platformdirs==4.5.1
propcache==0.4.1
protobuf==6.33.4
psutil==7.2.1
py-cpuinfo==9.0.0
pyarrow==23.0.0
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.19.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
pyzmq==27.0.1
regex==2026.1.15
requests==2.32.5
rich==14.2.0
safetensors==0.7.0
scikit-image==0.25.2
scipy==1.15.3
sentry-sdk==2.50.0
shtab==1.8.0
six==1.17.0
smmap==5.0.2
sympy==1.14.0
termcolor==3.2.0
tifffile==2025.5.10
tokenizers==0.21.4
torch==2.7.0
torchcodec==0.4.0
torchvision==0.22.0
tqdm==4.67.1
transformers==4.51.3
triton==3.3.0
typeguard==4.4.4
typing-inspection==0.4.2
typing_extensions==4.15.0
tyro==0.9.17
tzdata==2025.3
urllib3==2.6.3
wandb==0.23.0
xxhash==3.6.0
yarl==1.22.0
zipp==3.23.0
```
