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

**Attention**: This command takes too long and causes CPU to run at 100%

```bash
pip install flash-attn==2.7.1.post4 --no-build-isolation 
```

After running that single pip install line, this seems to work best to install flash-attn:

```bash
pip install -e . --no-build-isolation
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

```bash
# Install Huggingface CLI
curl -LsSf https://hf.co/cli/install.sh | bash

cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot

```