# How to install leisaac

This install leisaac using a specific version of isaaclab and a pip installation of IsaacSim


## Step 1 - Installation

```bash
git clone https://github.com/LightwheelAI/leisaac.git --recursive
cd leisaac
git checkout v0.3.0

# Create and activate environment
conda create -y -n leisaac python=3.11 # try 3.10
conda activate leisaac

# Install cuda-toolkit
conda install -y -c "nvidia/label/cuda-12.8.1" cuda-toolkit

# Install PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install IsaacSim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com # try 4.5

# Install IsaacLab
sudo apt install cmake build-essential

cd ~

cd leisaac/dependencies/IsaacLab
./isaaclab.sh --install

cd ../..
pip install -e source/leisaac
```

Experiment making a symbolic link --- This does not work

```bash
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/isaacsim"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# check if it runs as expected
${ISAACSIM_PATH}/isaac-sim.sh${ISAACSIM_PATH}/isaac-sim.sh

# checks that python path is set correctly
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

# enter the cloned repository
cd ~
cd leisaac/dependencies/IsaacLab
# create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

sudo apt install cmake build-essential
./isaaclab.sh --install # Asks for the EULA, need automation

cd ../..
pip install -e source/leisaac
```

For example in the container there is a Env var for Accepting the EULA so it doesnt stop installation

```bash
# Accept the NVIDIA Omniverse EULA by default
ACCEPT_EULA=Y
```


## 2. Asset Preparation

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


## Testing leisaac


```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.0 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```

Sometimes it tells that the --policy_host=127.0.0.1 --policy_port=5555 is busy

```bash
python scripts/evaluation/policy_inference.py --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=10 --policy_type=gr00tn1.6 --policy_host=127.0.0.1 --policy_port=5555 --policy_timeout_ms=5000 --policy_action_horizon=16 --policy_language_instruction="Pick up the orange and place it on the plate" --device=cuda --enable_cameras
```



## Issues

The simulation freezes when launched

remove conda environment

```bash
conda remove -y --name leisaac --all

unlink _isaac_sim

```

Issue when running the recommended source

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
wheel 0.46.3 requires packaging>=24.0, but you have packaging 23.0 which is incompatible.
Successfully installed Pillow-11.3.0 PyJWT-2.11.0 aioboto3-15.1.0 aiobotocore-2.24.0 aiodns-3.1.1 aiofiles-23.2.1 aiohappyeyeballs-2.4.4 aiohttp-3.11.11 aioitertools-0.11.0 aiosignal-1.3.2 annotated-types-0.7.0 anyio-4.12.1 asteval-1.0.6 async_timeout-5.0.1 attrs-25.1.0 awscrt-0.23.8 azure-core-1.28.0 azure-identity-1.13.0 azure-storage-blob-12.17.0 boto3-1.39.11 botocore-1.39.11 certifi-2026.1.4 cffi-2.0.0 charset_normalizer-3.3.2 click-8.1.7 contourpy-1.3.1 coverage-7.4.4 cryptography-44.0.0 cycler-0.11.0 fastapi-0.115.7 filelock-3.13.1 fonttools-4.55.3 frozenlist-1.5.0 fsspec-2024.6.1 gunicorn-23.0.0 h11-0.16.0 httptools-0.6.1 idna-3.10 idna-ssl-1.1.0 imageio-2.37.0 isaacsim-5.1.0.0 isaacsim-app-5.1.0.0 isaacsim-asset-5.1.0.0 isaacsim-benchmark-5.1.0.0 isaacsim-code-editor-5.1.0.0 isaacsim-core-5.1.0.0 isaacsim-cortex-5.1.0.0 isaacsim-example-5.1.0.0 isaacsim-extscache-kit-5.1.0.0 isaacsim-extscache-kit-sdk-5.1.0.0 isaacsim-extscache-physics-5.1.0.0 isaacsim-gui-5.1.0.0 isaacsim-kernel-5.1.0.0 isaacsim-replicator-5.1.0.0 isaacsim-rl-5.1.0.0 isaacsim-robot-5.1.0.0 isaacsim-robot-motion-5.1.0.0 isaacsim-robot-setup-5.1.0.0 isaacsim-ros1-5.1.0.0 isaacsim-ros2-5.1.0.0 isaacsim-sensor-5.1.0.0 isaacsim-storage-5.1.0.0 isaacsim-template-5.1.0.0 isaacsim-test-5.1.0.0 isaacsim-utils-5.1.0.0 isodate-0.6.1 jmespath-1.0.1 kiwisolver-1.4.4 llvmlite-0.42.0 markupsafe-2.1.3 matplotlib-3.10.3 msal-1.27.0 msal-extensions-1.0.0 multidict-6.1.0 nest_asyncio-1.5.6 networkx-3.3 numba-0.59.1 numpy-1.26.0 oauthlib-3.2.2 opencv-python-headless-4.11.0.86 osqp-0.6.7.post3 packaging-23.0 pint-0.20.1 portalocker-2.7.0 propcache-0.2.1 psutil-5.9.8 pycares-4.8.0 pycparser-3.0 pydantic-2.11.10 pydantic-core-2.33.2 pyparsing-3.0.9 pyperclip-1.8.0 pypng-0.20220715.0 python-dateutil-2.9.0.post0 python-multipart-0.0.20 pytz-2024.1 pyyaml-6.0.2 qdldl-0.1.7.post5 qrcode-7.4.2 requests-2.32.3 requests-oauthlib-1.3.1 rtree-1.3.0 s3transfer-0.13.1 scipy-1.15.3 sentry-sdk-2.29.1 six-1.17.0 starlette-0.45.3 sympy-1.13.3 toml-0.10.2 torchaudio-2.7.0 tornado-6.5.1 trimesh-4.5.1 typing-inspection-0.4.2 typing_extensions-4.12.2 urllib3-2.6.3 uvicorn-0.29.0 watchdog-4.0.0 websockets-12.0 wrapt-1.16.0 yarl-1.18.3

```


## Error on 3090 Machine

Latest LeIsaac has done changes that require the lerobot which causes conflicts with IsaacLab


```bash
/home/goat/anaconda3/envs/isaaclab/lib/python3.11/site-packages/pygame/pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists
Traceback (most recent call last):
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 60, in <module>
    from leisaac.utils.env_utils import (
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/__init__.py", line 11, in <module>
    from .tasks import *
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/__init__.py", line 17, in <module>
    import_packages(__name__, _BLACKLIST_PKGS)
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 40, in import_packages
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 84, in _walk_packages
    __import__(info.name)
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/__init__.py", line 1, in <module>
    from .bi_arm_env_cfg import (
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/bi_arm_env_cfg.py", line 22, in <module>
    from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/__init__.py", line 4, in <module>
    from .lekiwi import LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/__init__.py", line 1, in <module>
    from .lekiwi_gamepad import LeKiwiGamepad
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/lekiwi_gamepad.py", line 4, in <module>
    from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/utils/robot_utils.py", line 12, in <module>
    from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/enhance/datasets/lerobot_dataset_handler.py", line 6, in <module>
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
ModuleNotFoundError: No module named 'lerobot'
```

Installing LeRobot on IsaacLab environment causes the following errors:

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
nvidia-srl-usd-to-urdf 1.0.2 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
nvidia-srl-base 1.3.0 requires docstring-parser==0.16, which is not installed.
nvidia-srl-usd 2.0.0 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
nvidia-srl-usd-to-urdf 1.0.2 requires lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0 which is incompatible.
nvidia-srl-usd-to-urdf 1.0.2 requires numpy<2.0.0,>=1.21.5, but you have numpy 2.4.2 which is incompatible.
nvidia-srl-usd 2.0.0 requires numpy<2.0.0,>=1.21.5, but you have numpy 2.4.2 which is incompatible.
numba 0.59.1 requires numpy<1.27,>=1.22, but you have numpy 2.4.2 which is incompatible.
isaaclab-rl 0.4.7 requires numpy<2, but you have numpy 2.4.2 which is incompatible.
isaaclab-rl 0.4.7 requires packaging<24, but you have packaging 25.0 which is incompatible.
isaaclab-tasks 0.11.13 requires numpy<2, but you have numpy 2.4.2 which is incompatible.
isaaclab 0.54.3 requires numpy<2, but you have numpy 2.4.2 which is incompatible.
cmeel-boost 1.83.0 requires numpy~=1.26.0; python_version >= "3.9", but you have numpy 2.4.2 which is incompatible.
dex-retargeting 0.4.6 requires numpy<2.0.0,>=1.21.0, but you have numpy 2.4.2 which is incompatible.

Successfully installed InquirerPy-0.3.4 accelerate-1.12.0 av-15.1.0 cmake-4.1.3 datasets-4.1.1 deepdiff-8.6.1 diffusers-0.35.2 dill-0.4.0 draccus-0.10.0 evdev-1.9.3 hf-transfer-0.1.9 huggingface-hub-0.35.3 jsonlines-4.0.0 lerobot-0.4.4 mergedeep-1.3.4 multiprocess-0.70.16 mypy-extensions-1.1.0 numpy-2.4.2 orderly-set-5.5.0 packaging-25.0 pfzy-0.3.4 pyarrow-23.0.0 pynput-1.8.1 pyserial-3.5 python-xlib-0.33 pyyaml-include-1.4.1 rerun-sdk-0.26.2 setuptools-80.10.2 torchcodec-0.5 typing-inspect-0.9.0 wandb-0.24.2 xxhash-3.6.0
```

Dependencies:

- nvidia-srl-usd-to-urdf == 1.0.2 requires usd-core<26.0,>=25.2.post1
- nvidia-srl-base == 1.3.0 requires docstring-parser==0.16
- nvidia-srl-usd == 2.0.0 requires usd-core<26.0,>=25.2.post1
- nvidia-srl-usd == 2.0.0 requires numpy<2.0.0,>=1.21.5 (Big issue because other things need numpy 2.4.2)
- nvidia-srl-usd-to-urdf == 1.0.2 requires lxml<5.0.0,>=4.9.2 (have installed lxml 5.4.0)
- nvidia-srl-usd-to-urdf == 1.0.2 requires requires numpy<2.0.0,>=1.21.5 (Big issue because other things need numpy 2.4.2)
- numba == 0.59.1 requires numpy<1.27,>=1.22 (Big issue because other things need numpy 2.4.2)
- isaaclab-rl == 0.4.7 requires numpy<2 (Big issue because other things need numpy 2.4.2)
- isaaclab-rl == 0.4.7 requires packaging<24 (have installed packaging 25.0)
- isaaclab-tasks == 0.11.13 requires numpy<2 (Big issue because other things need numpy 2.4.2)
- isaaclab == 0.54.3 requires numpy<2 (Big issue because other things need numpy 2.4.2)
- cmeel-boost == 1.83.0 requires numpy~=1.26.0; python_version >= "3.9" (Big issue because other things need numpy 2.4.2)
- dex-retargeting == 0.4.6 requires numpy<2.0.0,>=1.21.0 (Big issue because other things need numpy 2.4.2)



[lxml](https://pypi.org/project/lxml/5.4.0/) is a Pythonic, mature binding for the libxml2 and libxslt libraries. It provides safe and convenient access to these libraries using the ElementTree API.

Installing numpy == 1.26.0

```bash
Successfully uninstalled numpy-2.4.2

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

nvidia-srl-usd-to-urdf 1.0.2 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
nvidia-srl-base 1.3.0 requires docstring-parser==0.16, which is not installed.
nvidia-srl-usd 2.0.0 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
nvidia-srl-usd-to-urdf 1.0.2 requires lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0 which is incompatible.
isaaclab-rl 0.4.7 requires packaging<24, but you have packaging 25.0 which is incompatible.
rerun-sdk 0.26.2 requires numpy>=2, but you have numpy 1.26.0 which is incompatible.

Successfully installed numpy-1.26.0
```



The dependencies issues caused after installing Lerobot + LeIsaac causes the IsacLab to be stuck handling Exception errors

```bash
[INFO]: Time taken for simulation start : 1.167973 seconds
Traceback (most recent call last):
  File "/home/goat/anaconda3/envs/leisaac/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 734, in make
    env = env_creator(**env_spec_kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py", line 80, in __init__
    super().__init__(cfg=cfg)
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py", line 173, in __init__
    self.sim.reset()
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py", line 530, in reset
    self.render()
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py", line 601, in render
    raise exception_to_raise
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/sensors/sensor_base.py", line 303, in _initialize_callback
    self._initialize_impl()
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py", line 217, in _initialize_impl
    annotator.attach(self._render_product_paths)
  File "/home/goat/isaacsim/extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/annotators.py", line 682, in attach
    activated_result = sdg_iface.activate_node_template(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/goat/isaacsim/extscache/omni.syntheticdata-0.6.13+69cbf6ad.lx64.r.cp311/omni/syntheticdata/scripts/SyntheticData.py", line 1607, in activate_node_template
    self._activate_node_rec(template_name, render_product_path_index, render_product_paths, render_var_activations)
  File "/home/goat/isaacsim/extscache/omni.syntheticdata-0.6.13+69cbf6ad.lx64.r.cp311/omni/syntheticdata/scripts/SyntheticData.py", line 1408, in _activate_node_rec
    SyntheticData._connect_nodes(connNode, node, connMap, True)
  File "/home/goat/isaacsim/extscache/omni.syntheticdata-0.6.13+69cbf6ad.lx64.r.cp311/omni/syntheticdata/scripts/SyntheticData.py", line 1104, in _connect_nodes
    success = (SyntheticData._add_node_downstream_intergraph_dependency(srcNode, dstNode.get_handle()) > 0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/goat/isaacsim/extscache/omni.syntheticdata-0.6.13+69cbf6ad.lx64.r.cp311/omni/syntheticdata/scripts/SyntheticData.py", line 1070, in _add_node_downstream_intergraph_dependency
    dep_attrib_data.set(dep_data)
TypeError: Unable to write from unknown dtype, kind=f, size=0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 281, in <module>
    main()
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 150, in main
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/goat/anaconda3/envs/leisaac/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 746, in make
    raise type(e)(
```


```bash
pip install "usd-core<26.0,>=25.2.post1"
pip install "lxml<5.0.0,>=4.9.2"
```

Updating the lxml causes: dex-retargeting 0.4.6 requires lxml>=5.2.2, but you have lxml 4.9.4 which is incompatible.