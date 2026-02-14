# How to install leisaac

This install leisaac using a specific version of isaaclab and a pip installation of IsaacSim


## Step 1 - Installation

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