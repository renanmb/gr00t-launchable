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
./isaaclab.sh --install

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

