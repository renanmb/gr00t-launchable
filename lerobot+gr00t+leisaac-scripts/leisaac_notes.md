# Notes for installing LeIsaac

leisaac seems to be installing the branch or commit from IsaacLab: **3c6e67b**



The installation required gcc-13

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-13 g++-13
```

Verify installation, since it is experimental build it is on gcc-13 

```bash
gcc-13 --version
```

Setting GCC 13 as Default (Optional)

If you need to make GCC 13 the default gcc command, you can use update-alternatives to configure it: 

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 --slave /usr/bin/g++ g++ /usr/bin/g++-13
```


Installing Build Essential

For a complete development environment (including make, libc, etc.), it is recommended to also install build-essential: 

```bash
sudo apt install build-essential
```

Other Issue

Issue gives me an error that CXXABI_1.3.15 is not found libstdcxx-ng>=13


Install or upggrade the libstdcxx-ng

```bash
conda install -c conda-forge "libstdcxx-ng>=13"
```

verify installation 

```bash
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep CXXABI
```

Force Conda Env Library with LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

there are also other ways like making a symbolic link or using LD_RELOAD

## IMPORTANT must download Assets

[leisaac-example-scene](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)

[kitchen_with_orange.zip](https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip)


## Notes on dependencies

lerobot = ["lerobot==0.4.2"]

## Install LeIsaac on IsaacLab and then install lerobot

It has an errorcd

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
nvidia-srl-usd-to-urdf 1.0.2 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
nvidia-srl-usd 2.0.0 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
nvidia-srl-base 1.3.0 requires docstring-parser==0.16, which is not installed.
nvidia-srl-usd-to-urdf 1.0.2 requires lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0 which is incompatible.
nvidia-srl-usd-to-urdf 1.0.2 requires numpy<2.0.0,>=1.21.5, but you have numpy 2.4.2 which is incompatible.
nvidia-srl-usd 2.0.0 requires numpy<2.0.0,>=1.21.5, but you have numpy 2.4.2 which is incompatible.
numba 0.59.1 requires numpy<1.27,>=1.22, but you have numpy 2.4.2 which is incompatible.
isaaclab-tasks 0.11.13 requires numpy<2, but you have numpy 2.4.2 which is incompatible.
isaaclab-rl 0.4.7 requires numpy<2, but you have numpy 2.4.2 which is incompatible.
isaaclab-rl 0.4.7 requires packaging<24, but you have packaging 25.0 which is incompatible.
cmeel-boost 1.83.0 requires numpy~=1.26.0; python_version >= "3.9", but you have numpy 2.4.2 which is incompatible.
dex-retargeting 0.4.6 requires numpy<2.0.0,>=1.21.0, but you have numpy 2.4.2 which is incompatible.
isaaclab 0.54.3 requires numpy<2, but you have numpy 2.4.2 which is incompatible.
```

New issues trying to fix the old one:

```bash
   98  cd lerobot/
   99  git checkout v0.4.2
  100  pip install -e .
  101  pip install numpy<2
  102  pip install numpy<=2
  103  pip install numpy==1.26.4
  104  pip install lxml
  105  pip install lxml>=5.2.2
  106  pip show packaging
  107  pip list --outdated packaging
  108  pip install packaging==23.2
  109  pip install packaging==24.2
```

Downgrade Packaging and have issues

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
lerobot 0.4.2 requires packaging<26.0,>=24.2, but you have packaging 23.2 which is incompatible.
wheel 0.46.3 requires packaging>=24.0, but you have packaging 23.2 which is incompatible.
dex-retargeting 0.4.6 requires lxml>=5.2.2, but you have lxml 4.9.4 which is incompatible.
```

Upgrade it and have more issues

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
isaaclab-rl 0.4.7 requires packaging<24, but you have packaging 24.2 which is incompatible.
isaaclab 2.3.2.post1 requires packaging<24, but you have packaging 24.2 which is incompatible.
dex-retargeting 0.4.6 requires lxml>=5.2.2, but you have lxml 4.9.4 which is incompatible.
```


```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
lerobot 0.4.2 requires packaging<26.0,>=24.2, but you have packaging 23.2 which is incompatible.
wheel 0.46.3 requires packaging>=24.0, but you have packaging 23.2 which is incompatible.
dex-retargeting 0.4.6 requires lxml>=5.2.2, but you have lxml 4.9.4 which is incompatible.
```

Following these 

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


```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
lerobot 0.4.2 requires packaging<26.0,>=24.2, but you have packaging 23.0 which is incompatible.
wheel 0.46.3 requires packaging>=24.0, but you have packaging 23.0 which is incompatible.
dex-retargeting 0.4.6 requires lxml>=5.2.2, but you have lxml 4.9.4 which is incompatible.
rerun-sdk 0.26.2 requires numpy>=2, but you have numpy 1.26.0 which is incompatible.

Successfully installed PyJWT-2.11.0 isaacsim-5.1.0.0 isaacsim-app-5.1.0.0 isaacsim-asset-5.1.0.0 isaacsim-benchmark-5.1.0.0 isaacsim-code-editor-5.1.0.0 isaacsim-core-5.1.0.0 isaacsim-cortex-5.1.0.0 isaacsim-example-5.1.0.0 isaacsim-extscache-kit-5.1.0.0 isaacsim-extscache-kit-sdk-5.1.0.0 isaacsim-extscache-physics-5.1.0.0 isaacsim-gui-5.1.0.0 isaacsim-kernel-5.1.0.0 isaacsim-replicator-5.1.0.0 isaacsim-rl-5.1.0.0 isaacsim-robot-5.1.0.0 isaacsim-robot-motion-5.1.0.0 isaacsim-robot-setup-5.1.0.0 isaacsim-ros1-5.1.0.0 isaacsim-ros2-5.1.0.0 isaacsim-sensor-5.1.0.0 isaacsim-storage-5.1.0.0 isaacsim-template-5.1.0.0 isaacsim-test-5.1.0.0 isaacsim-utils-5.1.0.0 markupsafe-2.1.3 numpy-1.26.0 packaging-23.0
```