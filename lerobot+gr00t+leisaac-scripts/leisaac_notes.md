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

## Experiment - Install IsaacLab from leisaac with symbolic link

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

Pip install IsaacSim

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
wheel 0.46.3 requires packaging>=24.0, but you have packaging 23.0 which is incompatible.

Successfully installed Pillow-11.3.0 PyJWT-2.11.0 aioboto3-15.1.0 aiobotocore-2.24.0 aiodns-3.1.1 aiofiles-23.2.1 aiohappyeyeballs-2.4.4 aiohttp-3.11.11 aioitertools-0.11.0 aiosignal-1.3.2 annotated-types-0.7.0 anyio-4.12.1 asteval-1.0.6 async_timeout-5.0.1 attrs-25.1.0 awscrt-0.23.8 azure-core-1.28.0 azure-identity-1.13.0 azure-storage-blob-12.17.0 boto3-1.39.11 botocore-1.39.11 certifi-2026.1.4 cffi-2.0.0 charset_normalizer-3.3.2 click-8.1.7 contourpy-1.3.1 coverage-7.4.4 cryptography-44.0.0 cycler-0.11.0 fastapi-0.115.7 filelock-3.13.1 fonttools-4.55.3 frozenlist-1.5.0 fsspec-2024.6.1 gunicorn-23.0.0 h11-0.16.0 httptools-0.6.1 idna-3.10 idna-ssl-1.1.0 imageio-2.37.0 isaacsim-5.1.0.0 isaacsim-app-5.1.0.0 isaacsim-asset-5.1.0.0 isaacsim-benchmark-5.1.0.0 isaacsim-code-editor-5.1.0.0 isaacsim-core-5.1.0.0 isaacsim-cortex-5.1.0.0 isaacsim-example-5.1.0.0 isaacsim-extscache-kit-5.1.0.0 isaacsim-extscache-kit-sdk-5.1.0.0 isaacsim-extscache-physics-5.1.0.0 isaacsim-gui-5.1.0.0 isaacsim-kernel-5.1.0.0 isaacsim-replicator-5.1.0.0 isaacsim-rl-5.1.0.0 isaacsim-robot-5.1.0.0 isaacsim-robot-motion-5.1.0.0 isaacsim-robot-setup-5.1.0.0 isaacsim-ros1-5.1.0.0 isaacsim-ros2-5.1.0.0 isaacsim-sensor-5.1.0.0 isaacsim-storage-5.1.0.0 isaacsim-template-5.1.0.0 isaacsim-test-5.1.0.0 isaacsim-utils-5.1.0.0 isodate-0.6.1 jmespath-1.0.1 kiwisolver-1.4.4 llvmlite-0.42.0 markupsafe-2.1.3 matplotlib-3.10.3 msal-1.27.0 msal-extensions-1.0.0 multidict-6.1.0 nest_asyncio-1.5.6 networkx-3.3 numba-0.59.1 numpy-1.26.0 oauthlib-3.2.2 opencv-python-headless-4.11.0.86 osqp-0.6.7.post3 packaging-23.0 pint-0.20.1 portalocker-2.7.0 propcache-0.2.1 psutil-5.9.8 pycares-4.8.0 pycparser-3.0 pydantic-2.11.10 pydantic-core-2.33.2 pyparsing-3.0.9 pyperclip-1.8.0 pypng-0.20220715.0 python-dateutil-2.9.0.post0 python-multipart-0.0.20 pytz-2024.1 pyyaml-6.0.2 qdldl-0.1.7.post5 qrcode-7.4.2 requests-2.32.3 requests-oauthlib-1.3.1 rtree-1.3.0 s3transfer-0.13.1 scipy-1.15.3 sentry-sdk-2.29.1 six-1.17.0 starlette-0.45.3 sympy-1.13.3 toml-0.10.2 torchaudio-2.7.0 tornado-6.5.1 trimesh-4.5.1 typing-inspection-0.4.2 typing_extensions-4.12.2 urllib3-2.6.3 uvicorn-0.29.0 watchdog-4.0.0 websockets-12.0 wrapt-1.16.0 yarl-1.18.3

```

