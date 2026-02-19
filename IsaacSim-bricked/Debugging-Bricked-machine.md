# Notes on how to debugg the bricked machine

Summary:

My belief is that recent changes to cuda 13 and how its libraries and headers are added to the linux machine have changed. SO the paths and changes to CuDNN have changed and there can be only a single CuDNN.

The issue was resolved by reinstalling IsaacSIm from scratch.

Changes were done to the IsaacSim binaries

TLDR:

The Cuda stuff should be inside one of these:

/usr/include

/usr/local/

/usr/local/cuda/include



Sources of Information:

[This is how I make my Ubuntu NVIDIA drivers (and CUDA) work](https://github.com/garylvov/dev_env/tree/main/NVIDIA)


Running the compatibility checker script:

```bash
./isaac-sim.compatibility_check.sh
```

Outputs message:

IOMMU Enabled

An input-output memory management unit (IOMMU) appears to be enabled on this system.

On bare-metal Linux systems, CUDA and the display driver do not support IOMMU-enabled PCIe peer to peer memory copy.

If you are on a bare-metal Linux system, please disable the IOMMU. Otherwise you risk image corruption and program instability.

This typically can be controlled via BIOS settings (Intel Virtualization Technology for Directed I/O (VT-d) or AMD I/O Virtualization Technology (AMD-Vi)) and kernel parameters (iommu, intel_iommu, amd_iommu).

Note that in virtual machines with GPU pass-through (vGPU) the IOMMU needs to be enabled.

Since we can not reliably detect whether this system is bare-metal or a virtual machine, we show this warning in any case when an IOMMU appears to be enabled.


Also, might want to turn off iommu

```bash
sudo vim /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=off"
sudo update-grub
sudo reboot
```


## Debugging CUDNN

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Output --- wrong
cat: /usr/local/cuda/include/cudnn_version.h: No such file or directory
```

```bash
sudo find / -name "cudnn_version.h" 2>/dev/null

/home/goat/.local/share/Trash/files/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/include/cudnn_version.h
/home/goat/Documents/GitHub/renanmb/Isaac-GR00T/.venv/lib/python3.10/site-packages/nvidia/cudnn/include/cudnn_version.h
/home/goat/Documents/GitHub/renanmb/IsaacLab/env_isaaclab/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_version.h
/home/goat/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/include/cudnn_version.h
/home/goat/.cache/uv/archive-v0/HpSm0WQKYk0bii1niJlG2/nvidia/cudnn/include/cudnn_version.h
/home/goat/.cache/uv/archive-v0/kkIXEjdyrIFJSYzFhBOGC/nvidia/cudnn/include/cudnn_version.h
/home/goat/anaconda3/envs/comfyui/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_version.h
/home/goat/anaconda3/envs/svraster/lib/python3.9/site-packages/nvidia/cudnn/include/cudnn_version.h
/home/goat/anaconda3/envs/gr00t/lib/python3.10/site-packages/nvidia/cudnn/include/cudnn_version.h
```

This shows that the system have cuDNN, but only the Python/Conda wheel version, not the system CUDA version.

All the paths are found inside environments like:

```swift
...site-packages/nvidia/cudnn/include/cudnn_version.h
```

That seems tp be the pip/conda packaged cuDNN used by PyTorch/TensorFlow.

But the issue is that Isaac Sim / Isaac Lab native components expect the system cuDNN in /usr/local/cuda.

Isaac Lab native extensions compile against system CUDA, therefore they look here:

```swift
/usr/local/cuda/include/cudnn_version.h
```

## Install CuDNN

Installing the wrong version of stuff brick the whole computer.

- [Installing cuDNN on Linux](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.0.0/installation/linux.html)
- [Nvidia Network Repository Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu)
- [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions)
- [Installing cuDNN Backend on Linux](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)


### Step 1 — Confirm your CUDA version

```bash
nvcc --version
```

1. Add NVIDIA CUDA repo



```bash
# Install Nvidia Network Repository 
$ wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<arch>/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb
```


```bash
sudo apt update
sudo apt install -y wget gnupg
# Install the cuda-keyring package:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
# Refresh the repository metadata.
sudo apt update
```

Where $distro/$arch should be replaced by one of the following:

- ubuntu2004/x86_64
- ubuntu2004/sbsa
- ubuntu2004/cross-linux-sbsa
- ubuntu2004/arm64
- ubuntu2004/cross-linux-aarch64
- ubuntu2204/x86_64
- ubuntu2204/sbsa
- ubuntu2204/cross-linux-sbsa
- ubuntu2204/arm64
- ubuntu2204/cross-linux-aarch64
- debian11/x86_64

For arm64-sbsa repos:

- Native: $distro/sbsa
- Cross: $distro/cross-linux-sbsa

For aarch64-jetson repos:

- Native: $distro/arm64
- Cross: $distro/cross-linux-aarch64



2. Install cuDNN for CUDA 13

This installs the system headers + libs into /usr/local/cuda:

```bash
sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13
```


In the Nvidia website documentation they recommend using the meta packages.

Install the per-CUDA meta-packages. Examples:

To install for CUDA 11, run:

```bash
sudo apt-get -y install cudnn9-cuda-11
```

To install for CUDA 12, run:

```bash
sudo apt-get -y install cudnn9-cuda-12
```

3. Verify installation


```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

4. Post-Installation Actions

The post-installation actions must be manually performed. These actions are split into mandatory, recommended, and optional sections.

**Mandatory Actions**

Some actions must be taken after the installation before the CUDA Toolkit can be used.


The ```PATH``` variable needs to include ```export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}```. Nsight Compute has moved to ```/opt/nvidia/nsight-compute/``` only in rpm/deb installation method. When using .run installer it is still located under ```/usr/local/cuda-13.0/```.

To add this path to the ```PATH``` variable:

```bash
export PATH=${PATH}:/usr/local/cuda-13.0/bin
```

In addition, **when using the runfile installation method**, the LD_LIBRARY_PATH variable needs to contain /usr/local/cuda-13.0/lib64 on a 64-bit system and /usr/local/cuda-13.0/lib for the 32 bit compatibility:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-13.0/lib64
```

Note that the above paths change when using a custom install path with the runfile installation method.

**Recommended Actions**

Other actions are recommended to verify the integrity of the installation.

Install Writable Samples

CUDA Samples are now located in https://github.com/nvidia/cuda-samples, which includes instructions for obtaining, building, and running the samples.

**Verify the Installation**

Before continuing, it is important to verify that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware. To do this, you need to compile and run some of the sample programs, located in https://github.com/nvidia/cuda-samples



## MISC

NVIDIA split cuDNN into runtime vs dev packages which does not seem to be fully explained in their documentation website. You need both runtime + development headers to run stuff like Isaac / native CUDA builds.

```bash
sudo apt-get install cudnn9-cuda-13
```

| Package                 | Contains                          | Needed for      |
| ----------------------- | --------------------------------- | --------------- |
| `cudnn9-cuda-13`        | meta package (pulls runtime libs) | running apps    |
| `libcudnn9-cuda-13`     | actual shared libraries (.so)     | runtime linking |
| `libcudnn9-dev-cuda-13` | headers + static libs             | **compilation** |


IsaacLab compiles native extensions, so it needs the headers:

```swift
/usr/local/cuda/include/cudnn*.h
```

So we install runtime and dev package explicitly. The meta package cudnn9-cuda-13 is optional and redundant once you install these directly.

```bash
sudo apt update
sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13
```

After Installing we can check:

```bash
ls /usr/local/cuda/include | grep cudnn

ls /usr/local/cuda-13/include | grep cudnn

ls /usr/local/cuda-13.1/include | grep cudnn
```

Expected to see:

```bash
cudnn.h
cudnn_version.h
```

### Confusion with Nvidia Meta Package

The ```cudnn9-cuda-13``` is just a meta package (a convenience package). The runtime libraries are ```libcudnn9-cuda-13``` and ```libcudnn9-dev-cuda-13```.

| Package                 | Type         | What it really is         |
| ----------------------- | ------------ | ------------------------- |
| `cudnn9-cuda-13`        | meta package | installer shortcut        |
| `libcudnn9-cuda-13`     | runtime libs | the real `.so` files      |
| `libcudnn9-dev-cuda-13` | dev headers  | the headers + static libs |

**Why not rely on the meta package ?**

Because depending on repo version, ```cudnn9-cuda-13``` may or may not pull the -dev package automatically.

For IsaacLab we must guarantee headers exist, so we install explicitly.


## Issues

```bash
echo $LD_LIBRARY_PATH
```

It prints the path: ```/usr/local/cuda-13.1/lib64/```

```bash
echo $PATH
```

It prints

```bash
/home/goat/anaconda3/bin:/home/goat/anaconda3/condabin:/usr/local/cuda-13.1/bin:/home/goat/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/goat/.local/bin:/home/goat/.vscode/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/scripts/noConfigScripts:/home/goat/.local/bin
```

Paths

- /home/goat/anaconda3/bin
- /home/goat/anaconda3/condabin
- /usr/local/cuda-13.1/bin
- /home/goat/.local/bin
- /usr/local/sbin
- /usr/local/bin
- /usr/sbin
- /usr/bin
- /sbin
- /bin
- /usr/games
- /usr/local/games
- /snap/bin
- /snap/bin
- /home/goat/.local/bin
- /home/goat/.vscode/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/scripts/noConfigScripts
- /home/goat/.local/bin



## Trying to run cuDNN smaples:

```bash
cd /usr/src/cudnn_samples_v9/mnistCUDNN
```

Obtain following error:

```bash
/usr/src/cudnn_samples_v9/mnistCUDNN$ sudo make clean && make
rm -rf *o
rm -rf mnistCUDNN
CUDA_VERSION is 13010
Linking agains cublasLt = true
CUDA VERSION: 13010
TARGET ARCH: x86_64
HOST_ARCH: x86_64
TARGET OS: linux
SMS: 75 80 86 87 90 100 110 120
/bin/sh: 1: cannot create test.c: Permission denied
/bin/sh: 1: cannot create test.c: Permission denied
cc1: fatal error: test.c: No such file or directory
compilation terminated.
>>> WARNING - FreeImage is not set up correctly. Please ensure FreeImage is set up correctly. <<<
[@] /usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include -ccbin g++ -m64 -std=c++11 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_87,code=sm_87 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_110,code=sm_110 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_90,code=compute_90 -o fp16_dev.o -c fp16_dev.cu
[@] g++ -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include -std=c++11 -o fp16_emu.o -c fp16_emu.cpp
[@] g++ -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include -std=c++11 -o mnistCUDNN.o -c mnistCUDNN.cpp
[@] /usr/local/cuda/bin/nvcc -ccbin g++ -m64 -std=c++11 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_87,code=sm_87 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_110,code=sm_110 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_90,code=compute_90 -o mnistCUDNN fp16_dev.o fp16_emu.o mnistCUDNN.o -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 -lcublasLt -LFreeImage/lib/linux/x86_64 -LFreeImage/lib/linux -lcudart -lcublas -lcudnn -lfreeimage -lstdc++ -lm
```

change permissions and then run the compiler

```bash
sudo chown -R $USER:$USER .

# install freeimage The dependency having issues
sudo apt-get update
sudo apt-get install -y libfreeimage-dev

# Add to the Folder: /usr/src/cudnn_samples_v9/mnistCUDNN
mkdir -p FreeImage/include
mkdir -p FreeImage/lib/linux/x86_64

# Add the symbolic links
ln -s /usr/include/FreeImage.h FreeImage/include/FreeImage.h
ln -s /usr/lib/x86_64-linux-gnu/libfreeimage.so FreeImage/lib/linux/x86_64/libfreeimage.so

# Now run
./mnistCUDNN
```

## Isaacsim checking the paths

According to the IsaacSim Issue on github

- [[Bug] undefined symbol #90](https://github.com/isaac-sim/IsaacSim/issues/90)

This commands run inside the ```isaacsim``` folder:

```bash
./python.sh -m pip show torch
```

Output:

```bash
Warning: running in conda env, please deactivate before executing this script
If conda is desired please source setup_conda_env.sh in your python 3.11 conda env and run python normally
WARNING: Package(s) not found: torch
There was an error running python
```

It should have outputed something like:

```bash
./python.sh -m pip show torch

Name: torch
Version: 2.7.0+cu128
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3-Clause
Location: /home/user/repos/omni_isaac_sim/_build/linux-x86_64/release/exts/omni.isaac.ml_archive/pip_prebundle
Requires: filelock, fsspec, jinja2, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-cufile-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-cusparselt-cu12, nvidia-nccl-cu12, nvidia-nvjitlink-cu12, nvidia-nvtx-cu12, sympy, triton, typing-extensions
Required-by: torchaudio, torchvision
```

My output means I have probably the paths or something terribly wrong on my machine.

