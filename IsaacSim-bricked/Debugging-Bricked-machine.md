# Notes on how to debugg the bricked machine

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
sudo apt-get install -y wget gnupg
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

This command seems wrong

In the Nvidia website documentation

Install the per-CUDA meta-packages.

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



