# General Infor about Brev instances

Experiment Tracking:

Instance Name:  test-g6e-8xlarge-f0631e

Has everything up to GR00T installed

Instance Name: test-g6e-8xlarge-65964e

Figuring out leisaac vs isaaclab vs lerobot dependencies conflict

## AWS
L40S

- g6e.8xlarge
- g6e.12xlarge
- g6e.24xlarge
- g6e.48xlarge

## NEBIUS
L40S

NO GO -- NEBIUS does not support

- gpu-l40s-a.1gpu-32vcpu-128gb

Nebius issues:

```bash
The following packages have unmet dependencies:
 xfce4-sensors-plugin : Depends: libxnvctrl0 but it is not installable
E: Unable to correct problems, you have held broken packages.
```

There is a possibility the NEBIUS provider does not support GUI or the installation requirements for running.


## GCP
H100

- a3-megagpu-8g:nvidia-h100-mega-80gb:8


## How to get System Information

Below there is an example of how to look for properties in the instance, properties that will give hint about comptability and system details important before you proceed.

Find Properties of instance example g6e.8xlarge:

```bash
uname -r
```

Output:

6.8.0-1044-aws

```bash
uname -a
```

Output:

Linux brev-9wk0otwek 6.8.0-1044-aws #46~22.04.1-Ubuntu SMP Tue Dec  2 12:52:18 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux


```bash
lsb_release -a
```

Output:

No LSB modules are available.	
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.5 LTS
Release:	22.04
Codename:	jammy

```bash
nvidia-smi
```

Output:

Driver Version: 580.126.09     CUDA Version: 13.0 

```bash
nvcc --version
```

Output:

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0

```bash
docker --version
```

Output:

Docker version 29.1.5, build 0e6fee6

```bash
docker compose version
```

Output:

Docker Compose version v5.0.2

```bash
nvidia-ctk --version
```

Output:

NVIDIA Container Toolkit CLI version 1.18.1
commit: efe99418ef87500dbe059cadc9ab418b2815b9d5

```bash
python3 --version
```

Output:

Python 3.10.12


```bash
pip --version
```

Output:

pip 25.3 from /usr/local/lib/python3.10/dist-packages/pip (python 3.10)