# Flash attention notes


Current information regarding flash attention issues

[Build flash-attn takes a lot of time  #1038](https://github.com/Dao-AILab/flash-attention/issues/1038?utm_source=chatgpt.com)

[[bug] build is verrrrrrrrrrrrrrrrrrrry slow #945](https://github.com/Dao-AILab/flash-attention/issues/945?utm_source=chatgpt.com)

[why installing flash-attn takes so long? #1776](https://github.com/Dao-AILab/flash-attention/issues/1776?utm_source=chatgpt.com)

[Has anyone had to install Flash-Attention? I've been compiling for 5 hours.](https://www.reddit.com/r/chileIT/comments/1ie7zu6/alguien_ha_tenido_que_instalar_flashattention_voy/?tl=en&utm_source=chatgpt.com)

There is a recommendation that you could run

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

There are several issue when compiling and installing flash_attn with current changes in gr00t.

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

FAscinating enough the L40S has compute_cap 8.9

```txt
PyTorch version: 2.7.0+cu126
CUDA Available: True
CUDA Version (torch): 12.6
GPU Index: 0
GPU Name: NVIDIA L40S
Compute Capability: 8.9
Total VRAM: 44.39 GB
Architecture: Ada
```

```txt
PyTorch version: 2.7.0+cu128
CUDA Available: True
CUDA Version (torch): 12.8
GPU Index: 0
GPU Name: NVIDIA GeForce RTX 3090
Compute Capability: 8.6
Total VRAM: 23.53 GB
Architecture: Ampere
```


```bash
(gr00t) ubuntu@brev-48mf8ilk:~/Isaac-GR00T$ python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/flash_attn/__init__.py", line 3, in <module>
    from flash_attn.flash_attn_interface import (
  File "/home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 15, in <module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: /home/ubuntu/.conda/envs/gr00t/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

test of 02/02/2026 --- instance ID: test-g6e-8xlarge-3dfe06


CHeck GPU architecture

```bash
python - <<'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit(0)

idx = torch.cuda.current_device()
props = torch.cuda.get_device_properties(idx)

print(f"CUDA Version (torch): {torch.version.cuda}")
print(f"GPU Index: {idx}")
print(f"GPU Name: {props.name}")
print(f"Compute Capability: {props.major}.{props.minor}")
print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GB")

# Architecture hint (useful for flash-attn / xformers decisions)
cc = props.major * 10 + props.minor
arch_map = {
    70: "Volta",
    75: "Turing",
    80: "Ampere",
    86: "Ampere",
    89: "Ada",
    90: "Hopper"
}
print(f"Architecture: {arch_map.get(cc, 'Unknown / Pre-Volta')}")
EOF
```

Possibly wrong

```cpp
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);

        printf("\n================ CUDA DEVICE %d ================\n", dev);
        printf("Name                         : %s\n", p.name);
        printf("Compute Capability           : %d.%d\n", p.major, p.minor);
        printf("Multiprocessors (SMs)        : %d\n", p.multiProcessorCount);
        printf("Warp Size                    : %d\n", p.warpSize);

        printf("\n--- Execution Limits ---\n");
        printf("Max Threads per Block        : %d\n", p.maxThreadsPerBlock);
        printf("Max Threads per SM           : %d\n", p.maxThreadsPerMultiProcessor);
        printf("Registers per Block          : %d\n", p.regsPerBlock);
        printf("Shared Memory per Block      : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("Shared Memory per SM         : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);

        printf("\n--- Memory ---\n");
        printf("Total Global Memory          : %.2f GB\n",
               p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Memory Bus Width             : %d bits\n", p.memoryBusWidth);
        printf("Memory Clock Rate            : %.0f MHz\n", p.memoryClockRate / 1000.0);
        printf("L2 Cache Size                : %d MB\n", p.l2CacheSize / (1024 * 1024));

        printf("\n--- Capabilities ---\n");
        printf("Unified Addressing           : %s\n", p.unifiedAddressing ? "Yes" : "No");
        printf("Concurrent Kernels           : %s\n", p.concurrentKernels ? "Yes" : "No");
        printf("Cooperative Launch           : %s\n", p.cooperativeLaunch ? "Yes" : "No");
    }

    return 0;
}
```

Then compile the code:

```bash
nvcc -O2 gpu_arch.cu -o gpu_arch
```


## Dependencies Experiments

The dependencies are finnick so need to experiment to what is going to work for the L40S

Create the conda env from the environment.yml

```bash
conda env create -f environment.yml
```


```bash
conda env update -n gr00t --file environment.yml
```

```bash
conda remove -n ENV_NAME --all
```


## Error track

```bash
The conflict is caused by:
    The user requested wandb==0.21.4
    lerobot 0.4.3 depends on wandb<0.25.0 and >=0.24.0

Additionally, some packages in these conflicts have no matching distributions available for your environment:
    wandb

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip to attempt to solve the dependency conflict


Pip subprocess error:
ERROR: Cannot install -r /home/ubuntu/condaenv.v97cxepl.requirements.txt (line 5) and wandb==0.21.4 because these package versions have conflicting dependencies.
ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts

failed

CondaEnvException: Pip failed

```
