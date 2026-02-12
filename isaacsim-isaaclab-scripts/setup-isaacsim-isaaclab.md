# Setting Up IsacSim and IsaacLab

First install conda then install IsaacSim 5.1 binaries and then install IsaacLab v2.3.0

| Dependency | Isaac Sim 5.1 |
| :--- | :---: |
| **Python** | 3.11 |
| **Isaac Lab** | v2.3.0 |
| **CUDA** | 12.8 |
| **PyTorch** | 2.7.0 |

Instance Name: test-g6e-8xlarge-f0631e

```bash
scp install-conda_v2.sh isaacsim_v2.sh isaaclab_v3.sh test-g6e-8xlarge-f0631e:~
```

Make sure it is executable

```bash
chmod +x install-conda_v2.sh isaacsim_v2.sh isaaclab_v3.sh
```

Order of Execution:

install-conda_v2.sh -> isaacsim_v2.sh -> isaaclab_v3.sh

Note:

- Need to run IsaacSim to shaders and several hidden things dont cause issues
- Need to run IsaacLab to test it for similar reasons

## IsaacSim

Run IsaacSim

```bash
cd isaacsim
./isaac-sim.sh
```

TODO:

- Install Leatherback extensions
- Create Script to install any extensions given a simple file.

Need to check why it needs to run on the noVNC not on the ssh shell for it to render on screen. maybe its a user thing ?


## IsaacLab

Run IsaacLab to verify everything works

```bash
cd IsaacLab
# Create an empty environment --- great to test without any learning framework
isaaclab -p scripts/tutorials/00_sim/create_empty.py
```

Need to check why it needs to run on the noVNC not on the ssh shell for it to render on screen. maybe its a user thing ?

Training Ant

environment: Isaac-Ant-v0

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

Run inference

environment: Isaac-Ant-v0

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Ant-v0 --num_envs 32
```

Training Quadruped

environment: Isaac-Velocity-Rough-Anymal-C-v0

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

Run inference

environment: Isaac-Velocity-Rough-Anymal-C-v0

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 32
```