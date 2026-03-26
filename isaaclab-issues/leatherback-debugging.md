# Leatherback has issues with the Python

The project runs fine on the local machine and the cloud instance it has run several times however there is an error that appears:

```python
# create runner from rsl-rl
if agent_cfg.class_name == "OnPolicyRunner":
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
elif agent_cfg.class_name == "DistillationRunner":
    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
else:
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
```        

The error leads to the line 199 at the ```rsl_rl/train.py``` script.


## Class Name Error with instance:

```bash
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
01:22:51 [actuator_pd.py] WARNING: The <ImplicitActuatorCfg> object has a value for 'effort_limit'. This parameter will be removed in the future. To set the effort limit, please use 'effort_limit_sim' instead.
01:22:51 [actuator_pd.py] WARNING: The <ImplicitActuatorCfg> object has a value for 'velocity_limit'. Previously, although this value was specified, it was not getting used by implicit actuators. Since this parameter affects the simulation behavior, we continue to not use it. This parameter will be removed in the future. To set the velocity limit, please use 'velocity_limit_sim' instead.
2026-03-14T01:22:52Z [11,704ms] [Warning] [omni.physx.fabric.plugin] FabricManager::initializePointInstancer mismatched prototypes on point instancer: /World/Visuals/Cones.
[INFO]: Time taken for simulation start : 1.753339 seconds
[INFO]: Completed setting up the environment...
[INFO] Instability logging setup: logs/instability_analysis/instability_analysis_2026-03-14_01-22-52.csv
Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 100, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/ubuntu/goat_racer_test/leatherback/scripts/rsl_rl/train.py", line 199, in main
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/isaaclab/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py", line 40, in __init__
    self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/isaaclab/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py", line 477, in construct_algorithm
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'class_name'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
2026-03-14T01:22:52Z [12,212ms] [Warning] [omni.physx.plugin] USD stage detach not called, holding a loose ptr to a stage!
2026-03-14T01:22:53Z [12,892ms] [Warning] [carb] Recursive unloadAllPlugins() detected!
```

## normal run 

It runs normallyinto the local machine and it also has worked sometimes with the cloud instance.

```bash
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
20:23:41 [actuator_pd.py] WARNING: The <ImplicitActuatorCfg> object has a value for 'effort_limit'. This parameter will be removed in the future. To set the effort limit, please use 'effort_limit_sim' instead.
20:23:41 [actuator_pd.py] WARNING: The <ImplicitActuatorCfg> object has a value for 'velocity_limit'. Previously, although this value was specified, it was not getting used by implicit actuators. Since this parameter affects the simulation behavior, we continue to not use it. This parameter will be removed in the future. To set the velocity limit, please use 'velocity_limit_sim' instead.
2026-03-14T01:23:41Z [7,784ms] [Warning] [omni.physx.fabric.plugin] FabricManager::initializePointInstancer mismatched prototypes on point instancer: /World/Visuals/Cones.
[INFO]: Time taken for simulation start : 0.756741 seconds
[INFO]: Completed setting up the environment...
[INFO] Instability logging setup: logs/instability_analysis/instability_analysis_2026-03-13_20-23-41.csv
/home/goat/anaconda3/envs/isaaclab232/lib/python3.11/site-packages/rsl_rl/utils/utils.py:245: UserWarning: The observation configuration dictionary 'obs_groups' must contain the 'policy' key. As an observation group with the name 'policy' was found, this is assumed to be the observation set. Consider adding the 'policy' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
  warnings.warn(
/home/goat/anaconda3/envs/isaaclab232/lib/python3.11/site-packages/rsl_rl/utils/utils.py:291: UserWarning: The observation configuration dictionary 'obs_groups' must contain the 'critic' key. As the configuration for 'critic' is missing, the observations from the 'policy' set are used. Consider adding the 'critic' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
  warnings.warn(
--------------------------------------------------------------------------------
Resolved observation sets: 
     policy :  ['policy']
     critic :  ['policy']
--------------------------------------------------------------------------------
Actor MLP: MLP(
  (0): Linear(in_features=8, out_features=32, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=32, out_features=32, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=32, out_features=2, bias=True)
)
Critic MLP: MLP(
  (0): Linear(in_features=8, out_features=32, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=32, out_features=32, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=32, out_features=1, bias=True)
)
################################################################################
                        Learning iteration 0/300                        

                       Computation: 226426 steps/s (collection: 0.414s, learning 0.165s)
             Mean action noise std: 1.07
          Mean value_function loss: 0.0586
               Mean surrogate loss: 0.0007
                 Mean entropy loss: 2.9040
                       Mean reward: 0.71
               Mean episode length: 28.82
--------------------------------------------------------------------------------
                   Total timesteps: 131072
                    Iteration time: 0.58s
                      Time elapsed: 00:00:00
                               ETA: 00:02:53

```

## Error with the Play script

Somehow it is parsig the script wrongly

```bash
[13.472s] Simulation App Startup Complete
Traceback (most recent call last):
  File "/home/ubuntu/goat_racer_test/leatherback/scripts/rsl_rl/play.py", line 232, in <module>
    main()
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 80, in wrapper
    env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 41, in register_task_to_hydra
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", line 58, in load_cfg_from_registry
    cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/isaaclab/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 1000, in spec
    _check_version_exists(ns, name, version)
  File "/opt/conda/envs/isaaclab/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 392, in _check_version_exists
    _check_name_exists(ns, name)
  File "/opt/conda/envs/isaaclab/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 369, in _check_name_exists
    raise error.NameNotFound(
gymnasium.error.NameNotFound: Environment `Template-Leatherback-Direct` doesn't exist.
```

## Rolling Back IsaacLab

IsaacLab v2.3.2 has several issues

[[Bug Report] flatdict can't be install for python3.11 because lack of pkg_resources in setuptools #4577](https://github.com/isaac-sim/IsaacLab/issues/4577)

Metnions that must fix some packages since the project has removed:

maybe this error is caused by the change of setuptools, they removed pkg_resources yesterday

```bash
pip install pip==23
pip install setuptools==65
pip install flatdict
./isaaclab.sh --install
```

There are further changes in the project like flatdict 


Always remember the USD assets and some stuff need to get Git LFS pull

```bash
git lfs fetch --all
git lfs pull
git lfs checkout
```


## Verifying multiple versions in Bash

```bash
# 1. Get the current version string (e.g., v2.3.2)
CURRENT_GIT_VER=$(git describe --tags --abbrev=0 2>/dev/null || git rev-parse --abbrev-ref HEAD)

# 2. Use 'sort -V' to check if CURRENT_GIT_VER is less than or equal to v2.3.2
# This returns true (exit code 0) if CURRENT_GIT_VER is <= v2.3.2
if [ "$(printf '%s\n%s' "v2.3.2" "$CURRENT_GIT_VER" | sort -V | head -n1)" == "$CURRENT_GIT_VER" ]; then
  echo "▶ Version $CURRENT_GIT_VER detected (v2.3.2 or lower). Applying legacy patches..."
  conda run -n "$CONDA_ENV_NAME" python -m pip install pip==23.3.2
  conda run -n "$CONDA_ENV_NAME" python -m pip install setuptools==65.5.0
  conda run -n "$CONDA_ENV_NAME" python -m pip install flatdict
else
  echo "▶ Version $CURRENT_GIT_VER is newer than v2.3.2. Skipping legacy patches."
fi
```


## Leatherback Installation Verification

The solution might be to use the built in tools in IsaacLab to test the installations

```bash
export TARGET_USER="ubuntu"
export CONDA_ENV_NAME="isaaclab"
export LEATHERBACK_DIR="/home/ubuntu/goat_racer_test/leatherback"
export TASK_NAME="Template-Leatherback-Direct-v0"

echo "▶ Verifying environment registration..."

# 1. Run the list_envs script
# 2. Grep for your specific task name
# 3. If grep finds it, it returns exit code 0 (success)
if sudo -H -u "$TARGET_USER" -i bash -c "conda run -n $CONDA_ENV_NAME python '$LEATHERBACK_DIR/scripts/list_envs.py'" | grep "Template-Leatherback-Direct-v0"; then
    echo "✅ Success: 'Template-Leatherback-Direct-v0' is registered and visible!"
else
    echo "❌ Failure: Environment 'Template-Leatherback-Direct-v0' not found in the list."
    echo "Available environments were:"
    sudo -H -u "$TARGET_USER" -i bash -c "conda run -n $CONDA_ENV_NAME python '$LEATHERBACK_DIR/scripts/list_envs.py'"
    # exit 1
fi
```

I dont think piping and using grep will work

```bash
grep -q "$TASK_NAME"

```

Try storing the output:

```bash
# Store the output to check and display
OUTPUT=$(sudo -H -u "$TARGET_USER" -i bash -c "conda run -n $CONDA_ENV_NAME --cwd '$LEATHERBACK_DIR' python scripts/list_envs.py")

if echo "$OUTPUT" | grep "$TASK_NAME"; then
    echo "✅ Found $TASK_NAME"
else
    echo "❌ Task missing from registration table!"
    # exit 1
fi
```