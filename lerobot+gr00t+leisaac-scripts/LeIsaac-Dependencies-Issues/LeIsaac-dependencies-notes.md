# Notes on the dependencies issues for LeIsaac

LeIsaac depends on LeRobot which depends on packages and several versions that causes issues to IsacSim, IsaacLab and many other Nvidia Software.

numpy version is a big one

Error Handling for avoiding Lerobot inside LeIsaac caused infinite loop.


## Summary

Summary of why LeRobot is causing Issues:

LeIsaac is using **LeRobotDatasetCfg** which is a class inherited from ```lerobot.datasets.lerobot_dataset import LeRobotDataset``` to build a dictionary with all the metadata to build the IsaacLab Environment.


```python
@configclass
class LeRobotDatasetCfg:
    """Configuration for the LeRobotDataset."""

    repo_id: str = None
    """Lerobot Dataset repository ID."""
    fps: int = 30
    """Lerobot Dataset frames per second."""
    robot_type: str = "so101_follower"
    """Robot type: so101_follower or bi_so101_follower, etc."""
    features: dict = None
    """Features for the LeRobotDataset."""
    action_align: bool = False
    """Whether the action shape equals to the joint number. If action align, we will convert action to lerobot limit range."""
```    

## Tracing the Issue - LEROBOT

**Step - 1**

File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 60

```python
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
```

**Step - 2**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/__init__.py", line 11

```python
"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *
from .utils import monkey_patch

```

**Step - 3**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/__init__.py", line 17

```python
"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
```

**Step - 4**

Now starts running Isaaclab stuff.

IsaacLab tasks has an importer util that is recursively importing all the required packages into the Python Wheel so it can leverage the Gym Wrapper to instantiate the envs


File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 40

This line has the Issue:

```python
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass
```


```python
"""Sub-module with utility for importing all modules in a package recursively."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    """Import all sub-packages in a package recursively.

    It is easier to use this function to import all sub-packages in a package recursively
    than to manually import each sub-package.

    It replaces the need of the following code snippet on the top of each package's ``__init__.py`` file:

    .. code-block:: python

        import .locomotion.velocity
        import .manipulation.reach
        import .manipulation.lift

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package itself
    package = importlib.import_module(package_name)
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass
```


**Step - 5**

File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 84


The line ```__import__(info.name)``` inside

```python
__import__(info.name)
```

Inside here

```python
        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])
```

Part of the Internal Helper

```python
"""
Internal helpers.
"""


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
        the ``blacklist_pkgs`` argument to skip blacklisted packages. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.

    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p: str, m: dict[str, bool] = {}) -> bool:
        """Check if a package has been seen before."""
        if p in m:
            return True
        m[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        # check blacklisted
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)
```


**Step - 6**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/__init__.py", line 1

```python
from .bi_arm_env_cfg import (
    BiArmObservationsCfg,
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmTerminationsCfg,
)
from .direct.bi_arm_env import BiArmTaskDirectEnv, BiArmTaskDirectEnvCfg
from .direct.single_arm_env import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .lekiwi_env_cfg import (
    LeKiwiActionsCfg,
    LeKiwiEventCfg,
    LeKiwiObservationsCfg,
    LeKiwiRewardsCfg,
    LeKiwiTaskEnvCfg,
    LeKiwiTaskSceneCfg,
    LeKiwiTerminationsCfg,
)
from .single_arm_env_cfg import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
```

**Step - 7**

Here will call a dependency that needs LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader which have LeRobot Implemented because it is made to generate a Dataset compatible with LeRobot.

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/bi_arm_env_cfg.py", line 22

```python
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
```


**Step - 8**

The issue starts by depending on these: LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/__init__.py", line 4


```python
from .device_base import DeviceBase
from .gamepad import SO101Gamepad
from .keyboard import SO101Keyboard
from .lekiwi import LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader # this line
from .lerobot import BiSO101Leader, SO101Leader
```

**Step - 9**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/__init__.py", line 1

```python
from .lekiwi_gamepad import LeKiwiGamepad # This line
from .lekiwi_keyboard import LeKiwiKeyboard
from .lekiwi_leader import LeKiwiLeader
```

**Step - 10**

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/lekiwi_gamepad.py", line 4

```python
from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env
```

The function **convert_lekiwi_wheel_action_robot2env** from robot_utils is trying to return state which is required by IsaacLab.

```python
    def get_device_state(self):
        arm_action = super().get_device_state()

        wheel_action_user = torch.tensor(self._vel_command, device=self.env.device).repeat(self.env.num_envs, 1)

        robot_base_theta = self.env.scene["robot"].data.joint_pos[:, self._joint_names.index("base_theta")]
        wheel_action_world = convert_lekiwi_wheel_action_robot2env(wheel_action_user, robot_base_theta)[0]
        wheel_action_world = wheel_action_world.cpu().numpy()

        return np.concatenate([arm_action, wheel_action_world])
```



**Step - 11**

**robot_utils** Depends on the Dataset Handler Which is trying to import LeRobot

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/utils/robot_utils.py", line 12

```python
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
```

It is using the **LeRobotDatasetCfg** to build the dictionary with features necessary for IsaacLab.

```python
def build_feature_from_env(env: ManagerBasedEnv | DirectRLEnv, dataset_cfg: LeRobotDatasetCfg) -> dict:
    """
    Build the feature from the environment.
    """
    features = {}

    default_feature_joint_names = env.cfg.default_feature_joint_names
    if isinstance(env, ManagerBasedEnv):
        action_dim = env.action_manager.total_action_dim
    else:
        action_dim = env.actions.shape[-1]

    if action_dim != len(default_feature_joint_names):
        # [A bit tricky, currently works because the action dimension matches the joints only when we use leader control]
        action_joint_names = [f"dim_{index}" for index in range(action_dim)]
        dataset_cfg.action_align = False
    else:
        action_joint_names = default_feature_joint_names
        dataset_cfg.action_align = True
    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(action_dim,), names=action_joint_names))
    features["observation.state"] = asdict(
        StateFeatureItem(dtype="float32", shape=(len(default_feature_joint_names),), names=default_feature_joint_names)
    )

    for camera_key, camera_sensor in env.scene.sensors.items():
        if isinstance(camera_sensor, Camera):
            height, width = camera_sensor.image_shape
            video_feature_item = VideoFeatureItem(
                dtype="video", shape=[height, width, 3], names=["height", "width", "channels"]
            )
            video_feature_item.video_info["video.height"] = height
            video_feature_item.video_info["video.width"] = width
            video_feature_item.video_info["video.fps"] = dataset_cfg.fps
            features[f"observation.images.{camera_key}"] = asdict(video_feature_item)

    return features
```

My guess the features dictionary will look like this:

| Key                       | Data Type | Description                                      |
| ------------------------- | --------- | ------------------------------------------------ |
| action                    | float32   | The control signals sent to the robot.           |
| observation.state         | float32   | The actual joint positions/velocities.           |
| observation.images.{name} | video     | Compressed video stream from each camera sensor. |


**Step - 12**

Here we have the culprit: ModuleNotFoundError: No module named 'lerobot'

File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/enhance/datasets/lerobot_dataset_handler.py", line 6

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```



```bash
Traceback (most recent call last):
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 60, in <module>
    from leisaac.utils.env_utils import (
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/__init__.py", line 11, in <module>
    from .tasks import *
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/__init__.py", line 17, in <module>
    import_packages(__name__, _BLACKLIST_PKGS)
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 40, in import_packages
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py", line 84, in _walk_packages
    __import__(info.name)
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/__init__.py", line 1, in <module>
    from .bi_arm_env_cfg import (
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/tasks/template/bi_arm_env_cfg.py", line 22, in <module>
    from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/__init__.py", line 4, in <module>
    from .lekiwi import LeKiwiGamepad, LeKiwiKeyboard, LeKiwiLeader
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/__init__.py", line 1, in <module>
    from .lekiwi_gamepad import LeKiwiGamepad
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/devices/lekiwi/lekiwi_gamepad.py", line 4, in <module>
    from leisaac.utils.robot_utils import convert_lekiwi_wheel_action_robot2env
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/utils/robot_utils.py", line 12, in <module>
    from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/enhance/datasets/lerobot_dataset_handler.py", line 6, in <module>
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
ModuleNotFoundError: No module named 'lerobot'
```

## Tracing issue with ZMQ



```bash
Creating window for environment.
[INFO]: Completed setting up the environment...
Traceback (most recent call last):
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 281, in <module>
    main()
  File "/home/goat/Documents/GitHub/renanmb/leisaac/scripts/evaluation/policy_inference.py", line 179, in main
    policy = Gr00t16ServicePolicyClient(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/policy/service_policy_clients.py", line 109, in __init__
    super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
  File "/home/goat/Documents/GitHub/renanmb/leisaac/source/leisaac/leisaac/policy/base.py", line 54, in __init__
    self.context = zmq.Context()
                   ^^^
NameError: name 'zmq' is not defined
```